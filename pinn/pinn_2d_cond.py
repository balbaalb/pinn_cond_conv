from pinn.model import *
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

"""
Practice runs on physics-informed neural networks (PINN) for solving conduction only 
conservation equtaion in a 2D rectangular domain.

Refs: 
    Raissi, M., Perdikaris, P. and Karniadakis, G.E., 2019. Physics-informed neural networks: 
    A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. 
    Journal of Computational physics, 378, pp.686-707.

    Cai, S., Wang, Z., Wang, S., Perdikaris, P. and Karniadakis, G.E., 2021. 
    Physics-informed neural networks for heat transfer problems. Journal of Heat Transfer, 143(6), p.060801.

    Cai, Z., Chen, J. and Liu, M., 2021. Least-squares ReLU neural network (LSNN) 
    method for linear advection-reaction equation. Journal of Computational Physics, 443, p.110514.
"""


THIS_PATH = Path(__file__).parent.resolve()


def gen_square_domain(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    boundary_value: Callable[[float, float], float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates PyTorch domain nodes and boundary condition training data for a PINN for a
    rectangular domain.
    Inputs:
        - Lx: Horizontal side length of the rectangular domain
        - Ly: Vertical side length of the rectangular domain
        - Nx: Number of nodes on a horizntal row of nodes
        - Ny: Number of nodes on a vertical column of nodes
        - boundary_value: a function (x ,y) returning the Dirichlet boundary condition on the
            boundary node (x, y). This function should be numpy based.
    Outputs: (X, Y, XY, XY_boundary, phi_boundary)
        - X: A torch tensor with shape (Nx * Ny , 1) containing x coordinate of domain nodes
        - Y: A torch tensor with shape (Nx * Ny , 1) containing y coordinate of domain nodes
        - XY: A torch tensor with shape (Nx * Ny , 2) containing (x,y) coordinate of domain nodes,
            this is just X and Y outputs concatenated.
        - XY_boundary: A torch tensor with shape (2 * Nx + 2 * Ny , 2) containing (x , y) coordinate
            of boundary
        - phi_boundary: A torch tensor with shape (2 * Nx + 2 * Ny , 1) containing Dirichlet boundary
            condition values for points in XY_boundary.

    """
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    xx = np.repeat(x, Ny)
    yy = np.tile(y, Nx)
    X = torch.FloatTensor(xx).reshape(-1, 1)
    X.requires_grad = True
    Y = torch.FloatTensor(yy).reshape(-1, 1)
    Y.requires_grad = True
    XY = torch.cat([X, Y], dim=1)

    xe = np.zeros(Ny) + Lx
    ye = np.linspace(0, Ly, Ny)
    xw = np.zeros(Ny)
    yw = np.linspace(Ly, 0, Ny)
    xn = np.linspace(Lx, 0, Nx)
    yn = np.zeros(Nx) + Ly
    xs = np.linspace(0, Lx, Nx)
    ys = np.zeros(Nx)
    x_boundary = np.concatenate((xs[:-1], xe[:-1], xn[:-1], xw[:-1])).reshape(-1, 1)
    y_boundary = np.concatenate((ys[:-1], ye[:-1], yn[:-1], yw[:-1])).reshape(-1, 1)
    xy_boundary = np.concatenate((x_boundary, y_boundary), axis=1)
    XY_boundary = torch.FloatTensor(xy_boundary)
    if boundary_value is None:
        return X, Y, XY, XY_boundary
    phi_boundary_np = boundary_value(x_boundary, y_boundary)
    phi_boundary = torch.FloatTensor(phi_boundary_np)
    return X, Y, XY, XY_boundary, phi_boundary


class Model_Phi_XY1(nn.Module):
    """
    provides a feed forward neural network that gets (x, y) coordintaes
    and outputs

    φ(x,y) = φ0 + φ1 * sin(alpha_x * x + beta_x) * exp(alpha_y * y + beta_y)

    This is used for testing the PINN solver training method.

    Note that if alpha_x = ± alpha_y, then

    ΔT = ∂^2T/∂x^2 + ∂^2T/∂y^2 = 0.

    """

    def __init__(
        self,
        alpha_x: float = 1,
        beta_x: float = 0,
        alpha_y: float = -1,
        beta_y: float = 0,
        phi0: float = 0,
        phi1: float = 1,
    ) -> None:
        super().__init__()
        alpha_x *= np.pi
        alpha_y *= np.pi
        self.nx = nn.Linear(in_features=2, out_features=1)
        self.ny = nn.Linear(in_features=2, out_features=1)
        self.out = nn.Bilinear(in1_features=1, in2_features=1, out_features=1)
        self.nx.weight = nn.Parameter(torch.FloatTensor([[alpha_x, 0.0]]))
        self.nx.bias = nn.Parameter(torch.FloatTensor([beta_x]))
        self.ny.weight = nn.Parameter(torch.FloatTensor([[0, alpha_y]]))
        self.ny.bias = nn.Parameter(torch.FloatTensor([beta_y]))
        self.out.weight = nn.Parameter(torch.FloatTensor([[[phi1]]]))
        self.out.bias = nn.Parameter(torch.FloatTensor([phi0]))

    def forward(self, x):
        zx = torch.sin(self.nx(x))
        zy = torch.exp(self.ny(x))
        return self.out(zx, zy)


class Phi_xy1:
    """
    A functor for the function

    φ(x,y) = φ0 + φ1 * sin(alpha_x * x + beta_x) * exp(alpha_y * y + beta_y)

    Note that if alpha_x = ± alpha_y, then

    Δφ = ∂^2φ/∂x^2 + ∂^2φ/∂y^2 = 0.

    """

    def __init__(
        self,
        phi0: float,
        phi1: float,
        alpha_x: float,
        alpha_y: float,
        beta_x: float,
        beta_y: float,
        numpy_based: bool = True,
    ) -> None:
        self.phi0 = phi0
        self.phi1 = phi1
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.numpy_based = numpy_based

    def __call__(self, x: float, y: float) -> float:
        if self.numpy_based:
            return self.phi0 + self.phi1 * np.sin(
                self.alpha_x * np.pi * x + self.beta_x
            ) * np.exp(self.alpha_y * np.pi * y + self.beta_y)
        else:
            return self.phi0 + self.phi1 * torch.sin(
                self.alpha_x * torch.pi * x + self.beta_x
            ) * torch.exp(self.alpha_y * torch.pi * y + self.beta_y)


class ProblemType(Enum):
    """
    Flags for type of network that should be used in pinn_2d_cond() function below.
        PDE_CONDUCTION: Train a NN that provides the exact solution to cond-conv PDE.
        MODEL_PHI_XY1 : Use the network in class Model_Phi_XY1 which outputs the analytical
            solution to the PDE.

    """

    PDE_CONDUCTION = auto()
    MODEL_PHI_XY1 = auto()


class PinnCond2D:
    """
    Creates a pinn model for the equation

    k Δφ = 0,

    on a 2D rectangular domain. Here Δ is the Laplacian operator, k is diffusivity,
    and φ is the conserved quantity.
    """

    def __init__(
        self,
        depths: list[int],
        Lx: float,
        Ly: float,
        Nx_train: int,
        Ny_train: int,
        phi_theory_np: Callable[[float, float], float],
        kappa: float,
    ) -> None:
        self.net = SeqModel(in_features=2, depths=depths, out_features=1)
        self.X, self.Y, self.XY, self.XY_boundary, self.phi_boundary = (
            gen_square_domain(
                Nx=Nx_train, Ny=Ny_train, Lx=Lx, Ly=Ly, boundary_value=phi_theory_np
            )
        )
        self.kappa = kappa
        self.losses = []
        self.epoch = 0

    def __pde_residual(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Returns k Δφ

        """
        phi_x = torch.autograd.grad(
            phi, self.X, torch.ones_like(phi), create_graph=True, retain_graph=True
        )[0]
        phi_xx = torch.autograd.grad(
            phi_x, self.X, torch.ones_like(phi_x), create_graph=True, retain_graph=True
        )[0]
        phi_y = torch.autograd.grad(
            phi, self.Y, torch.ones_like(phi), create_graph=True, retain_graph=True
        )[0]
        phi_yy = torch.autograd.grad(
            phi_y, self.Y, torch.ones_like(phi_y), create_graph=True, retain_graph=True
        )[0]
        residual = self.kappa * (phi_xx + phi_yy)
        return residual

    def __loss_pinn(self) -> torch.Tensor:
        """
        Returns Loss function for training: residual of PDE + loss of deciation from the given boundary condions
        """
        self.epoch += 1
        phi = self.net(self.XY)
        residual = self.__pde_residual(phi=phi)
        loss_pde = self.criterion(residual, torch.zeros_like(residual))
        phi_pred_boundary = self.net(self.XY_boundary)
        loss_bc = self.criterion(phi_pred_boundary, self.phi_boundary)
        loss = loss_pde + loss_bc
        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        if (self.epoch) % 100 == 0:
            print(
                f"Epoch = {self.epoch}, loss = {loss.item()} = {loss_pde.item()} (PDE) + {loss_bc.item()} (BC)"
            )
        return loss

    def train_pinn(self, lr: float, epochs: int) -> None:
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        for _ in range(epochs):
            self.optimizer.step(self.__loss_pinn)
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.optimizer.step(self.__loss_pinn)
        if len(self.losses) > 0:
            plt.plot(self.losses)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.show()


def pinn_2d_cond(mode: ProblemType) -> None:
    """
    Solves the conduction only conservation equation

    Δφ = ∂^2φ/∂x^2 + ∂^2φ/∂y^2 = 0,

    using PINN. on a rectangular domain. Here Δ is the Laplacian operator, and φ is the conserved quantity
    """
    # ===Problem===================================
    Lx = 1.0
    Ly = 1.0
    phi0 = 0.0
    phi1 = 1.0
    alpha_x = 1 / Lx
    alpha_y = -alpha_x  # to satisfy phi_xx + phi_yy = 0 => alpha_x = ± alpha_y
    beta_x = 0
    beta_y = 0
    phi_theory_np = Phi_xy1(
        phi0=phi0,
        phi1=phi1,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        beta_x=beta_x,
        beta_y=beta_y,
        numpy_based=True,
    )
    title = "φxx + φyy = 0, φ_theory = φ0 + φ1 * sin(ax * x + beta_x) * exp(-ay * y + beta_y)"
    title += f"\nφ0 = {phi0}, φ1 = {phi1}, alpha_x = {alpha_x}, alpha_y = {alpha_y}, beta_x = {beta_x}, beta_y = {beta_y}"
    # ===Parameters================================
    epochs = 1 if mode == ProblemType.MODEL_PHI_XY1 else 100
    lr = 0.001
    depths = [64, 64]
    act_func_type = ACT_FUNCT_TYPE.TANH
    Nx_train = 101
    Ny_train = 101
    title += f"\nepochs = {epochs}, lr = {lr}, depths = {depths}, N_train = {(Nx_train , Ny_train)}, sigma = {act_func_type}"
    # ====Training==================================
    torch.manual_seed(72)
    model = PinnCond2D(
        depths=depths,
        Lx=Lx,
        Ly=Ly,
        Nx_train=Nx_train,
        Ny_train=Ny_train,
        phi_theory_np=phi_theory_np,
        kappa=1.0,
    )
    if mode == ProblemType.MODEL_PHI_XY1:
        model.net = Model_Phi_XY1(
            alpha_x=alpha_x,
            beta_x=beta_x,
            alpha_y=alpha_y,
            beta_y=beta_y,
            phi0=phi0,
            phi1=phi1,
        )
    file_name = "model_weights_2d_cond.pth"
    model_file = THIS_PATH / file_name
    duration_mins = 0
    if model_file.exists() and mode == ProblemType.PDE_CONDUCTION:
        model.net.load_state_dict(torch.load(model_file))
    else:
        t_start = time.time()
        model.train_pinn(lr=lr, epochs=epochs)
        torch.save(model.net.state_dict(), model_file)
        t_end = time.time()
        duration_mins = (t_end - t_start) / 60
        title += f" , training time = {round(duration_mins , 3)} min"
    # ====Error evaluation=============================
    with torch.no_grad():
        max_err = 0
        xy = model.XY.detach().numpy()
        phi = model.net(model.XY)
        phi_exact = phi_theory_np(xy[:, 0], xy[:, 1]).reshape(-1, 1)
        phi_pred = phi.detach().numpy()
        max_err = np.max(np.fabs(phi_exact - phi_pred))
        print(f"max_err = {max_err}")

        title += f"\nmax_abs_error = {round(max_err , 5)}"
        N_diag = 101
        xy_antidiag = np.concatenate(
            (
                np.linspace(0, Lx, N_diag).reshape(-1, 1),
                np.linspace(0, Ly, N_diag).reshape(-1, 1),
            ),
            axis=1,
        )
        XY_antidiag = torch.FloatTensor(xy_antidiag)
        phi_antidiag_exact = phi_theory_np(xy_antidiag[:, 0], xy_antidiag[:, 1])
        phi_antidiag_pred = model.net(XY_antidiag)
        phi_antidiag_pred = phi_antidiag_pred.detach().numpy()

        xy_diag = np.concatenate(
            (
                np.linspace(0, Lx, N_diag).reshape(-1, 1),
                np.linspace(Ly, 0, N_diag).reshape(-1, 1),
            ),
            axis=1,
        )
        XY_diag = torch.FloatTensor(xy_diag)
        phi_diag_exact = phi_theory_np(xy_diag[:, 0], xy_diag[:, 1])
        phi_diag_pred = model.net(XY_diag)
        phi_diag_pred = phi_diag_pred.detach().numpy()
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.title(title)
    plt.plot(np.linspace(0.0, 1.0, N_diag), phi_antidiag_exact, label="φ_exact")
    plt.plot(
        np.linspace(0.0, 1.0, N_diag),
        phi_antidiag_pred,
        label="φ_PINN",
        linestyle="dashed",
    )
    plt.xlabel("non-dim distance along anti-diagonal")
    plt.ylabel("φ")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0.0, 1.0, N_diag), phi_diag_exact, label="φ_exact")
    plt.plot(
        np.linspace(0.0, 1.0, N_diag), phi_diag_pred, label="φ_PINN", linestyle="dashed"
    )
    plt.xlabel("non-dim distance along diagonal")
    plt.ylabel("φ")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pinn_2d_cond(mode=ProblemType.PDE_CONDUCTION)

# py -m pinn.pinn_2d_cond
