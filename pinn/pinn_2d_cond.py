from pinn.model import *
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
    model = PinnCondConv2D(
        depths=depths,
        Lx=Lx,
        Ly=Ly,
        Nx_train=Nx_train,
        Ny_train=Ny_train,
        phi_boundary=phi_theory_np,
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
