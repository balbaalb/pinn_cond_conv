from model import *
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

THIS_PATH = Path(__file__).parent.resolve()


def gen_square_domain(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    boundary_value: Callable[[float, float], float] = None,
):
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
    T_boundary_np = boundary_value(x_boundary, y_boundary)
    T_boundary = torch.FloatTensor(T_boundary_np)
    return X, Y, XY, XY_boundary, T_boundary


class Model_TXY1(nn.Module):
    # provides T(x,y) = T0 + T1 * sin(alpha_x * x + beta_x) * exp(alpha_y * y + beta_y)
    def __init__(
        self,
        alpha_x: float = 1,
        beta_x: float = 0,
        alpha_y: float = -1,
        beta_y: float = 0,
        T0: float = 0,
        T1: float = 1,
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
        self.out.weight = nn.Parameter(torch.FloatTensor([[[T1]]]))
        self.out.bias = nn.Parameter(torch.FloatTensor([T0]))

    def forward(self, x):
        zx = torch.sin(self.nx(x))
        zy = torch.exp(self.ny(x))
        return self.out(zx, zy)


class Txy1:
    def __init__(
        self,
        T0: float,
        T1: float,
        alpha_x: float,
        alpha_y: float,
        beta_x: float,
        beta_y: float,
        numpy_based: bool = True,
    ) -> None:
        self.T0 = T0
        self.T1 = T1
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.numpy_based = numpy_based

    def __call__(self, x: float, y: float) -> float:
        if self.numpy_based:
            return self.T0 + self.T1 * np.sin(
                self.alpha_x * np.pi * x + self.beta_x
            ) * np.exp(self.alpha_y * np.pi * y + self.beta_y)
        else:
            return self.T0 + self.T1 * torch.sin(
                self.alpha_x * torch.pi * x + self.beta_x
            ) * torch.exp(self.alpha_y * torch.pi * y + self.beta_y)


class ProblemType(Enum):
    PDE_CONDUCTION = auto()
    MODEL_TXY1 = auto()


def pinn_2d_cond() -> None:
    mode = ProblemType.PDE_CONDUCTION
    # ===Problem===================================
    Lx = 1.0
    Ly = 1.0
    T0 = 0.0
    T1 = 1.0
    alpha_x = 1 / Lx
    alpha_y = -alpha_x  # to satisfy Txx + Tyy = 0 => alpha_x = ± alpha_y
    beta_x = 0
    beta_y = 0
    T_theory_np = Txy1(
        T0=T0,
        T1=T1,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        beta_x=beta_x,
        beta_y=beta_y,
        numpy_based=True,
    )
    title = "Txx + Tyy = 0, T_theory = T0 + T1 * sin(ax * x + beta_x) * exp(-ay * y + beta_y)"
    title += f"\nT0 = {T0}, T1 = {T1}, alpha_x = {alpha_x}, alpha_y = {alpha_y}, beta_x = {beta_x}, beta_y = {beta_y}"
    # ===Parameters================================
    epochs = 1 if mode == ProblemType.MODEL_TXY1 else 10000
    lr = 0.001
    depths = [64, 64]
    act_func_type = ACT_FUNCT_TYPE.TANH
    Nx_train = 101
    Ny_train = 101
    title += f"\nepochs = {epochs}, lr = {lr}, depths = {depths}, N_train = {(Nx_train , Ny_train)}, sigma = {act_func_type}"
    # ====Training==================================
    torch.manual_seed(72)
    if mode == ProblemType.MODEL_TXY1:
        model = Model_TXY1(
            alpha_x=alpha_x, beta_x=beta_x, alpha_y=alpha_y, beta_y=beta_y, T0=T0, T1=T1
        )
    else:
        model = SeqModel(
            in_features=2, depths=depths, out_features=1, act_func_type=act_func_type
        )
    file_name = "model_weights_lap_orig." + str(mode) + ".pth"
    model_file = THIS_PATH / file_name
    X, Y, XY, XY_boundary, T_boundary = gen_square_domain(
        Nx=Nx_train, Ny=Ny_train, Lx=Lx, Ly=Ly, boundary_value=T_theory_np
    )
    duration_mins = 0
    losses = []
    if model_file.exists():
        model.load_state_dict(torch.load(model_file))
    else:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        t_start = time.time()
        for i in range(epochs):
            T = model(XY)
            Tx = torch.autograd.grad(
                T, X, torch.ones_like(T), create_graph=True, retain_graph=True
            )[0]
            Txx = torch.autograd.grad(
                Tx, X, torch.ones_like(Tx), create_graph=True, retain_graph=True
            )[0]
            Ty = torch.autograd.grad(
                T, Y, torch.ones_like(T), create_graph=True, retain_graph=True
            )[0]
            Tyy = torch.autograd.grad(
                Ty, Y, torch.ones_like(Ty), create_graph=True, retain_graph=True
            )[0]
            residual = Txx + Tyy
            loss_pde = criterion(residual, torch.zeros_like(residual))
            T_pred_boundary = model(XY_boundary)
            loss_bc = criterion(T_pred_boundary, T_boundary)
            loss = loss_pde + loss_bc
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch = {i+1}, loss = {loss.item()} = {loss_pde.item()} + {loss_bc.item()}"
                )
        torch.save(model.state_dict(), model_file)
        t_end = time.time()
        duration_mins = (t_end - t_start) / 60
        title += f" , training time = {duration_mins} min"
        title += f"\nloss = {loss.item()}, loss_pde = {loss_pde.item()}, loss_bc = {loss_bc.item()}"
    # ====Error evaluation=============================
    max_err = 0
    xy = XY.detach().numpy()
    T = model(XY)
    T_exact = T_theory_np(xy[:, 0], xy[:, 1]).reshape(-1, 1)
    T_pred = T.detach().numpy()
    max_err = np.max(np.fabs(T_exact - T_pred))
    print(f"max_err = {max_err}")
    title += f"\nmax_abs_error = {max_err}"
    N_diag = 101
    xy_antidiag = np.concatenate(
        (
            np.linspace(0, Lx, N_diag).reshape(-1, 1),
            np.linspace(0, Ly, N_diag).reshape(-1, 1),
        ),
        axis=1,
    )
    XY_antidiag = torch.FloatTensor(xy_antidiag)
    T_antidiag_exact = T_theory_np(xy_antidiag[:, 0], xy_antidiag[:, 1])
    T_antidiag_pred = model(XY_antidiag)
    T_antidiag_pred = T_antidiag_pred.detach().numpy()

    xy_diag = np.concatenate(
        (
            np.linspace(0, Lx, N_diag).reshape(-1, 1),
            np.linspace(Ly, 0, N_diag).reshape(-1, 1),
        ),
        axis=1,
    )
    XY_diag = torch.FloatTensor(xy_diag)
    T_diag_exact = T_theory_np(xy_diag[:, 0], xy_diag[:, 1])
    T_diag_pred = model(XY_diag)
    T_diag_pred = T_diag_pred.detach().numpy()
    plt.subplot(3, 1, 1)
    plt.title(title)
    plt.plot(np.linspace(0.0, 1.0, N_diag), T_antidiag_exact, label="T_exact")
    plt.plot(
        np.linspace(0.0, 1.0, N_diag),
        T_antidiag_pred,
        label="T_PINN",
        linestyle="dashed",
    )
    plt.xlabel("non-dim distance along anti-diagonal")
    plt.ylabel("T")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(0.0, 1.0, N_diag), T_diag_exact, label="T_exact")
    plt.plot(
        np.linspace(0.0, 1.0, N_diag), T_diag_pred, label="T_PINN", linestyle="dashed"
    )
    plt.xlabel("non-dim distance along diagonal")
    plt.ylabel("T")
    plt.legend()

    if len(losses) > 0:
        plt.subplot(3, 1, 3)
        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    pinn_2d_cond()

# py -m FVM_PINN_workshop.pinn_2d_cond
