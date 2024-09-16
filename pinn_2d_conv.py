from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from model import *
from pinn_2d_cond import gen_square_domain

THIS_PATH = Path(__file__).parent.resolve()


class Txy2:
    def __init__(
        self,
        kappa: float = 1,
        T0: float = 0,
        T1: float = 1,
        u: float = 1,
        theta_deg: float = 45,
        x_max: float = 1,
        y_max: float = 1,
        numpy_based: bool = True,
    ) -> None:
        self.kappa = kappa
        self.box = np if numpy_based else torch
        theta = theta_deg / 180 * np.pi
        self.ux = u * np.cos(theta)
        self.uy = u * np.sin(theta)
        self.z_max = (self.ux * x_max + self.uy * y_max) / kappa
        self.beta = (T1 - T0) / (1 - self.box.exp(-self.z_max))
        self.alpha = T1 - self.beta

    def __call__(self, x: float, y: float) -> float:
        z = (self.ux * x + self.uy * y) / self.kappa
        return self.alpha + self.beta * self.box.exp(z - self.z_max)


def pinn_2d_conv() -> None:
    Lx = 1.0
    Ly = 1.0
    T0 = 0.0
    T1 = 1.0
    kappa = 1.0
    u = 2.0
    theta_deg = 30.0
    T_theory_np = Txy2(
        kappa=kappa,
        T0=T0,
        T1=T1,
        u=u,
        theta_deg=theta_deg,
        x_max=Lx,
        y_max=Ly,
        numpy_based=True,
    )
    title = "ux Tx + uy Ty = Txx + Tyy, T_theory = alpha + beta * exp(z - z_max)"
    title += "z = (ux * x + uy * y) / kappa"
    title += "T(0,0) = T0, T(x_max, y_max) = T1"
    title += f"\nT0 = {T0}, T1 = {T1}, u = {u}, theta_deg = {theta_deg}, kappa = {kappa}, Lx = {Lx}, Ly = {Ly}"
    # ===Parameters================================
    epochs = 10000
    lr = 0.0001
    depths = [64, 64]
    act_func_type = ACT_FUNCT_TYPE.TANH
    Nx_train = 101
    Ny_train = 101
    title += f"\nepochs = {epochs}, lr = {lr}, depths = {depths}, N_train = {(Nx_train , Ny_train)}, sigma = {act_func_type}"
    # ====Training==================================
    torch.manual_seed(731)
    model = SeqModel(
        in_features=2, depths=depths, out_features=1, act_func_type=act_func_type
    )
    file_name = "model_weights_cond_conv.pth"
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
        theta = theta_deg / 180 * np.pi
        ux = u * np.cos(theta)
        uy = u * np.sin(theta)
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
            residual = kappa * (Txx + Tyy) - (ux * Tx + uy * Ty)
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
    title += f", max_abs_error = {max_err}"
    print(f"max_err = {max_err}")
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
    pinn_2d_conv()

# py -m FVM_PINN_workshop.pinn_2d_conv
