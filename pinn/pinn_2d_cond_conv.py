from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from pinn.model import *

"""
Practice runs on physics-informed neural networks (PINN) for solving conduction-convection 
conservation equtaion in a two-dimensional rectangular domain.

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


class Txy2:
    """
    A functor for the function:

        φ(x, y) = alpha + beta * exp(z - z_max),

    where

        z = u * (x * cos(theta) + y * sin(theta)) / kappa

    """

    def __init__(
        self,
        kappa: float = 1,
        phi0: float = 0,
        phi1: float = 1,
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
        self.beta = (phi1 - phi0) / (1 - self.box.exp(-self.z_max))
        self.alpha = phi1 - self.beta

    def __call__(self, x: float, y: float) -> float:
        z = (self.ux * x + self.uy * y) / self.kappa
        return self.alpha + self.beta * self.box.exp(z - self.z_max)


def pinn_2d_cond_conv() -> None:
    """
    Using PINN to solve the conduction-convection conservation equation,

    u ⋅ ∇φ = k Δφ,

    on a 2D rectangular domain. Here Δ is the Laplacian operator, k is diffusivity,
    and φ is the conserved quantity, ∇ is the graident operator and u is the 2D velocity field.

    The following Dirichlet boundary condition is enforced:

    φ(x,y| (x,y) ∈ domain boundary) = φ0 + φ1 exp((x * ux + y * uy) / k - argmax)

    where argmax = (Lx * ux + Ly * uy) / k.
    """
    Lx = 1.0
    Ly = 1.0
    phi0 = 0.0
    phi1 = 1.0
    kappa = 1.0
    u = 2.0
    theta_deg = 30.0
    theta = theta_deg / 180 * np.pi
    ux = u * np.cos(theta)
    uy = u * np.sin(theta)
    phi_theory_np = Txy2(
        kappa=kappa,
        phi0=phi0,
        phi1=phi1,
        u=u,
        theta_deg=theta_deg,
        x_max=Lx,
        y_max=Ly,
        numpy_based=True,
    )
    title = "ux φ,x + uy φ,y = φ,xx + φ,yy, φ_theory = alpha + beta * exp(z - z_max), "
    title += "z = (ux * x + uy * y) / kappa, "
    title += "φ(0,0) = φ, φ(x_max, y_max) = φ1, "
    title += f"\nφ0 = {phi0}, φ1 = {phi1}, u = {u}, theta_deg = {theta_deg}, kappa = {kappa}, Lx = {Lx}, Ly = {Ly}"
    # ===Parameters================================
    epochs = 100
    lr = 0.0001
    depths = [64, 64]
    Nx_train = 101
    Ny_train = 101
    title += f"\nepochs = {epochs}, lr = {lr}, depths = {depths}, N_train = {(Nx_train , Ny_train)}"
    # ====Training==================================
    torch.manual_seed(731)
    model = PinnCondConv2D(
        depths=depths,
        Lx=Lx,
        Ly=Ly,
        Nx_train=Nx_train,
        Ny_train=Ny_train,
        phi_theory_np=phi_theory_np,
        kappa=kappa,
        ux=ux,
        uy=uy,
    )
    file_name = "model_weights_2d_cond_conv.pth"
    model_file = THIS_PATH / file_name
    duration_mins = 0
    if model_file.exists():
        model.net.load_state_dict(torch.load(model_file))
    else:

        t_start = time.time()
        model.train_pinn(lr=lr, epochs=epochs)
        t_end = time.time()
        duration_mins = (t_end - t_start) / 60
        title += f" , training time = {duration_mins} min"
        torch.save(model.net.state_dict(), model_file)
    # ====Error evaluation=============================
    with torch.no_grad():
        max_err = 0
        xy = model.XY.detach().numpy()
        phi = model.net(model.XY)
        phi_exact = phi_theory_np(xy[:, 0], xy[:, 1]).reshape(-1, 1)
        phi_pred = phi.detach().numpy()
        max_err = np.max(np.fabs(phi_exact - phi_pred))
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
    pinn_2d_cond_conv()

# py -m pinn.pinn_2d_cond_conv
