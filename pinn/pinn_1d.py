import numpy as np
import matplotlib.pyplot as plt
from pinn.model import *

"""
Practice runs on solving 1D conduction-convection equation using physics-informed neural network (PINN).

Refs: 
    Raissi, M., Perdikaris, P. and Karniadakis, G.E., 2019. Physics-informed neural networks: 
    A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. 
    Journal of Computational physics, 378, pp.686-707.

    Cai, S., Wang, Z., Wang, S., Perdikaris, P. and Karniadakis, G.E., 2021. 
    Physics-informed neural networks for heat transfer problems. Journal of Heat Transfer, 143(6), p.060801.

    Cai, Z., Chen, J. and Liu, M., 2021. Least-squares ReLU neural network (LSNN) 
    method for linear advection-reaction equation. Journal of Computational Physics, 443, p.110514.

"""


def pinn_1d_cond():
    """
    Solving 1D conduction equation using PINN.
    Equation:
        d(k dφ/dx)/dx = -q, const k, φ(0) = φ0, φ(Lx) = φ1
    One possible solution, that is examined here:
        φ = -q x^2 / (2k) + C1 x + φ0
    with
        C1 = (φ1 - φ0) / Lx + q^2/2/k*Lx

    Constraint:
        0 <= φ0, φ1 <= φ1
    """
    title = "d(k dφ/dx)/dx = -q, const k, φ(0) = φ0, φ(Lx) = φ1"
    k = 0.5
    Lx = 1.0
    phi0 = 0.2
    phi1 = 1.0
    q = -5
    title += f"\nk = {k}, q = {q}, φ0 = {phi0}, φ1 = {phi1}, Lx = {Lx}"
    # ==== Parameters ======
    epochs = 100
    lr = 0.01
    depths = [64, 64, 64]
    N_train = 1001
    # ======================
    torch.manual_seed(616)

    model = PinnCondConv1D(
        depths=depths,
        Lx=Lx,
        N_train=N_train,
        kappa=k,
        u=0,
        q=q,
        phi0=phi0,
        phi1=phi1,
    )
    model.train_pinn(lr=lr, epochs=epochs)
    with torch.no_grad():
        C1 = (phi1 - phi0) / Lx + q / 2 / k * Lx
        x0 = np.linspace(0.0, Lx, 313)
        y0 = -q * x0**2 / 2 / k + C1 * x0 + phi0
        x_test = torch.FloatTensor(x0).reshape(-1, 1)
        phi_pred = model.net(x_test)
        phi_pred = phi_pred.detach().numpy()
    plt.figure(figsize=(10, 10))
    plt.plot(x0, y0, label="exact solution")
    plt.plot(x0, phi_pred, label="PINN", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("φ")
    plt.title(title)
    plt.legend()
    plt.show()


def pinn_1d_cond_scaled():
    """
    This function does the same thing as pinn_1d_cond() (above) but without
    constarining boundary condition and thus the solution to remain
    between 0 and 1.
    Equation:
        d(k dφ/dx)/dx = -q, const k, φ(0) = φ0, φ(Lx) = φ1
    One possible solution that examined here:
        φ = -q x^2 / (2k) + C1 x + phi0
        C1 = (phi1 - phi0) / Lx + q^2/2/k*Lx
    """
    title = "d(k dφ/dx)/dx = -q, const k, φ(0) = φ0, φ(Lx) = φ1"
    k = 0.5
    Lx = 10.0
    phi0 = 0.2
    phi1 = 100.0
    q = -5
    title += f"\nk = {k}, q = {q},φ0 = {phi0}, φ1 = {phi1}, Lx = {Lx}"
    # ==== Parameters ====================================
    epochs = 100
    lr = 0.01
    depths = [64, 64, 64]
    N_train = 1001
    # =========================================
    torch.manual_seed(616)
    model = PinnCondConv1D(
        depths=depths,
        Lx=1.0,
        N_train=N_train,
        kappa=1.0,
        u=0,
        q=q * Lx**2 / (phi1 - phi0) / k,
        phi0=0.0,
        phi1=1.0,
    )
    model.train_pinn(lr=lr, epochs=epochs)
    N_test = 313
    C1 = (phi1 - phi0) / Lx + q / 2 / k * Lx
    x0 = np.linspace(0.0, Lx, N_test)
    y0 = -q * x0**2 / 2 / k + C1 * x0 + phi0
    with torch.no_grad():
        X = torch.linspace(0, 1.0, N_test).reshape(-1, 1)
        phi_pred = model.net(X)
        phi_pred = phi_pred.detach().numpy()
        phi_pred = phi_pred * (phi1 - phi0) + phi0
    plt.figure(figsize=(10, 10))
    plt.plot(x0, y0, label="exact solution")
    plt.plot(x0, phi_pred, label="PINN", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("φ")
    plt.title(title)
    plt.legend()
    plt.show()


def pinn_1d_cond_conv():
    """
    Solving 1D conduction-convection equation with PINN.
    Equation:
        u dφ/dx = k d^2φ/dx^2, const k & u, φ(0) = φ0, φ(Lx) = φ1
    One possible solution:
        φ(x) = φ0 + (φ1 - phi0)(exp(Pe(x/Lx - 1)) - exp(-Pe))
                            / (1 - exp(-Pe))

    where, Pe, is the Péclet number.

    Pe = u * Lx / k
    """
    Lx = 1.0
    Pe = 2.0
    phi0 = 0.0
    phi1 = 1.0
    title = "u dφ/dx = k d^2φ/dx^2, const k, φ(0) = φ0, φ(Lx) = φ1"
    title += f"\nφ0 = {phi0}, φ1 = {phi1}, Lx = {Lx}, Pe = {Pe}"
    # ===== PINN Parameters ==============
    epochs = 100
    lr = 0.0005
    depths = [64, 64, 64]
    N_train = 1001
    act_func_type = ACT_FUNCT_TYPE.EXP  # TANH loss was flattening at loss_ode = 0.22
    title += f"\nepochs = {epochs}, lr = {lr}, depths = {depths}, N_train = {N_train}, sigma = {act_func_type.name}"
    # =====================================
    phi_exact = lambda x: (
        phi0
        + (phi1 - phi0) * (np.exp(Pe * (x / Lx - 1)) - np.exp(-Pe)) / (1 - np.exp(-Pe))
        if Pe > 0
        else (np.exp(Pe * (x / Lx)) - 1) / (np.exp(Pe) - 1) if Pe < 0 else x / Lx
    )
    assert phi_exact(0) == phi0
    assert phi_exact(Lx) == phi1

    model = PinnCondConv1D(
        depths=depths,
        Lx=Lx,
        N_train=N_train,
        kappa=1.0,
        u=Pe,
        q=0,
        phi0=phi0,
        phi1=phi1,
    )
    model.train_pinn(lr=lr, epochs=epochs)
    N_test = 313
    with torch.no_grad():
        X = torch.linspace(0.0, Lx, N_test).reshape(-1, 1)
        phi_pred = model.net(X)
        x0 = X.detach().numpy()
        y0 = phi_exact(x0)
        phi_pred = phi_pred.detach().numpy()
    plt.figure(figsize=(10, 10))
    plt.plot(x0, y0, label="Exact solution")
    plt.plot(x0, phi_pred, label="PINN", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("φ")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pinn_1d_cond()
    pinn_1d_cond_scaled()
    pinn_1d_cond_conv()

# py -m pinn.pinn_1d
