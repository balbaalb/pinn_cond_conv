import numpy as np
import matplotlib.pyplot as plt
from pinn.model import *

"""
Practice runs on solving 1D conduction-convection equation using 
physics-informed neural network (PINN).
"""


def pinn_1d_cond():
    """
    Solving 1D conduction equation using PINN.
    Equation:
        d(k dT/dx)/dx = -q, const k, T(0) = T0, T(Lx) = T1
    One possible solution, that is examined here:
        T = -q x^2 / (2k) + C1 x + T0
    with
        C1 = (T1 - T0) / Lx + q^2/2/k*Lx

    Constraint:
        0 <= T0, T1 <= T1
    """
    title = "d(k dT/dx)/dx = -q, const k, T(0) = T0, T(Lx) = T1"
    k = 0.5
    Lx = 1.0
    T0 = 0.2
    T1 = 1.0
    q = -5
    title += f"\nk = {k}, q = {q}, T0 = {T0}, T1 = {T1}, Lx = {Lx}"
    # ==== Parameters ======
    epochs = 10000
    lr = 0.01
    depths = [64, 64, 64]
    N_train = 1001
    act_func_type = ACT_FUNCT_TYPE.TANH
    title += f"\nepochs = {epochs}, lr = {lr}, depths = {depths}, N_train = {N_train}, sigma = {act_func_type}"
    # ======================
    C1 = (T1 - T0) / Lx + q / 2 / k * Lx
    x0 = np.linspace(0.0, Lx, 1001)
    y0 = -q * x0**2 / 2 / k + C1 * x0 + T0
    torch.manual_seed(616)
    model = Model(in_features=1, depths=depths, out_features=1)  # T = f(x)
    model.set_act_func_type(act_func_type)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    losses = []
    x = torch.linspace(0.0, Lx, N_train).reshape(-1, 1)
    x.requires_grad = True
    x_bc = torch.FloatTensor([[0.0], [Lx]])
    y_bc = torch.FloatTensor([[T0], [T1]])
    for i in range(epochs):
        y_pred_bc = model(x_bc)
        loss_bc = criterion(y_pred_bc, y_bc)
        y_pred = model(x)
        yx_pred = torch.autograd.grad(
            y_pred, x, torch.ones_like(y_pred), create_graph=True, retain_graph=True
        )[0]
        yxx_pred = torch.autograd.grad(
            yx_pred, x, torch.ones_like(yx_pred), create_graph=True, retain_graph=True
        )[0]
        residual = yxx_pred * k + q
        loss_ode = criterion(residual, torch.zeros_like(residual))
        loss = loss_ode + loss_bc
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(
                f"Epoch = {i+1}   loss = {loss.item()} = {loss_bc.item()} + {loss_ode.item()}"
            )
    plt.subplot(2, 1, 1)
    plt.plot(x0, y0, label="exact solution")
    plt.plot(
        x.detach().numpy(), y_pred.detach().numpy(), label="PINN", linestyle="dashed"
    )
    plt.xlabel("x")
    plt.ylabel("T")
    plt.title(title)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.show()


def pinn_1d_cond_scaled():
    """
    Does the same thing as pinn_1d_cond() (above) but without
    constarining boundary condition and thus the solution to remain
    between 0 and 1.
    Equation:
        d(k dT/dx)/dx = -q, const k, T(0) = T0, T(Lx) = T1
    One possible solution that examined here:
        T = -q x^2 / (2k) + C1 x + T0
        C1 = (T1 - T0) / Lx + q^2/2/k*Lx
    """
    title = "d(k dT/dx)/dx = -q, const k, T(0) = T0, T(Lx) = T1"
    k = 0.5
    Lx = 10.0
    T0 = 0.2
    T1 = 100.0
    q = -5
    title += f"\nk = {k}, q = {q}, T0 = {T0}, T1 = {T1}, Lx = {Lx}"
    # ==== Parameters ====================================
    epochs = 10000
    lr = 0.01
    depths = [64, 64, 64]
    N_train = 1001
    act_func_type = ACT_FUNCT_TYPE.TANH
    title += f"\nepochs = {epochs}, lr = {lr}, depths = {depths}, N_train = {N_train}, sigma = {act_func_type}"
    # =========================================
    C1 = (T1 - T0) / Lx + q / 2 / k * Lx
    x0 = np.linspace(0.0, Lx, 1001)
    y0 = -q * x0**2 / 2 / k + C1 * x0 + T0
    torch.manual_seed(616)
    model = Model(in_features=1, depths=depths, out_features=1)  # T = f(x)
    model.set_act_func_type(act_func_type)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    losses = []
    X = torch.linspace(0.0, 1.0, N_train).reshape(-1, 1)
    X.requires_grad = True
    X_bc = torch.FloatTensor([[0.0], [1.0]])
    theta_bc = torch.FloatTensor([[0.0], [1.0]])
    for i in range(epochs):
        theta_pred_bc = model(X_bc)
        loss_bc = criterion(theta_pred_bc, theta_bc)
        theta_pred = model(X)
        theta_x_pred = torch.autograd.grad(
            theta_pred,
            X,
            torch.ones_like(theta_pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        theta_xx_pred = torch.autograd.grad(
            theta_x_pred,
            X,
            torch.ones_like(theta_x_pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        residual = theta_xx_pred + q * Lx**2 / (T1 - T0) / k
        loss_ode = criterion(residual, torch.zeros_like(residual))
        loss = loss_ode + loss_bc
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(
                f"Epoch = {i+1}   loss = {loss.item()} = {loss_bc.item()} + {loss_ode.item()}"
            )
    x = X * Lx
    y_pred = theta_pred * (T1 - T0) + T0
    plt.subplot(2, 1, 1)
    plt.plot(x0, y0, label="exact solution")
    plt.plot(
        x.detach().numpy(), y_pred.detach().numpy(), label="PINN", linestyle="dashed"
    )
    plt.xlabel("x")
    plt.ylabel("T")
    plt.title(title)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.show()


def pinn_1d_cond_conv():
    """
    Solving 1D conduction-convection equation with PINN.
    Equation:
        u dT/dx = k d^2T/dx^2, const k & u, T(0) = T0, T(Lx) = T1
    One possible solution:
        T(x) = T0 + (T1 - T0)(exp(Pe(x/Lx - 1)) - exp(-Pe))
                            / (1 - exp(-Pe))
    """
    Lx = 1.0
    Pe = 2.0
    T0 = 0.0
    T1 = 1.0
    title = "u dT/dx = k d^2T/dx^2, const k, T(0) = T0, T(Lx) = T1"
    title += f"\nT0 = {T0}, T1 = {T1}, Lx = {Lx}, Pe = {Pe}"
    # ===== PINN Parameters ==============
    epochs = 10000
    lr = 0.0005
    depths = [64, 64, 64]
    N_train = 1001
    act_func_type = ACT_FUNCT_TYPE.EXP  # TANH loss was flattening at loss_ode = 0.22
    title += f"\nepochs = {epochs}, lr = {lr}, depths = {depths}, N_train = {N_train}, sigma = {act_func_type}"
    # =====================================
    T_exact = lambda x: (
        T0 + (T1 - T0) * (np.exp(Pe * (x / Lx - 1)) - np.exp(-Pe)) / (1 - np.exp(-Pe))
        if Pe > 0
        else (np.exp(Pe * (x / Lx)) - 1) / (np.exp(Pe) - 1) if Pe < 0 else x / Lx
    )
    assert T_exact(0) == T0
    assert T_exact(Lx) == T1

    x = torch.linspace(0, 1.0, N_train).reshape(-1, 1)
    x.requires_grad = True
    torch.manual_seed(616)
    model = Model(in_features=1, depths=depths, out_features=1)  # T = f(x)
    model.set_act_func_type(act_func_type)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    losses = []
    x_bc = torch.FloatTensor([[0.0], [1.0]])
    y_bc = torch.FloatTensor([[0.0], [1.0]])
    for i in range(epochs):
        y_pred_bc = model(x_bc)
        loss_bc = criterion(y_pred_bc, y_bc)
        y_pred = model(x)
        yx_pred = torch.autograd.grad(
            y_pred, x, torch.ones_like(y_pred), create_graph=True, retain_graph=True
        )[0]
        yxx_pred = torch.autograd.grad(
            yx_pred, x, torch.ones_like(yx_pred), create_graph=True, retain_graph=True
        )[0]
        residual = yxx_pred - yx_pred * Pe
        loss_ode = criterion(residual, torch.zeros_like(residual))
        loss = loss_ode + loss_bc
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(
                f"Epoch = {i+1}   loss = {loss.item()} = {loss_ode.item()} + {loss_bc.item()}"
            )
    x0 = x.detach().numpy() * Lx
    y0 = T_exact(x0)
    theta_pred = y_pred.detach().numpy()
    y_pred = theta_pred * (T1 - T0) + T0
    plt.subplot(2, 1, 1)
    plt.plot(x0, y0, label="Exact solution")
    plt.plot(x0, y_pred, label="PINN", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.title(title)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    pinn_1d_cond()
    pinn_1d_cond_scaled()
    pinn_1d_cond_conv()


# py -m pinn.pinn_1d
