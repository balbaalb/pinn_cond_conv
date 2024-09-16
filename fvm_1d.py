import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from enum import Enum, auto
from model import *


class TVD_TYPE(Enum):
    UPSREAM_FULL = auto()
    VAN_LEER = auto()
    VAN_ALABADA = auto()


class Psi:
    def __init__(self, tvd_type: TVD_TYPE = TVD_TYPE.UPSREAM_FULL) -> None:
        self.tvd_type = tvd_type

    def __call__(self, r: float) -> float:
        return (
            ((r + np.fabs(r)) / (1 + r) if r > 0 else 0)
            if self.tvd_type == TVD_TYPE.VAN_LEER
            else (r + r**2) / (1 + r**2) if self.tvd_type == TVD_TYPE.VAN_ALABADA else 0
        )


def grid_1d_cond():
    plot = False
    k = 1  # does not matter
    Lx = 10
    T0 = 1
    T1 = 11
    T_exact = lambda x: T0 + x / Lx * (T1 - T0)
    assert T_exact(0) == T0
    assert T_exact(Lx) == T1
    Nx = 10  # number of cells
    xb = np.linspace(0, Lx, Nx + 1)  # boundary points
    Dx = Lx / Nx  # cell size
    xc = np.linspace(Dx / 2, Lx - Dx / 2, Nx)

    K = np.zeros((Nx, Nx), dtype=float)
    B = np.zeros(Nx, dtype=float)
    dy = 1
    for n in range(len(xc)):
        dx_w = xc[n] - xc[n - 1] if n > 0 else xc[0]
        Dw = k * dy / dx_w
        aW = Dw

        dx_e = xc[n + 1] - xc[n] if n < len(xc) - 1 else xb[n + 1] - xc[n]
        De = k * dy / dx_e
        aE = De

        aP = aE + aW
        K[n, n] = aP
        if n > 0:
            K[n, n - 1] = -aW
        else:
            B[0] = aW * T0
        if n < len(xc) - 1:
            K[n, n + 1] = -aE
        else:
            B[-1] = aE * T1

    T = np.linalg.solve(K, B)
    err_percent = np.fabs(T_exact(xc) - T) / T_exact(xc) * 100
    max_err_percent = np.max(err_percent)
    print(f"max error = {max_err_percent} %")
    if plot:
        plt.plot(xc, T)
        plt.show()
        plt.plot(xc, err_percent)
        plt.show()
    assert max_err_percent < 1.0e-10
    print(" ====== SUCCESS grid_1d_cond ========= ")


def grid_1d_conv_cond_solver(
    k: float,
    vol_capacity: float,
    u: float,
    bc: tuple[float, float],
    Lx: float,
    Nx: int,
    use_patankar_coeff: bool,
    tvd_type: TVD_TYPE = TVD_TYPE.UPSREAM_FULL,
    max_iter: int = 1,
    convergence_tolerance: float = 1.0e-5,
):
    T0 = bc[0]
    T1 = bc[1]
    Dx = Lx / Nx  # cell size
    xb = np.linspace(0, Lx, Nx + 1)  # boundary points
    xc = np.linspace(Dx / 2, Lx - Dx / 2, Nx)
    dy = 1
    K = np.zeros((Nx, Nx), dtype=float)
    B = np.zeros((Nx), dtype=float)
    T = np.zeros((Nx), dtype=float)
    T_prev = np.zeros_like(T)
    A = lambda D, F: np.max([0, (1 - 0.1 * np.fabs(F / D)) ** 5]) if D > 0 else 0
    psi = Psi(tvd_type=tvd_type)
    for i in range(max_iter):
        for n in range(len(xc)):
            dx_w = xc[n] - xc[n - 1] if n > 0 else xc[0]
            Dw = k * dy / dx_w
            Fw = vol_capacity * dy * u
            if not use_patankar_coeff:
                aW = Dw + (Fw if Fw > 0 else 0)
            else:
                aW = Dw * A(D=Dw, F=Fw) + (Fw if Fw > 0 else 0)

            dx_e = xc[n + 1] - xc[n] if n < len(xc) - 1 else xb[n + 1] - xc[n]
            De = k * dy / dx_e
            Fe = vol_capacity * dy * u
            if not use_patankar_coeff:
                aE = De - (Fe if Fe < 0 else 0)
            else:
                aE = De * A(D=De, F=Fe) - (Fe if Fe < 0 else 0)
            aP = aE + aW + Fe - Fw
            K[n, n] = aP
            if n > 0:
                K[n, n - 1] = -aW
            else:
                B[0] = aW * T0
            if n < len(xc) - 1:
                K[n, n + 1] = -aE
            else:
                B[-1] = aE * T1
            T_P = T[n]
            T_W = T[n - 1] if n > 0 else T0
            T_WW = T[n - 2] if n > 1 else T0 if n == 1 else 2 * T0 - T[0]
            T_E = T[n + 1] if n < len(xc) - 1 else T1
            T_EE = (
                T[n + 2]
                if n < len(xc) - 2
                else T1 if n == len(xc) - 2 else 2 * T1 - T[-1]
            )
            r_w_plus = (T_W - T_WW) / (T_P - T_W) if np.fabs(T_P - T_W) > 1.0e-15 else 0
            r_w_minus = (T_P - T_E) / (T_W - T_P) if np.fabs(T_P - T_W) > 1.0e-15 else 0
            alpha_w = 1.0 if Fw > 0 else 0.0
            S_DC = (
                0.5
                * Fw
                * (alpha_w * psi(r_w_plus) - (1 - alpha_w) * psi(r_w_minus))
                * (T_P - T_W)
            )

            r_e_plus = (T_P - T_W) / (T_E - T_P) if np.fabs(T_P - T_E) > 1.0e-15 else 0
            r_e_minus = (
                (T_E - T_EE) / (T_P - T_E) if np.fabs(T_P - T_E) > 1.0e-15 else 0
            )
            alpha_e = 1.0 if Fe > 0 else 0.0
            S_DC += (
                0.5
                * Fe
                * ((1.0 - alpha_e) * psi(r_e_minus) - alpha_e * psi(r_e_plus))
                * (T_E - T_P)
            )
            B[n] += S_DC
        T = np.linalg.solve(K, B)
        convergence = np.max(np.fabs(T - T_prev))
        if convergence < convergence_tolerance:
            break
        if (i + 1) % 100 == 0:
            print(f"iter = {i+1}, convergence = {convergence}")
        T_prev = T + 0
    print(f"convergence after iter = {i}, convergence = {convergence}, Nx = {Nx}")
    return xc, T


def grid_1d_cond_conv_upstream(tvd_type: TVD_TYPE = TVD_TYPE.UPSREAM_FULL):
    plot = True
    title = f"TVD = {tvd_type}\nwith and without Patankar multiplier for the diffusion term."
    k = 10
    rho = 1
    C_p = 1
    kappa = k / rho / C_p
    Lx = 10
    Pe_list = [1.0e-7, 0.1, 1, 10, 1.0e7]
    T_list = []
    err_percent_list = []
    T_exact_list = []
    for Pe in Pe_list:
        u = Pe * kappa / Lx
        print(f"Pe = {Pe}")
        T0 = 1
        T1 = 11
        T_exact = lambda x: (
            T0
            + (T1 - T0) * (np.exp(Pe * (x / Lx - 1)) - np.exp(-Pe)) / (1 - np.exp(-Pe))
            if Pe > 0
            else (np.exp(Pe * (x / Lx)) - 1) / (np.exp(Pe) - 1) if Pe < 0 else x / Lx
        )
        assert T_exact(0) == T0
        assert T_exact(Lx) == T1
        T_exact_list.append(T_exact)
        Nx = 1001  # number of cells
        xb = np.linspace(0, Lx, Nx + 1)  # boundary points
        Dx = Lx / Nx  # cell size
        xc = np.linspace(Dx / 2, Lx - Dx / 2, Nx)
        T = []  # np.zeros((Nx, 2), dtype=float)
        err_percent = np.zeros((Nx, 2), dtype=float)
        max_err_percent = np.zeros(2)
        max_iter = 1 if tvd_type == TVD_TYPE.UPSREAM_FULL else 100
        for i in range(2):
            use_patankar_coeff = i == 1
            T.append(
                grid_1d_conv_cond_solver(
                    k=k,
                    vol_capacity=rho * C_p,
                    u=u,
                    bc=(T0, T1),
                    Lx=Lx,
                    Nx=Nx,
                    use_patankar_coeff=use_patankar_coeff,
                    tvd_type=tvd_type,
                    max_iter=max_iter,
                )[1]
            )
            err_percent[:, i] = np.fabs(T_exact(xc) - T[i]) / T_exact(xc) * 100
            max_err_percent[i] = np.max(err_percent[:, i])
            if Nx <= 11:
                print(f"xb = {xb}")
                print(f"xc = {xc}")
                print(f"T = {T}")
                print(f"T_exact(xc) = {T_exact(xc)}")
        print(f"** max error = {max_err_percent} %, for Pe = {Pe}")
        T_list.append(T)
        err_percent_list.append(err_percent)
    if plot:
        for T, Pe, T_exact in zip(T_list, Pe_list, T_exact_list):
            sampling_size = 50
            plt.plot(xc, T_exact(xc), label=f"T_theory, Pe = {Pe}", color="blue")
            plt.scatter(
                xc[0::sampling_size],
                T[0][0::sampling_size],
                label=f"T_FVM_upstream , Pe = {Pe}",
                marker="+",
            )
            plt.scatter(
                xc[0::sampling_size],
                T[1][0::sampling_size],
                label=f"T_FVM_patankar, Pe = {Pe}",
                marker="x",
            )
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("T")
            plt.title(title)
            plt.show()

        for err_percent, Pe in zip(err_percent_list, Pe_list):
            plt.plot(xc, err_percent[:, 0], label=f"T_FVM_upstream , Pe = {Pe}")
            plt.plot(
                xc,
                err_percent[:, 1],
                label=f"T_FVM_patankar , Pe = {Pe}",
                linestyle="dashed",
            )
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("error%")
        plt.title(title)
        plt.show()
    if tvd_type == TVD_TYPE.UPSREAM_FULL:
        expected_upstream_eer_bounds = [0.0002, 0.0002, 0.015, 0.58, 0.3]
        for i, (Pe, err_arr) in enumerate(zip(Pe_list, err_percent_list)):
            assert np.max(err_arr[:, 1]) < 0.0002
            assert np.max(err_arr[:, 0]) < expected_upstream_eer_bounds[i]
    elif tvd_type == TVD_TYPE.VAN_ALABADA:
        expected_upstream_eer_bounds = [
            [1.0e-6, 1.0e-6],
            [1.0e-3, 1.0e-3],
            [1.5, 1.5],
            [56, 56],
            [0.14, 0.03],
        ]
        for i, (Pe, err_arr) in enumerate(zip(Pe_list, err_percent_list)):
            print(f"i = {i}")
            print(
                f"np.max(err_arr[:, 0]) = {np.max(err_arr[:, 0])} , expected_upstream_eer_bounds[i][0] = {expected_upstream_eer_bounds[i][0]}"
            )
            print(
                f"np.max(err_arr[:, 1]) = {np.max(err_arr[:, 1])} , expected_upstream_eer_bounds[i][1] = {expected_upstream_eer_bounds[i][1]}"
            )
            assert np.max(err_arr[:, 0]) < expected_upstream_eer_bounds[i][0]
            assert np.max(err_arr[:, 1]) < expected_upstream_eer_bounds[i][1]
    elif tvd_type == TVD_TYPE.VAN_LEER:
        expected_upstream_eer_bounds = [
            [1.0e-6, 1.0e-6],
            [1.0e-3, 1.0e-3],
            [1.5, 1.5],
            [56, 56],
            [0.11, 0.03],
        ]
        for i, (Pe, err_arr) in enumerate(zip(Pe_list, err_percent_list)):
            print(f"i = {i}")
            print(
                f"np.max(err_arr[:, 0]) = {np.max(err_arr[:, 0])} , expected_upstream_eer_bounds[i][0] = {expected_upstream_eer_bounds[i][0]}"
            )
            print(
                f"np.max(err_arr[:, 1]) = {np.max(err_arr[:, 1])} , expected_upstream_eer_bounds[i][1] = {expected_upstream_eer_bounds[i][1]}"
            )
            assert np.max(err_arr[:, 0]) < expected_upstream_eer_bounds[i][0]
            assert np.max(err_arr[:, 1]) < expected_upstream_eer_bounds[i][1]

    print(
        f"len(Pe) = {len(Pe_list)}, len(T_exact) = {len(T_exact_list)} , len(T) = {len(T_list)}, len(err_percent) = {len(err_percent_list)}"
    )
    print(" ====== SUCCESS grid_1d_cond_conv_upstream ========= ")


def grid_1d_cond_conv_full_upstream_various_Nx(
    tvd_type: TVD_TYPE = TVD_TYPE.UPSREAM_FULL,
):
    Nx_arr = [500, 1000]
    k = 10
    rho = 1
    C_p = 1
    kappa = k / rho / C_p
    Lx = 10
    Pe = 10
    u = Pe * kappa / Lx
    T0 = 1
    T1 = 11
    T_exact = lambda x: (
        T0 + (T1 - T0) * (np.exp(Pe * (x / Lx - 1)) - np.exp(-Pe)) / (1 - np.exp(-Pe))
        if Pe > 0
        else (np.exp(Pe * (x / Lx)) - 1) / (np.exp(Pe) - 1) if Pe < 0 else x / Lx
    )
    assert T_exact(0) == T0
    assert T_exact(Lx) == T1
    x = np.linspace(0, Lx, 101)
    T_exact_arr = T_exact(x)
    plt.plot(x, T_exact_arr, label="T_theory")
    for Nx in Nx_arr:
        print(f"tvd_type = {tvd_type}, Nx = {Nx}")
        xc, T = grid_1d_conv_cond_solver(
            k=k,
            vol_capacity=rho * C_p,
            u=u,
            bc=(T0, T1),
            Lx=Lx,
            Nx=Nx,
            use_patankar_coeff=False,
            tvd_type=tvd_type,
            max_iter=1000,
            convergence_tolerance=1.0e-3,
        )
        plt.scatter(
            xc[:: int(Nx / 101)], T[:: int(Nx / 101)], label=f"Nx = {Nx}", marker="x"
        )
    plt.legend()
    plt.title(f"tvd_type= {tvd_type}, Pe = {Pe}")
    plt.show()
    err_percent = np.fabs(T_exact(xc) - T) / T_exact(xc) * 100
    max_err_percent = np.max(err_percent)
    print(f"Nx = {Nx}, max_err_percent = {max_err_percent}")
    plt.plot(xc, err_percent, label="error%")
    plt.legend()
    plt.title(f"tvd_type= {tvd_type}, Pe = {Pe}")
    plt.show()
    print(" ====== SUCCESS grid_1d_cond_conv_full_upstream_various_Nx ========= ")


def pinn_1d_cond():
    # d(k dT/dx)/dx = -q, const k, T(0) = T0, T(Lx) = T1
    # solution: T = -q x^2 / (2k) + C1 x + T0
    # C1 = (T1 - T0) / Lx + q^2/2/k*Lx
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
    # d(k dT/dx)/dx = -q, const k, T(0) = T0, T(Lx) = T1
    # solution: T = -q x^2 / (2k) + C1 x + T0
    # C1 = (T1 - T0) / Lx + q^2/2/k*Lx
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
    Lx = 1.0
    Pe = 10.0
    T0 = 0.0
    T1 = 1.0
    title = "u dT/dx = kd^2T/dx^2, const k, T(0) = T0, T(Lx) = T1"
    title += f"\nT0 = {T0}, T1 = {T1}, Lx = {Lx}, Pe = {Pe}"
    # ===== PINN Parameters ==============
    epochs = 100000
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


def pinn_1d_cond_conv_integral_eq():
    Lx = 1.0
    Pe = 10.0
    T0 = 0.0
    T1 = 1.0
    title = "u dT/dx = kd^2T/dx^2, const k, T(0) = T0, T(Lx) = T1"
    title += f"\nT0 = {T0}, T1 = {T1}, Lx = {Lx}, Pe = {Pe}"
    # ===== PINN Parameters ==============
    epochs = 10000
    lr = 0.00001
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
    hx = 0.5 / N_train
    hx /= 1.0
    x_cell = torch.linspace(hx, 1.0 - hx, N_train).reshape(-1, 1)
    x_cell.requires_grad = True
    x_e = x_cell + hx
    x_w = x_cell - hx
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
        y_e = model(x_e)
        yx_e = torch.autograd.grad(
            y_e, x_e, torch.ones_like(y_e), create_graph=True, retain_graph=True
        )[0]
        y_w = model(x_w)
        yx_w = torch.autograd.grad(
            y_w, x_w, torch.ones_like(y_w), create_graph=True, retain_graph=True
        )[0]
        residual = (Pe * y_e - yx_e) - (Pe * y_w - yx_w)
        loss_integral = criterion(residual, torch.zeros_like(residual))
        loss = loss_integral + loss_bc
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(
                f"Epoch = {i+1}   loss = {loss.item()} = {loss_integral.item()} + {loss_bc.item()}"
            )
    x0 = x_cell.detach().numpy() * Lx
    y0 = T_exact(x0)
    y_pred = model(x_cell)
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
    # grid_1d_cond()
    # grid_1d_cond_conv_upstream(tvd_type=TVD_TYPE.UPSREAM_FULL)
    # grid_1d_cond_conv_upstream(tvd_type=TVD_TYPE.VAN_ALABADA)
    # grid_1d_cond_conv_upstream(tvd_type=TVD_TYPE.VAN_LEER)
    # grid_1d_cond_conv_full_upstream_various_Nx(tvd_type=TVD_TYPE.UPSREAM_FULL)
    # grid_1d_cond_conv_full_upstream_various_Nx(tvd_type=TVD_TYPE.VAN_ALABADA)
    # grid_1d_cond_conv_full_upstream_various_Nx(tvd_type=TVD_TYPE.VAN_LEER)
    # pinn_1d_cond()
    # pinn_1d_cond_scaled()
    # pinn_1d_cond_conv()
    pinn_1d_cond_conv_integral_eq()

# py -m FVM_PINN_workshop.fvm_1d
