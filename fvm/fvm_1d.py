import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from enum import Enum, auto
from pinn.model import *

"""
Practice runs on various TVD Schemes in a one-dimensional grid of cells.
References:
    Versteeg, H.K., Malalasekera, W. 2007 An introduction to computational fluid dynamics: 
        the finite volume method, 2/E. Pearson Education Limited. 
    Patankar, S., 2018. Numerical heat transfer and fluid flow. CRC press.

"""


class TVD_TYPE(Enum):
    """
    Types of Total Variation Diminishing (TVD). Here only 3 types are explored.
    """

    UPSREAM_FULL = auto()
    VAN_LEER = auto()
    VAN_ALABADA = auto()


class Psi:
    """
    The TVD function.
    Ref: Versteeg, H.K., Malalasekera, W. 2007
    An introduction to computational fluid dynamics: the finite volume method, 2/E. Pearson Education Limited.
    """

    def __init__(self, tvd_type: TVD_TYPE = TVD_TYPE.UPSREAM_FULL) -> None:
        self.tvd_type = tvd_type

    def __call__(self, r: float) -> float:
        return (
            ((r + np.fabs(r)) / (1 + r) if r > 0 else 0)
            if self.tvd_type == TVD_TYPE.VAN_LEER
            else (r + r**2) / (1 + r**2) if self.tvd_type == TVD_TYPE.VAN_ALABADA else 0
        )


def grid_1d_cond():
    """
    Simple 1D conduction equation solving:
    d^2u/dx^2 = 0
    """
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
    """
    Solves the 1D conduction-convection equation.
    Inputs:
    -   k: coefficient of the diffusion term
    -   vol_capacity: Volumetric capacity, coefficient of velocity in the convection term
    -   u: constant Velocity
    -   bc: (T0 ,T1) , Dirichlit boundary conditions at the 2 ends of the grid.
    -   Lx: Length of the grid
    -   Nx: Number of cells
    -   use_patankar_coeff: if True, applies the function A(Pe) to the diffusion term.
            Pe is the local Péclet number. Ref:
            Patankar, S., 2018. Numerical heat transfer and fluid flow. CRC press.
    -   tvd_type: TVD function type
    -   max_iter: A bound on number of iterations
    -   convergence_tolerance: Minimum acceptable convergence

    """
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


def grid_1d_cond_conv_upstream(
    tvd_type: TVD_TYPE = TVD_TYPE.UPSREAM_FULL, plot: bool = True
):
    """
    Solves the 1D conduction-convection equation with a given TVD method with and without the Patankar's
    diffusion coefficient (Patankar 2018) for a selected values of the Péclet numbers.
    Ref: Patankar, S., 2018. Numerical heat transfer and fluid flow. CRC press.
    """
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
                label=f"TVD type = {tvd_type} without Patankar's coeff, Pe = {Pe}",
                marker="+",
            )
            plt.scatter(
                xc[0::sampling_size],
                T[1][0::sampling_size],
                label=f"TVD type = {tvd_type} with Patankar's coeff, Pe = {Pe}",
                marker="x",
            )
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("T")
            plt.title(title)
            plt.show()

        for err_percent, Pe in zip(err_percent_list, Pe_list):
            plt.plot(
                xc,
                err_percent[:, 0],
                label=f"TVD type = {tvd_type} without Patankar's coeff, Pe = {Pe}",
            )
            plt.plot(
                xc,
                err_percent[:, 1],
                label=f"TVD type = {tvd_type} with Patankar's coeff, Pe = {Pe}",
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
                f"np.max(err_arr[:, 0]) = {np.max(err_arr[:, 0])} , expected_VAN_ALABADA_err_bounds[i][0] = {expected_upstream_eer_bounds[i][0]}"
            )
            print(
                f"np.max(err_arr[:, 1]) = {np.max(err_arr[:, 1])} , expected_VAN_ALABADA_err_bounds[i][1] = {expected_upstream_eer_bounds[i][1]}"
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
                f"np.max(err_arr[:, 0]) = {np.max(err_arr[:, 0])} , expected_VAN_LEER_err_bounds[i][0] = {expected_upstream_eer_bounds[i][0]}"
            )
            print(
                f"np.max(err_arr[:, 1]) = {np.max(err_arr[:, 1])} , expected_VAN_LEER_err_bounds[i][1] = {expected_upstream_eer_bounds[i][1]}"
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
    """
    Investigating the effect of number of cells for solving 1D conduction-convection
    when using a TVD method that is specified in the input.
    """
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


if __name__ == "__main__":
    grid_1d_cond()
    grid_1d_cond_conv_upstream(tvd_type=TVD_TYPE.UPSREAM_FULL)
    grid_1d_cond_conv_upstream(tvd_type=TVD_TYPE.VAN_ALABADA)
    grid_1d_cond_conv_upstream(tvd_type=TVD_TYPE.VAN_LEER)
    grid_1d_cond_conv_full_upstream_various_Nx(tvd_type=TVD_TYPE.UPSREAM_FULL)
    grid_1d_cond_conv_full_upstream_various_Nx(tvd_type=TVD_TYPE.VAN_ALABADA)
    grid_1d_cond_conv_full_upstream_various_Nx(tvd_type=TVD_TYPE.VAN_LEER)

# py -m fvm.fvm_1d
