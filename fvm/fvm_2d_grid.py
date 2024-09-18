import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Callable


class Grid:
    def __init__(self, Lx: float, Ly: float, Nx: int, Ny: int) -> None:
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.n_cells = Nx * Ny
        self.hx = Lx / Nx
        self.hy = Ly / Ny

    def E_index(self, i_cell: int) -> int:
        return (i_cell + 1) if (i_cell + 1) % self.Nx > 0 else -1

    def W_index(self, i_cell: int) -> int:
        return (i_cell - 1) if i_cell % self.Nx > 0 else -1

    def N_index(self, i_cell: int) -> int:
        return (i_cell + self.Nx) if (i_cell + self.Nx) < self.Nx * self.Ny else -1

    def S_index(self, i_cell: int) -> int:
        return (i_cell - self.Nx) if (i_cell - self.Nx) >= 0 else -1

    def dx_e(self, i_cell: int) -> float:
        return self.hx if self.E_index(i_cell=i_cell) >= 0 else self.hx / 2.0

    def dx_w(self, i_cell: int) -> float:
        return self.hx if self.W_index(i_cell=i_cell) >= 0 else self.hx / 2.0

    def dy_n(self, i_cell: int) -> float:
        return self.hy if self.N_index(i_cell=i_cell) >= 0 else self.hy / 2.0

    def dy_s(self, i_cell: int) -> float:
        return self.hy if self.S_index(i_cell=i_cell) >= 0 else self.hy / 2.0

    def n_x(self, i_cell):
        return i_cell % self.Nx

    def n_y(self, i_cell):
        return int(i_cell / self.Nx)

    def coords(self, i_cell) -> tuple[float, float]:
        nx = self.n_x(i_cell=i_cell)
        ny = self.n_y(i_cell=i_cell)
        x = (nx + 0.5) * self.hx
        y = (ny + 0.5) * self.hy
        return x, y


def fvm_2d_grid_solver(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    k: float,
    bc: Callable[[float, float], float],
    vol_cap: float = 0,
    u: Callable[[float, float], tuple[float, float]] = None,
    use_patankar_A: bool = False,
) -> tuple[npt.ArrayLike, Grid]:
    grid = Grid(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    dz = 1.0
    A = lambda D, F: (
        np.max([0, (1 - 0.1 * np.fabs(F / D)) ** 5])
        if D > 0 and use_patankar_A
        else 1.0
    )
    K = np.zeros((grid.n_cells, grid.n_cells), dtype=float)
    B = np.zeros(grid.n_cells, dtype=float)
    for i_cell in range(grid.n_cells):
        x, y = grid.coords(i_cell=i_cell)

        Fe = (
            u(x=x + grid.hx / 2, y=y)[0] * vol_cap * grid.hy * dz
            if u is not None
            else 0
        )
        De = k / grid.dx_e(i_cell=i_cell) * grid.hy * dz
        a = A(D=De, F=Fe)
        aE = De * A(D=De, F=Fe) + np.max([-Fe, 0])

        Fw = (
            u(x=x - grid.hx / 2, y=y)[0] * vol_cap * grid.hy * dz
            if u is not None
            else 0
        )
        Dw = k / grid.dx_w(i_cell=i_cell) * grid.hy * dz
        aW = Dw * A(D=Dw, F=Fw) + np.max([Fw, 0])

        Fn = (
            u(x=x, y=y + grid.hy / 2)[1] * vol_cap * grid.hx * dz
            if u is not None
            else 0
        )
        Dn = k / grid.dy_n(i_cell=i_cell) * grid.hx * dz
        aN = Dn * A(D=Dn, F=Fn) + np.max([-Fn, 0])

        Fs = (
            u(x=x, y=y - grid.hy / 2)[1] * vol_cap * grid.hx * dz
            if u is not None
            else 0
        )
        Ds = k / grid.dy_s(i_cell=i_cell) * grid.hx * dz
        aS = Ds * A(D=Ds, F=Fs) + np.max([Fs, 0])

        b = 0
        aP = aE + aW + aN + aS
        iE = grid.E_index(i_cell=i_cell)
        iW = grid.W_index(i_cell=i_cell)
        iN = grid.N_index(i_cell=i_cell)
        iS = grid.S_index(i_cell=i_cell)

        if iE >= 0:
            K[i_cell, iE] = -aE
        else:
            Tb_e = bc(x=x + grid.hx / 2, y=y)
            B[i_cell] += aE * Tb_e

        if iW >= 0:
            K[i_cell, iW] = -aW
        else:
            Tb_w = bc(x=x - grid.hx / 2, y=y)
            B[i_cell] += aW * Tb_w

        if iN >= 0:
            K[i_cell, iN] = -aN
        else:
            Tb_n = bc(x=x, y=y + grid.hy / 2)
            B[i_cell] += aN * Tb_n

        if iS >= 0:
            K[i_cell, iS] = -aS
        else:
            Tb_s = bc(x=x, y=y - grid.hy / 2)
            B[i_cell] += aS * Tb_s

        K[i_cell, i_cell] = aP
        B[i_cell] += b
    T = np.linalg.solve(K, B)
    return T, grid


def fvm_2d_grid_cond():
    Lx = 10
    Ly = 10
    k = 1.0  # conductvity
    T0 = 1.0
    T1 = 2.0
    alpha = np.pi / Lx
    beta = 0
    T_theory = lambda x, y: T0 + T1 * np.sin(alpha * x + beta) * np.exp(-alpha * y)
    N_list = range(1, 13)
    max_error = np.zeros(len(N_list))
    for i, N in enumerate(N_list):
        T, grid = fvm_2d_grid_solver(Lx=Lx, Ly=Ly, Nx=N, Ny=N, k=k, bc=T_theory)
        error = np.zeros_like(T)
        for i_cell in range(grid.n_cells):
            x, y = grid.coords(i_cell=i_cell)
            T_exact = T_theory(x, y)
            error = np.fabs(T[i_cell] - T_exact) / T_exact * 100
        print(f"max_error = {np.max(error)}")
        max_error[i] = np.max(error)
    plt.plot(N_list, max_error)
    plt.xlabel("grid size")
    plt.ylabel("max error %")
    plt.show()
    assert max_error[-1] < 0.01


def fvm_2d_grid_cond_conv():
    Lx = 10
    Ly = 10
    k = 1.0  # conductvity
    rho = 1.0
    Cp = 1.0
    kappa = k / rho / Cp
    u_theta = np.deg2rad(63.5)
    Pe_list = [100, 200, 500, 1000]
    N_list = range(1, 81)
    max_error = np.zeros((len(N_list), 2, len(Pe_list)))
    for p, Pe in enumerate(Pe_list):
        u_mag = Pe * kappa / Lx
        u = lambda x, y: (u_mag * np.cos(u_theta), u_mag * np.sin(u_theta))
        T0 = 1.0
        T1 = 2.0
        arg_max = u_mag * (Lx * np.cos(u_theta) + Ly * np.sin(u_theta)) / kappa
        T_theory = lambda x, y: T0 + T1 * np.exp(
            u_mag * (x * np.cos(u_theta) + y * np.sin(u_theta)) / kappa - arg_max
        )
        for a in range(2):
            for i, N in enumerate(N_list):
                T, grid = fvm_2d_grid_solver(
                    Lx=Lx,
                    Ly=Ly,
                    Nx=N,
                    Ny=N,
                    k=k,
                    bc=T_theory,
                    u=u,
                    vol_cap=rho * Cp,
                    use_patankar_A=(a == 1),
                )
                error = np.zeros_like(T)
                for i_cell in range(grid.n_cells):
                    x, y = grid.coords(i_cell=i_cell)
                    T_exact = T_theory(x, y)
                    error = np.fabs(T[i_cell] - T_exact) / T_exact * 100
                print(
                    f"Pe = {Pe}, N = {N}, use_patankar = {(a == 1)}, max_error = {np.max(error)}"
                )
                max_error[i, a, p] = np.max(error)
        plt.plot(N_list, max_error[:, 0, p], label=f"Pe = {Pe} without Patankar term")
        plt.plot(
            N_list,
            max_error[:, 1, p],
            label=f"Pe = {Pe} with Patankar term",
            linestyle="dashed",
        )
    plt.xlabel("grid size")
    plt.ylabel("max error %")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    fvm_2d_grid_cond()
    fvm_2d_grid_cond_conv()
