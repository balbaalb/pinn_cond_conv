import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Callable
from scipy.sparse.linalg import gmres
from scipy.sparse import csc_matrix
import time

"""
Practice runs on Finite Volume Method (FVM) in a two-dimensional rectangular Cartesian grid.
References:
    Versteeg, H.K., Malalasekera, W. 2007 An introduction to computational fluid dynamics: 
        the finite volume method, 2/E. Pearson Education Limited. 
    Patankar, S., 2018. Numerical heat transfer and fluid flow. CRC press.

"""


class Grid:
    """
    A class to manage a simple Cartesian grid in a rectangular domain.
    """

    def __init__(self, Lx: float, Ly: float, Nx: int, Ny: int) -> None:
        """
        Lx: Length of rectangular domain horizontal (west to east) side
        Ly: Length of rectangular domain vertical (south to north) side
        Nx: Number of cells per row (west to east)
        Ny: Number of cells per column (south to north)

        Derived Attributes:
        n_cells: Total number of cells
        hx: Horizontal distance between two neighboring cells
        hy: Vertical distance between two neighboring cells
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.n_cells = Nx * Ny
        self.hx = Lx / Nx
        self.hy = Ly / Ny

    def E_index(self, i_cell: int) -> int:
        """
        Returns the index of the east side neighboreing cell of cell # i_cell.
        Return -1 if no eastern nighboring cell exists.
        """
        return (i_cell + 1) if (i_cell + 1) % self.Nx > 0 else -1

    def W_index(self, i_cell: int) -> int:
        """
        Returns the index of the west side neighboreing cell of cell # i_cell.
        Return -1 if no western nighboring cell exists.
        """
        return (i_cell - 1) if i_cell % self.Nx > 0 else -1

    def N_index(self, i_cell: int) -> int:
        """
        Returns the index of the north side neighboreing cell of cell # i_cell.
        Return -1 if no northern nighboring cell exists.
        """
        return (i_cell + self.Nx) if (i_cell + self.Nx) < self.Nx * self.Ny else -1

    def S_index(self, i_cell: int) -> int:
        """
        Returns the index of the south side neighboreing cell of cell # i_cell.
        Return -1 if no southern nighboring cell exists.
        """
        return (i_cell - self.Nx) if (i_cell - self.Nx) >= 0 else -1

    def dx_e(self, i_cell: int) -> float:
        """
        Returns distance between the centers of cell # i_cell and its eastern nighbor.
        If there is not eastern neighboring cell, it returns the distance between the
        center of cell # i_cell and the its eastern boundary.
        """
        return self.hx if self.E_index(i_cell=i_cell) >= 0 else self.hx / 2.0

    def dx_w(self, i_cell: int) -> float:
        """
        Returns distance between the centers of cell # i_cell and its western nighbor.
        If there is not western neighboring cell, it returns the distance between the
        center of cell # i_cell and the its western boundary.
        """
        return self.hx if self.W_index(i_cell=i_cell) >= 0 else self.hx / 2.0

    def dy_n(self, i_cell: int) -> float:
        """
        Returns distance between the centers of cell # i_cell and its northern nighbor.
        If there is not northern neighboring cell, it returns the distance between the
        center of cell # i_cell and the its northern boundary.
        """
        return self.hy if self.N_index(i_cell=i_cell) >= 0 else self.hy / 2.0

    def dy_s(self, i_cell: int) -> float:
        """
        Returns distance between the centers of cell # i_cell and its southern nighbor.
        If there is not southern neighboring cell, it returns the distance between the
        center of cell # i_cell and the its southern boundary.
        """
        return self.hy if self.S_index(i_cell=i_cell) >= 0 else self.hy / 2.0

    def n_x(self, i_cell):
        """
        Returns horizontal index of cell # i_cell. The horizontal index is counted from left
        side (western boundary) of the grid starting from 0.
        """
        return i_cell % self.Nx

    def n_y(self, i_cell):
        """
        Returns vertical index of cell # i_cell. The vertical index is counted from bottom
        side (southern boundary) of the grid starting from 0.
        """
        return int(i_cell / self.Nx)

    def coords(self, i_cell) -> tuple[float, float]:
        """
        Returns (x, y) coordinate of cell # i_cell.
        """
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
    u: Callable[[float, float], tuple[float, float]] = None,
    use_patankar_A: bool = False,
) -> tuple[npt.ArrayLike, Grid]:
    """
    Solves the Conduction-Convection problem,

    u ⋅ ∇φ = k Δφ,

    On a reactangular Cartesian grid. Here ∇ is the gradient operator, Δ the Laplacian operator, k the diffusivity,
    and φ is the conserved quantity. c is a multiplier (0 or 1 values only) for the convective term to
    turn on and off the convection.

    Lx: Length of the horizontal domain side.
    Ly: Length of the vertical domain side.
    Nx: Number of cells per horizontal row of the grid.
    Ny: Number of cells per vertical column of the grid.
    k: Conductivity
    bc: A function of (x,y) returning the value of Dirischlit boundary condition on
        any point on the boundary of the grid.

    u: (ux , uy) 2D velocity field. if u = None, the convective term is ignored.
    use_patankar_A : A flag to apply the Patankar's coefficient on the diffusion term.
    """
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

        Fe = u(x=x + grid.hx / 2, y=y)[0] * grid.hy * dz if u is not None else 0
        De = k / grid.dx_e(i_cell=i_cell) * grid.hy * dz
        aE = De * A(D=De, F=Fe) + np.max([-Fe, 0])

        Fw = u(x=x - grid.hx / 2, y=y)[0] * grid.hy * dz if u is not None else 0
        Dw = k / grid.dx_w(i_cell=i_cell) * grid.hy * dz
        aW = Dw * A(D=Dw, F=Fw) + np.max([Fw, 0])

        Fn = u(x=x, y=y + grid.hy / 2)[1] * grid.hx * dz if u is not None else 0
        Dn = k / grid.dy_n(i_cell=i_cell) * grid.hx * dz
        aN = Dn * A(D=Dn, F=Fn) + np.max([-Fn, 0])

        Fs = u(x=x, y=y - grid.hy / 2)[1] * grid.hx * dz if u is not None else 0
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
    K_sparse = csc_matrix(K)
    phi, exit_code = gmres(K_sparse, B, rtol=1.0e-6)
    if exit_code:
        print(f"GMRES exit_code = {exit_code}, so we have to use numpy's solve.")
        phi = np.linalg.solve(K, B)
    return phi, grid


def fvm_2d_grid_cond():
    """
    An experiment on the relation between grid resolution and
    the FVM error% for the conduction only conservation equation,

    k Δφ = 0,

    on a 2D Cartesian grid. Here Δ is the Laplacian operator, k is diffusivity,
    and φ is the conserved quanity.

    The following Dirichlet boundary condition is enforced:

    φ(x,y| (x,y) ∈ domain boundary) = φ0 + φ1 sin(αx + β) * exp(-αy + β)
    """
    Lx = 10
    Ly = 10
    k = 1.0  # conductvity
    phi0 = 1.0
    phi1 = 2.0
    alpha = np.pi / Lx
    beta = 0
    phi_theory = lambda x, y: phi0 + phi1 * np.sin(alpha * x + beta) * np.exp(
        -alpha * y
    )
    N_list = range(1, 13)
    max_error = np.zeros(len(N_list))
    for i, N in enumerate(N_list):
        phi, grid = fvm_2d_grid_solver(Lx=Lx, Ly=Ly, Nx=N, Ny=N, k=k, bc=phi_theory)
        error = np.zeros_like(phi)
        for i_cell in range(grid.n_cells):
            x, y = grid.coords(i_cell=i_cell)
            phi_exact = phi_theory(x, y)
            error = np.fabs(phi[i_cell] - phi_exact) / phi_exact * 100
        print(f"max_error = {np.max(error)}")
        max_error[i] = np.max(error)
    plt.plot(N_list, max_error)
    plt.xlabel("grid size")
    plt.ylabel("max error %")
    plt.title("Max error% for the conduction equation on a Cartesian grid.")
    plt.show()
    assert max_error[-1] < 0.01


def fvm_2d_grid_cond_conv():
    """
    An experiment on the relation between grid resolution and
    the FVM error% for the conduction only conservation equation,

    u ⋅ ∇φ = k Δφ,

    on a 2D Cartesian grid. Here Δ is the Laplacian operator, k is diffusivity,
    and φ is the conserved quanity, ∇ is the graident operator and u is the 2D velocity field.

    The following Dirichlet boundary condition is enforced:

    φ(x,y| (x,y) ∈ domain boundary) = φ0 + φ1 exp((x * ux + y * uy) / k - argmax)

    where argmax = (Lx * ux + Ly * uy) / k.
    """
    Lx = 10
    Ly = 10
    k = 1.0  # conductvity
    u_theta = np.deg2rad(63.5)
    Pe_list = [100, 200, 500, 1000]
    N_list = np.arange(2, 101, 2)
    max_error = np.zeros((len(N_list), len(Pe_list)))
    start_time = time.time()
    for p, Pe in enumerate(Pe_list):
        u_mag = Pe * k / Lx
        u = lambda x, y: (u_mag * np.cos(u_theta), u_mag * np.sin(u_theta))
        phi0 = 1.0
        phi1 = 2.0
        arg_max = u_mag * (Lx * np.cos(u_theta) + Ly * np.sin(u_theta)) / k
        phi_theory = lambda x, y: phi0 + phi1 * np.exp(
            u_mag * (x * np.cos(u_theta) + y * np.sin(u_theta)) / k - arg_max
        )
        for i, N in enumerate(N_list):
            phi, grid = fvm_2d_grid_solver(
                Lx=Lx,
                Ly=Ly,
                Nx=N,
                Ny=N,
                k=k,
                bc=phi_theory,
                u=u,
                use_patankar_A=True,
            )
            error = np.zeros_like(phi)
            for i_cell in range(grid.n_cells):
                x, y = grid.coords(i_cell=i_cell)
                phi_exact = phi_theory(x, y)
                error = np.fabs(phi[i_cell] - phi_exact) / phi_exact * 100
            print(
                f"Pe = {Pe}, N = {N}, use_patankar = True, max_error = {np.max(error)}"
            )
            max_error[i, p] = np.max(error)
        plt.plot(N_list, max_error[:, p], label=f"Pe = {Pe}")
    end_time = time.time()
    print(f"cond-conv: dt = {end_time - start_time} s, max_err = {np.max(max_error)} %")
    plt.xlabel("grid size")
    plt.ylabel("max error %")
    plt.title("Max error% for the conduction-convection equation on a Cartesian grid.")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    fvm_2d_grid_cond()
    fvm_2d_grid_cond_conv()

# py -m fvm.fvm_2d_grid
