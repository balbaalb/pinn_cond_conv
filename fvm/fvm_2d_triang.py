import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from fvm.fvm_2d_grid import Grid
from typing import Callable

"""
Practice runs on Finite Volume Method (FVM) in a two-dimensional rectangular Cartesian grid.
References:
    Versteeg, H.K., Malalasekera, W. 2007 An introduction to computational fluid dynamics: 
        the finite volume method, 2/E. Pearson Education Limited. 
    Patankar, S., 2018. Numerical heat transfer and fluid flow. CRC press.

"""


class TriangulationGrid:
    """
    This class generates and manages a structured triangulation on rectangular grid of side lengths Lx and Ly.
    The recrangular domian is divided into Nx Ny blocks, with Nx and Ny denoting number of blocks in the horizontal
    and vertical directions, respectively. Each block contains 4 triangles by placing a node on the centre of the
    block and connecting that node to 4 vertices of the block. For example, Nx = 5 and Ny = 3 means a triangulation with
    number of vertices, Nv = 60, number of triangles, Nt = 39 (number of faces, Nf = 40), and number of edges, Ne = 98.
    """

    def __set_node_coords(self) -> None:
        """
        Should be used internally. Generates nodes (vertices) coordinates.
        """
        for i_node in range((self.Nx + 1) * (self.Ny + 1)):
            self.coords[i_node, 0] = i_node % (self.Nx + 1) * self.hx
            self.coords[i_node, 1] = int(i_node / (self.Nx + 1)) * self.hy
        for i_node in range((self.Nx + 1) * (self.Ny + 1), self.n_nodes):
            gn = i_node - (self.Nx + 1) * (self.Ny + 1)
            px = 0.5
            py = 0.5
            self.coords[i_node, 0] = (px + gn % (self.Nx)) * self.hx
            self.coords[i_node, 1] = (py + int(gn / (self.Nx))) * self.hy

    def __set_cell_nodes(self) -> None:
        """
        Should be used internally. Generates list nodes (vertices) for each of cell (triangle).
        """
        for gn in range(self.Nx * self.Ny):
            j = int(gn / self.Nx)
            a = gn + j
            b = a + 1
            c = b + self.Nx + 1
            d = c - 1
            f = gn + (self.Nx + 1) * (self.Ny + 1)
            self.cell_nodes[gn * 4, :] = [a, b, f]
            self.cell_nodes[gn * 4 + 1, :] = [b, c, f]
            self.cell_nodes[gn * 4 + 2, :] = [c, d, f]
            self.cell_nodes[gn * 4 + 3, :] = [d, a, f]

    def __set_neighbors(self):
        """
        Should be used internally. Generates list neighboring cells (triangles) for each cell.
        """
        sq_grid = Grid(Lx=0, Ly=0, Nx=self.Nx, Ny=self.Ny)
        for gn in range(self.Nx * self.Ny):
            self.cell_neighbors[gn * 4, 1] = (
                (sq_grid.S_index(gn) * 4 + 2) if sq_grid.S_index(gn) >= 0 else -1
            )
            self.cell_neighbors[gn * 4 + 1, 1] = (
                (sq_grid.E_index(gn) * 4 + 3) if sq_grid.E_index(gn) >= 0 else -1
            )
            self.cell_neighbors[gn * 4 + 2, 1] = (
                (sq_grid.N_index(gn) * 4) if sq_grid.N_index(gn) >= 0 else -1
            )
            self.cell_neighbors[gn * 4 + 3, 1] = (
                (sq_grid.W_index(gn) * 4 + 1) if sq_grid.W_index(gn) >= 0 else -1
            )
            for i in range(4):
                self.cell_neighbors[gn * 4 + i, 0] = gn * 4 + (i + 3) % 4
                self.cell_neighbors[gn * 4 + i, 2] = gn * 4 + (i + 1) % 4

    def __init__(self, Lx: float, Ly: float, Nx: int, Ny: int) -> None:
        """
        Lx: Horizontal length of the rectangular domain.
        Ly: Vertical length of the rectangular domain.
        Nx: Number of horizontal blocks.
        Ny: Number of vertical blocks.
        """
        self.Nx: int = Nx
        self.Ny: int = Ny
        self.Lx: float = Lx
        self.Ly: float = Ly
        self.n_cells: int = Nx * Ny * 4
        self.n_nodes: int = (Nx + 1) * (Ny + 1) + Nx * Ny
        self.hx: float = Lx / Nx
        self.hy: float = Ly / Ny
        self.coords = np.zeros((self.n_nodes, 2), dtype=float)
        self.__set_node_coords()
        self.cell_nodes = np.zeros((self.n_cells, 3), dtype=int)
        self.__set_cell_nodes()
        self.cell_neighbors = np.zeros(
            (self.n_cells, 3), dtype=int
        )  # Left , base, right neighboring cell indices
        self.__set_neighbors()

    def cell_coord(self, i_cell: int) -> tuple[float, float]:
        """
        Return (x, y) coordinates of the geometrical centre of cell # i_cell
        """
        cn = [self.cell_nodes[i_cell, i] for i in range(3)]
        r = np.zeros(2, dtype=float)
        for i in range(3):
            r += self.coords[cn[i], :] / 3.0
        return r[0], r[1]

    def edge_coords(self, i_cell: int, i_edge: int) -> tuple[float, float]:
        """
        Returns the mid-point coordinates of edge # i_edge for cell # i_cell.
        i_edge can be 0, 1, 2.
        """
        en = [
            self.cell_nodes[i_cell, i_edge],
            self.cell_nodes[i_cell, (i_edge + 1) % 3],
        ]
        r = 0.5 * self.coords[en[0], :] + 0.5 * self.coords[en[1], :]
        return r[0], r[1]


def fvm_2d_triang_grid_solver(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    k: float,
    bc: Callable[[float, float], float],
    vol_cap: float = 0,
    u: Callable[[float, float], tuple[float, float]] = None,
) -> tuple[npt.ArrayLike, Grid]:
    grid = Grid(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    dz = 1.0
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
        aE = De + np.max([-Fe, 0])

        Fw = (
            u(x=x - grid.hx / 2, y=y)[0] * vol_cap * grid.hy * dz
            if u is not None
            else 0
        )
        Dw = k / grid.dx_w(i_cell=i_cell) * grid.hy * dz
        aW = Dw + np.max([Fw, 0])

        Fn = (
            u(x=x, y=y + grid.hy / 2)[1] * vol_cap * grid.hx * dz
            if u is not None
            else 0
        )
        Dn = k / grid.dy_n(i_cell=i_cell) * grid.hx * dz
        aN = Dn + np.max([-Fn, 0])

        Fs = (
            u(x=x, y=y - grid.hy / 2)[1] * vol_cap * grid.hx * dz
            if u is not None
            else 0
        )
        Ds = k / grid.dy_s(i_cell=i_cell) * grid.hx * dz
        aS = Ds + np.max([Fs, 0])

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
