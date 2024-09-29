from pytest import approx
from fvm.fvm_1d import *
from fvm.fvm_2d_grid import *
from fvm.fvm_2d_triang import *


def test_Psi():
    psi = Psi()
    assert psi(1.569) == 0
    psi = Psi(tvd_type=TVD_TYPE.VAN_LEER)
    assert psi(0) == 0
    assert psi(1) == 1
    assert psi(10) == 20 / 11
    psi = Psi(tvd_type=TVD_TYPE.VAN_ALABADA)
    assert psi(0) == 0
    assert psi(1) == 1
    assert psi(10) == 110 / 101


def test_grid():
    grid = Grid(Lx=2.0, Ly=7.0, Nx=5, Ny=3)
    assert grid.n_cells == 15
    assert grid.hx == 0.4
    assert grid.hy == 7 / 3
    assert grid.E_index(0) == 1
    assert grid.E_index(1) == 2
    assert grid.E_index(3) == 4
    assert grid.E_index(4) == -1
    assert grid.E_index(5) == 6
    assert grid.E_index(8) == 9
    assert grid.E_index(9) == -1
    assert grid.E_index(10) == 11
    assert grid.E_index(11) == 12
    assert grid.E_index(12) == 13
    assert grid.E_index(13) == 14
    assert grid.E_index(14) == -1
    assert grid.W_index(0) == -1
    assert grid.W_index(1) == 0
    assert grid.W_index(4) == 3
    assert grid.W_index(5) == -1
    assert grid.W_index(10) == -1
    assert grid.W_index(14) == 13
    assert grid.N_index(0) == 5
    assert grid.N_index(1) == 6
    assert grid.N_index(4) == 9
    assert grid.N_index(6) == 11
    assert grid.N_index(9) == 14
    for i in range(10, 15):
        assert grid.N_index(i) == -1
    for i in range(5):
        assert grid.S_index(i) == -1
    assert grid.S_index(5) == 0
    assert grid.S_index(6) == 1
    assert grid.S_index(7) == 2
    assert grid.S_index(9) == 4
    assert grid.S_index(10) == 5
    assert grid.S_index(14) == 9

    assert grid.dx_e(0) == 0.4
    assert grid.dx_w(0) == 0.2
    assert grid.dy_n(0) == 7 / 3
    assert grid.dy_s(0) == 7 / 6
    assert grid.dx_e(1) == 0.4
    assert grid.dx_w(1) == 0.4
    assert grid.dy_n(1) == 7 / 3
    assert grid.dy_s(1) == 7 / 6
    assert grid.dx_e(4) == 0.2
    assert grid.dx_w(4) == 0.4
    assert grid.dy_n(4) == 7 / 3
    assert grid.dy_s(4) == 7 / 6
    assert grid.dx_e(6) == 0.4
    assert grid.dx_w(6) == 0.4
    assert grid.dy_n(6) == 7 / 3
    assert grid.dy_s(6) == 7 / 3
    assert grid.dx_e(10) == 0.4
    assert grid.dx_w(10) == 0.2
    assert grid.dy_n(10) == 7 / 6
    assert grid.dy_s(10) == 7 / 3
    assert grid.dx_e(11) == 0.4
    assert grid.dx_w(11) == 0.4
    assert grid.dy_n(11) == 7 / 6
    assert grid.dy_s(11) == 7 / 3
    assert grid.dx_e(14) == 0.2
    assert grid.dx_w(14) == 0.4
    assert grid.dy_n(14) == 7 / 6
    assert grid.dy_s(14) == 7 / 3

    for i_cell in [0, 5, 10]:
        assert grid.n_x(i_cell=i_cell) == 0

    for i_cell in [1, 6, 11]:
        assert grid.n_x(i_cell=i_cell) == 1

    for i_cell in [2, 7, 12]:
        assert grid.n_x(i_cell=i_cell) == 2

    for i_cell in [3, 8, 13]:
        assert grid.n_x(i_cell=i_cell) == 3

    for i_cell in [4, 9, 14]:
        assert grid.n_x(i_cell=i_cell) == 4

    for i_cell in range(0, 5):
        assert grid.n_y(i_cell=i_cell) == 0

    for i_cell in range(5, 10):
        assert grid.n_y(i_cell=i_cell) == 1

    for i_cell in range(10, 15):
        assert grid.n_y(i_cell=i_cell) == 2

    assert grid.coords(0) == approx((0.2, 7 / 6))
    assert grid.coords(1) == approx((0.6, 7 / 6))
    assert grid.coords(2) == approx((1.0, 7 / 6))
    assert grid.coords(3) == approx((1.4, 7 / 6))
    assert grid.coords(4) == approx((1.8, 7 / 6))

    assert grid.coords(5) == approx((0.2, 3.5))
    assert grid.coords(6) == approx((0.6, 3.5))
    assert grid.coords(7) == approx((1.0, 3.5))
    assert grid.coords(8) == approx((1.4, 3.5))
    assert grid.coords(9) == approx((1.8, 3.5))

    assert grid.coords(10) == approx((0.2, 35 / 6))
    assert grid.coords(11) == approx((0.6, 35 / 6))
    assert grid.coords(12) == approx((1.0, 35 / 6))
    assert grid.coords(13) == approx((1.4, 35 / 6))
    assert grid.coords(14) == approx((1.8, 35 / 6))


def test_fvm_2d_grid_solver():
    Lx = 10
    Ly = 10
    k = 1.0  # conductvity
    T0 = 1.0
    T1 = 2.0
    alpha = np.pi / Lx
    beta = 0
    T_theory = lambda x, y: T0 + T1 * np.sin(alpha * x + beta) * np.exp(-alpha * y)
    T, grid = fvm_2d_grid_solver(Lx=Lx, Ly=Ly, Nx=12, Ny=12, k=k, bc=T_theory)
    error = np.zeros_like(T)
    for i_cell in range(grid.n_cells):
        x, y = grid.coords(i_cell=i_cell)
        T_exact = T_theory(x, y)
        error = np.fabs(T[i_cell] - T_exact) / T_exact * 100
    assert np.max(error) < 0.01


def test_fvm_2d_grid_solver_conv():
    Lx = 10
    Ly = 10
    k = 1.0  # conductvity
    u_mag = 1.0
    u_theta = np.deg2rad(63.5)
    u = lambda x, y: (u_mag * np.cos(u_theta), u_mag * np.sin(u_theta))
    T0 = 1.0
    T1 = 2.0
    arg_max = u_mag * (Lx * np.cos(u_theta) + Ly * np.sin(u_theta)) / k
    T_theory = lambda x, y: T0 + T1 * np.exp(
        u_mag * (x * np.cos(u_theta) + y * np.sin(u_theta)) / k - arg_max
    )
    N = 60
    T, grid = fvm_2d_grid_solver(
        Lx=Lx,
        Ly=Ly,
        Nx=N,
        Ny=N,
        k=k,
        bc=T_theory,
        u=u,
        use_patankar_A=False,
    )
    error = np.zeros_like(T)
    for i_cell in range(grid.n_cells):
        x, y = grid.coords(i_cell=i_cell)
        T_exact = T_theory(x, y)
        error = np.fabs(T[i_cell] - T_exact) / T_exact * 100
    assert np.max(error) < 0.06
    T, grid = fvm_2d_grid_solver(
        Lx=Lx,
        Ly=Ly,
        Nx=N,
        Ny=N,
        k=k,
        bc=T_theory,
        u=u,
        use_patankar_A=True,
    )
    error = np.zeros_like(T)
    for i_cell in range(grid.n_cells):
        x, y = grid.coords(i_cell=i_cell)
        T_exact = T_theory(x, y)
        error = np.fabs(T[i_cell] - T_exact) / T_exact * 100
    assert np.max(error) < 0.0005


def test_triangulation_grid():
    g = TriangulationGrid(Lx=2.0, Ly=9.0, Nx=5, Ny=3)
    assert g.n_cells == 60
    assert g.n_nodes == 39
    assert g.coords[0, 0] == 0 and g.coords[0, 1] == 0
    assert g.coords[1, 0] == 0.4 and g.coords[1, 1] == 0
    assert g.coords[4, 0] == 1.6 and g.coords[4, 1] == 0
    assert g.coords[5, 0] == 2.0 and g.coords[5, 1] == 0
    assert g.coords[6, 0] == 0.0 and g.coords[6, 1] == 3.0
    assert g.coords[8, 0] == 0.8 and g.coords[8, 1] == 3.0
    assert g.coords[11, 0] == 2.0 and g.coords[11, 1] == 3.0
    assert g.coords[12, 0] == 0.0 and g.coords[12, 1] == 6.0
    assert g.coords[17, 0] == 2.0 and g.coords[17, 1] == 6.0
    assert g.coords[18, 0] == 0.0 and g.coords[18, 1] == 9.0
    assert g.coords[23, 0] == 2.0 and g.coords[23, 1] == 9.0
    assert g.coords[24, 0] == 0.2 and g.coords[24, 1] == 1.5
    assert g.coords[25, 0] == approx(0.6) and g.coords[25, 1] == approx(1.5)
    assert g.coords[26, 0] == approx(1.0) and g.coords[26, 1] == approx(1.5)
    assert g.coords[27, 0] == approx(1.4) and g.coords[27, 1] == approx(1.5)
    assert g.coords[28, 0] == approx(1.8) and g.coords[28, 1] == approx(1.5)
    assert g.coords[29, 0] == approx(0.2) and g.coords[29, 1] == approx(4.5)
    assert g.coords[33, 0] == approx(1.8) and g.coords[33, 1] == approx(4.5)
    assert g.coords[34, 0] == approx(0.2) and g.coords[34, 1] == approx(7.5)
    assert g.coords[38, 0] == approx(1.8) and g.coords[38, 1] == approx(7.5)

    assert np.all(g.cell_nodes[0] == [0, 1, 24])
    assert np.all(g.cell_nodes[1] == [1, 7, 24])
    assert np.all(g.cell_nodes[2] == [7, 6, 24])
    assert np.all(g.cell_nodes[3] == [6, 0, 24])
    assert np.all(g.cell_nodes[4] == [1, 2, 25])
    assert np.all(g.cell_nodes[6] == [8, 7, 25])
    assert np.all(g.cell_nodes[13] == [4, 10, 27])
    assert np.all(g.cell_nodes[16] == [4, 5, 28])
    assert np.all(g.cell_nodes[17] == [5, 11, 28])
    assert np.all(g.cell_nodes[18] == [11, 10, 28])
    assert np.all(g.cell_nodes[19] == [10, 4, 28])
    assert np.all(g.cell_nodes[20] == [6, 7, 29])
    assert np.all(g.cell_nodes[21] == [7, 13, 29])
    assert np.all(g.cell_nodes[58] == [23, 22, 38])
    assert np.all(g.cell_nodes[59] == [22, 16, 38])

    assert np.all(g.cell_neighbors[0] == [3, -1, 1])
    assert np.all(g.cell_neighbors[1] == [0, 7, 2])
    assert np.all(g.cell_neighbors[2] == [1, 20, 3])
    assert np.all(g.cell_neighbors[3] == [2, -1, 0])
    assert np.all(g.cell_neighbors[4] == [7, -1, 5])
    assert np.all(g.cell_neighbors[5] == [4, 11, 6])
    assert np.all(g.cell_neighbors[6] == [5, 24, 7])
    assert np.all(g.cell_neighbors[7] == [6, 1, 4])
    assert np.all(g.cell_neighbors[16] == [19, -1, 17])
    assert np.all(g.cell_neighbors[17] == [16, -1, 18])
    assert np.all(g.cell_neighbors[18] == [17, 36, 19])
    assert np.all(g.cell_neighbors[19] == [18, 13, 16])
    assert np.all(g.cell_neighbors[28] == [31, 10, 29])
    assert np.all(g.cell_neighbors[29] == [28, 35, 30])
    assert np.all(g.cell_neighbors[30] == [29, 48, 31])
    assert np.all(g.cell_neighbors[31] == [30, 25, 28])
    assert np.all(g.cell_neighbors[40] == [43, 22, 41])
    assert np.all(g.cell_neighbors[41] == [40, 47, 42])
    assert np.all(g.cell_neighbors[42] == [41, -1, 43])
    assert np.all(g.cell_neighbors[43] == [42, -1, 40])
    assert np.all(g.cell_neighbors[56] == [59, 38, 57])
    assert np.all(g.cell_neighbors[57] == [56, -1, 58])
    assert np.all(g.cell_neighbors[58] == [57, -1, 59])
    assert np.all(g.cell_neighbors[59] == [58, 53, 56])

    assert g.cell_coord(i_cell=0) == approx((0.2, 0.5))
    assert g.cell_coord(i_cell=28) == approx((1.0, 3.5))
    assert g.cell_coord(i_cell=1) == approx((0.4 - 0.2 / 3.0, 1.5))
    assert g.cell_coord(i_cell=29) == approx((1.2 - 0.2 / 3.0, 4.5))
    assert g.cell_coord(i_cell=7) == approx((0.4 + 0.2 / 3.0, 1.5))
    assert g.cell_coord(i_cell=35) == approx((1.2 + 0.2 / 3.0, 4.5))
    assert g.cell_coord(i_cell=2) == approx((0.2, 2.5))
    assert g.cell_coord(i_cell=30) == approx((1.0, 5.5))

    assert g.edge_coords(i_cell=0, i_edge=0) == approx((0.2, 0.0))
    assert g.base_edge_coords(i_cell=0) == approx((0.2, 0.0))
    assert g.edge_coords(i_cell=0, i_edge=1) == approx((0.3, 0.75))
    assert g.right_edge_coords(i_cell=0) == approx((0.3, 0.75))
    assert g.edge_coords(i_cell=0, i_edge=2) == approx((0.1, 0.75))
    assert g.left_edge_coords(i_cell=0) == approx((0.1, 0.75))
    assert g.left_edge_coords(i_cell=1) == approx((0.3, 0.75))
    assert g.base_edge_coords(i_cell=1) == approx((0.4, 1.5))
    assert g.right_edge_coords(i_cell=1) == approx((0.3, 2.25))
    assert g.left_edge_coords(i_cell=2) == approx((0.3, 2.25))
    assert g.base_edge_coords(i_cell=2) == approx((0.2, 3.0))
    assert g.right_edge_coords(i_cell=2) == approx((0.1, 2.25))
    assert g.left_edge_coords(i_cell=3) == approx((0.1, 2.25))
    assert g.base_edge_coords(i_cell=3) == approx((0.0, 1.5))
    assert g.right_edge_coords(i_cell=3) == approx((0.1, 0.75))

    assert g.left_edge_coords(i_cell=28) == approx((0.9, 3.75))
    assert g.base_edge_coords(i_cell=28) == approx((1, 3))
    assert g.right_edge_coords(i_cell=28) == approx((1.1, 3.75))
    assert g.left_edge_coords(i_cell=29) == approx((1.1, 3.75))
    assert g.base_edge_coords(i_cell=29) == approx((1.2, 4.5))
    assert g.right_edge_coords(i_cell=29) == approx((1.1, 5.25))
    assert g.left_edge_coords(i_cell=30) == approx((1.1, 5.25))
    assert g.base_edge_coords(i_cell=30) == approx((1, 6))
    assert g.right_edge_coords(i_cell=30) == approx((0.9, 5.25))
    assert g.left_edge_coords(i_cell=31) == approx((0.9, 5.25))
    assert g.base_edge_coords(i_cell=31) == approx((0.8, 4.5))
    assert g.right_edge_coords(i_cell=31) == approx((0.9, 3.75))
