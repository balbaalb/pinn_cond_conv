from pytest import approx
from fvm.fvm_1d import *
from fvm.fvm_2d_grid import *


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
