from pinn_2d_cond import *
from pytest import approx


def test_gen_square_domain():
    Lx = 5.0
    Ly = 3.0
    Nx = 6
    Ny = 4
    boundar_value = lambda x, y: x**2 + y**3
    X, Y, XY, XY_boundary, T_boundary = gen_square_domain(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, boundary_value=boundar_value
    )
    # in progress ...
    n = 0
    for i in range(6):
        for j in range(4):
            assert X[n] == i
            assert Y[n] == j
            assert XY[n, 0] == i
            assert XY[n, 1] == j
            n += 1
    assert XY_boundary.shape == (16, 2)
    n = 0
    for i in range(6):
        x = XY_boundary[n, 0]
        y = XY_boundary[n, 1]
        assert x == i and y == 0
        assert T_boundary[n] == x**2 + y**3
        n += 1
    for j in range(1, 3):
        x = XY_boundary[n, 0]
        y = XY_boundary[n, 1]
        assert x == 5 and y == j
        assert T_boundary[n] == x**2 + y**3
        n += 1
    for i in range(6):
        x = XY_boundary[n, 0]
        y = XY_boundary[n, 1]
        assert x == 5 - i and y == 3
        assert T_boundary[n] == x**2 + y**3
        n += 1
    for j in range(1, 3):
        x = XY_boundary[n, 0]
        y = XY_boundary[n, 1]
        assert x == 0 and y == 3 - j
        assert T_boundary[n] == x**2 + y**3
        n += 1


def test_Model_TXY1():
    Lx = 1.0
    Ly = 1.0
    T0 = 0.0
    T1 = 1.0
    alpha_x = 1 / Lx
    alpha_y = -alpha_x  # to satisfy Txx + Tyy = 0 => alpha_x = ± alpha_y
    beta_x = 0
    beta_y = 0
    T_theory_np = lambda x, y: T0 + T1 * np.sin(np.pi * alpha_x * x + beta_x) * np.exp(
        np.pi * alpha_y * y + beta_y
    )
    m = Model_TXY1(
        alpha_x=alpha_x, beta_x=beta_x, alpha_y=alpha_y, beta_y=beta_y, T0=T0, T1=T1
    )
    assert m(torch.FloatTensor([0, 0])).item() == approx(0)
    assert m(torch.FloatTensor([1, 0])).item() == approx(0, abs=1.0e-7)
    assert m(torch.FloatTensor([1, 1])).item() == approx(0, abs=1.0e-7)
    assert m(torch.FloatTensor([0, 1])).item() == approx(0)
    x0 = 0.25
    y0 = 0.75
    x1 = torch.FloatTensor([x0]).reshape(-1, 1)
    x1.requires_grad = True
    y1 = torch.FloatTensor([y0]).reshape(-1, 1)
    y1.requires_grad = True
    x = torch.cat([x1, y1], dim=1)
    T = m(x)
    Tth = T_theory_np(x0, y0)
    assert T.item() == approx(Tth)
    Tx = torch.autograd.grad(
        T, x1, torch.ones_like(T), create_graph=True, retain_graph=True
    )[0]
    Txx = torch.autograd.grad(
        Tx, x1, torch.ones_like(Tx), create_graph=True, retain_graph=True
    )[0]
    Ty = torch.autograd.grad(
        T, y1, torch.ones_like(T), create_graph=True, retain_graph=True
    )[0]
    Tyy = torch.autograd.grad(
        Ty, y1, torch.ones_like(Ty), create_graph=True, retain_graph=True
    )[0]
    Nx = 11
    Ny = 11
    x0 = np.linspace(0, Lx, Nx)
    y0 = np.linspace(0, Ly, Ny)
    x0 = np.repeat(x0, repeats=Ny)
    y0 = np.tile(y0, reps=Nx)
    x1 = torch.FloatTensor([x0.tolist()]).reshape(-1, 1)
    x1.requires_grad = True
    y1 = torch.FloatTensor([y0.tolist()]).reshape(-1, 1)
    y1.requires_grad = True
    x = torch.cat([x1, y1], dim=1)
    T = m(x)
    Tth = T_theory_np(x0, y0)
    Tx = torch.autograd.grad(
        T, x1, torch.ones_like(T), create_graph=True, retain_graph=True
    )[0]
    Txx = torch.autograd.grad(
        Tx, x1, torch.ones_like(Tx), create_graph=True, retain_graph=True
    )[0]
    Ty = torch.autograd.grad(
        T, y1, torch.ones_like(T), create_graph=True, retain_graph=True
    )[0]
    Tyy = torch.autograd.grad(
        Ty, y1, torch.ones_like(Ty), create_graph=True, retain_graph=True
    )[0]
    max_residual_err = torch.max(torch.abs(Txx + Tyy)).item()
    assert max_residual_err < 5e-7


def test_Model_TXY1_test2():
    Lx = 1.0
    Ly = 1.0
    T0 = 0.0
    T1 = 1.0
    alpha_x = 1 / Lx
    alpha_y = -alpha_x  # to satisfy Txx + Tyy = 0 => alpha_x = ± alpha_y
    beta_x = 0
    beta_y = 0
    T_theory_np = lambda x, y: T0 + T1 * np.sin(np.pi * alpha_x * x + beta_x) * np.exp(
        np.pi * alpha_y * y + beta_y
    )
    Nx_train = 101
    Ny_train = 101
    model = Model_TXY1(
        alpha_x=alpha_x, beta_x=beta_x, alpha_y=alpha_y, beta_y=beta_y, T0=T0, T1=T1
    )
    criterion = nn.MSELoss()
    X, Y, XY, XY_boundary, T_boundary = gen_square_domain(
        Nx=Nx_train, Ny=Ny_train, Lx=Lx, Ly=Ly, boundary_value=T_theory_np
    )
    T = model(XY)
    Tx = torch.autograd.grad(
        T, X, torch.ones_like(T), create_graph=True, retain_graph=True
    )[0]
    Txx = torch.autograd.grad(
        Tx, X, torch.ones_like(Tx), create_graph=True, retain_graph=True
    )[0]
    Ty = torch.autograd.grad(
        T, Y, torch.ones_like(T), create_graph=True, retain_graph=True
    )[0]
    Tyy = torch.autograd.grad(
        Ty, Y, torch.ones_like(Ty), create_graph=True, retain_graph=True
    )[0]
    residual = Txx + Tyy
    loss_pde = criterion(residual, torch.zeros_like(residual))
    T_pred_boundary = model(XY_boundary)
    loss_bc = criterion(T_pred_boundary, T_boundary)
    assert loss_pde.item() == approx(0)
    assert loss_bc.item() == approx(0)


# py -m FVM_PINN_workshop.test_pinn_2d
