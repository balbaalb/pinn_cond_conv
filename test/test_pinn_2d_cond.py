from pinn.pinn_2d_cond import *
from pytest import approx


def test_gen_square_domain():
    Lx = 5.0
    Ly = 3.0
    Nx = 6
    Ny = 4
    boundar_value = lambda x, y: x**2 + y**3
    X, Y, XY, XY_boundary, phi_boundary = gen_square_domain(
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
        assert phi_boundary[n] == x**2 + y**3
        n += 1
    for j in range(1, 3):
        x = XY_boundary[n, 0]
        y = XY_boundary[n, 1]
        assert x == 5 and y == j
        assert phi_boundary[n] == x**2 + y**3
        n += 1
    for i in range(6):
        x = XY_boundary[n, 0]
        y = XY_boundary[n, 1]
        assert x == 5 - i and y == 3
        assert phi_boundary[n] == x**2 + y**3
        n += 1
    for j in range(1, 3):
        x = XY_boundary[n, 0]
        y = XY_boundary[n, 1]
        assert x == 0 and y == 3 - j
        assert phi_boundary[n] == x**2 + y**3
        n += 1


def test_Model_TXY1():
    Lx = 1.0
    Ly = 1.0
    phi0 = 0.0
    phi1 = 1.0
    alpha_x = 1 / Lx
    alpha_y = -alpha_x  # to satisfy phi_xx + phi_yy = 0 => alpha_x = ± alpha_y
    beta_x = 0
    beta_y = 0
    phi_theory_np = lambda x, y: phi0 + phi1 * np.sin(np.pi * alpha_x * x + beta_x) * np.exp(
        np.pi * alpha_y * y + beta_y
    )
    m = Model_Phi_XY1(
        alpha_x=alpha_x, beta_x=beta_x, alpha_y=alpha_y, beta_y=beta_y, phi0=phi0, phi1=phi1
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
    phi = m(x)
    phi_theory = phi_theory_np(x0, y0)
    assert phi.item() == approx(phi_theory)
    phi_x = torch.autograd.grad(
        phi, x1, torch.ones_like(phi), create_graph=True, retain_graph=True
    )[0]
    phi_xx = torch.autograd.grad(
        phi_x, x1, torch.ones_like(phi_x), create_graph=True, retain_graph=True
    )[0]
    phi_y = torch.autograd.grad(
        phi, y1, torch.ones_like(phi), create_graph=True, retain_graph=True
    )[0]
    phi_yy = torch.autograd.grad(
        phi_y, y1, torch.ones_like(phi_y), create_graph=True, retain_graph=True
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
    phi = m(x)
    phi_theory = phi_theory_np(x0, y0)
    phi_x = torch.autograd.grad(
        phi, x1, torch.ones_like(phi), create_graph=True, retain_graph=True
    )[0]
    phi_xx = torch.autograd.grad(
        phi_x, x1, torch.ones_like(phi_x), create_graph=True, retain_graph=True
    )[0]
    phi_y = torch.autograd.grad(
        phi, y1, torch.ones_like(phi), create_graph=True, retain_graph=True
    )[0]
    phi_yy = torch.autograd.grad(
        phi_y, y1, torch.ones_like(phi_y), create_graph=True, retain_graph=True
    )[0]
    max_residual_err = torch.max(torch.abs(phi_xx + phi_yy)).item()
    assert max_residual_err < 5e-7


def test_Model_TXY1_test2():
    Lx = 1.0
    Ly = 1.0
    phi0 = 0.0
    phi1 = 1.0
    alpha_x = 1 / Lx
    alpha_y = -alpha_x  # to satisfy phi_xx + phi_yy = 0 => alpha_x = ± alpha_y
    beta_x = 0
    beta_y = 0
    phi_theory_np = lambda x, y: phi0 + phi1 * np.sin(np.pi * alpha_x * x + beta_x) * np.exp(
        np.pi * alpha_y * y + beta_y
    )
    Nx_train = 101
    Ny_train = 101
    model = Model_Phi_XY1(
        alpha_x=alpha_x, beta_x=beta_x, alpha_y=alpha_y, beta_y=beta_y, phi0=phi0, phi1=phi1
    )
    criterion = nn.MSELoss()
    X, Y, XY, XY_boundary, phi_boundary = gen_square_domain(
        Nx=Nx_train, Ny=Ny_train, Lx=Lx, Ly=Ly, boundary_value=phi_theory_np
    )
    phi = model(XY)
    phi_x = torch.autograd.grad(
        phi, X, torch.ones_like(phi), create_graph=True, retain_graph=True
    )[0]
    phi_xx = torch.autograd.grad(
        phi_x, X, torch.ones_like(phi_x), create_graph=True, retain_graph=True
    )[0]
    phi_y = torch.autograd.grad(
        phi, Y, torch.ones_like(phi), create_graph=True, retain_graph=True
    )[0]
    phi_yy = torch.autograd.grad(
        phi_y, Y, torch.ones_like(phi_y), create_graph=True, retain_graph=True
    )[0]
    residual = phi_xx + phi_yy
    loss_pde = criterion(residual, torch.zeros_like(residual))
    phi_pred_boundary = model(XY_boundary)
    loss_bc = criterion(phi_pred_boundary, phi_boundary)
    assert loss_pde.item() == approx(0)
    assert loss_bc.item() == approx(0)


# py -m FVM_PINN_workshop.test_pinn_2d
