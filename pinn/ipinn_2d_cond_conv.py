import numpy as np
import torch
from pathlib import Path
from pinn.pinn_2d_cond_conv import PinnCondConv2D

THIS_PATH = Path(__file__).parent.resolve()


def gen_experimental_data(
    kappa: float,
    u_mag: float,
    u_angle_deg: float,
    Lx: float,
    Ly: float,
    N_data: int,
    phi0: float = 0,
    phi1: float = 1,
):
    """
    This function generates training data for the iPINN model.

    The iPINN cannot see the selected angle inside this function but would know about
    other parameters. iPINN model has to find the angles used in generating data from
    this function.
    """
    theta = np.deg2rad(u_angle_deg)
    ux = np.cos(theta) * u_mag
    uy = np.sin(theta) * u_mag
    Pe = np.array([ux, uy]) / kappa
    L = np.array([Lx, Ly])
    xy = np.random.random((N_data, 2)) * L
    arg = Pe * xy - Pe * L
    arg = np.sum(arg, axis=1)
    phi = phi0 + phi1 * np.exp(arg)
    xy = torch.FloatTensor(xy).reshape(-1, 2)
    xy.requires_grad = True
    phi = torch.FloatTensor(phi).reshape(-1, 1)
    return xy, phi


def ipinn_2d_cond_conv():
    k = 1.2
    u_mag = 2.5
    theta_deg = 35
    N_data = 1000
    Lx = 1.0
    Ly = 1.0
    phi0 = 0.0
    phi1 = 1.0
    depths = [64, 64, 64]
    epochs = 100
    lr = 0.0001
    Nx_train = 101
    Ny_train = 101
    xy_data, phi_data = gen_experimental_data(
        kappa=k,
        u_mag=u_mag,
        u_angle_deg=theta_deg,
        Lx=Lx,
        Ly=Ly,
        N_data=N_data,
        phi0=phi0,
        phi1=phi1,
    )
    model = PinnCondConv2D(
        depths=depths,
        Lx=Lx,
        Ly=Ly,
        Nx_train=Nx_train,
        Ny_train=Ny_train,
        phi_boundary=None,
        kappa=k,
        ux="find",
        uy="find",
        xy_experiment=xy_data,
        phi_experiment=phi_data,
    )
    file_name = "model_weights_2d_cond_conv.ipinn.pth"
    model_file = THIS_PATH / file_name
    if model_file.exists():
        model.net.load_state_dict(torch.load(model_file))
    else:
        model.train_pinn(lr=lr, epochs=epochs)
        torch.save(model.net.state_dict(), model_file)
    ipinn_u_mag = torch.sqrt(model.ux**2 + model.uy**2).item()
    err_umag = (ipinn_u_mag - u_mag) / u_mag * 100
    ipinn_theta_deg = torch.atan2(model.uy, model.ux).item() / np.pi * 180.0
    err_theta = (ipinn_theta_deg - theta_deg) / theta_deg * 100
    print(f"u_mag from iPINN = {ipinn_u_mag}, Expected = {u_mag}, error = {err_umag} %")
    print(
        f"theta_deg from iPINN = {ipinn_theta_deg}, Expected = {theta_deg}, error = {err_theta} %"
    )


if __name__ == "__main__":
    ipinn_2d_cond_conv()

# py -m  pinn.ipinn_2d_cond_conv
