import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Callable
from enum import Enum, auto


class ACT_FUNCT_TYPE(Enum):
    """
    Enum class for activation type.
    """

    RELU = auto()
    TANH = auto()
    EXP = auto()


class Model(nn.Module):
    """
    A standard torch fully connected feed forward neural network model class.
    """

    def __init__(self, in_features=1, depths=[], out_features=1) -> None:
        super().__init__()
        self.layers = []
        n_input = in_features
        for depth in depths:
            self.layers.append(nn.Linear(in_features=n_input, out_features=depth))
            n_input = depth
        self.out = nn.Linear(in_features=n_input, out_features=out_features)
        self.f = F.relu

    def set_act_func_type(self, act_func_type: ACT_FUNCT_TYPE) -> None:
        match act_func_type:
            case ACT_FUNCT_TYPE.RELU:
                self.f = F.relu
            case ACT_FUNCT_TYPE.TANH:
                self.f = F.tanh
            case ACT_FUNCT_TYPE.EXP:
                self.f = torch.exp

    def forward(self, x):
        for layer in self.layers:
            x = self.f(layer(x))
        x = self.out(x)
        return x


class Exp(nn.Module):
    """
    Because torch.nn.function doesn't have Exp(), here one is created to be used in the SeqModel class (implemeted below).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class SeqModel(nn.Sequential):
    """
    Creating a fully connected feed forward neural network inherited from torch.nn.Sequential.
    Although in theory this class should be behaving in the same exact manner as nn.Module; however,
    it is observed that model classes inherited from nn.Sequential are optimized faster compared to the
    ones inhertited from nn.Module.
    """

    def __init__(
        self,
        in_features=1,
        depths=[],
        out_features=1,
        act_func_type: ACT_FUNCT_TYPE = ACT_FUNCT_TYPE.TANH,
    ) -> None:
        super().__init__()
        n_input = in_features
        for i, depth in enumerate(depths):
            self.add_module(
                name=str(i), module=nn.Linear(in_features=n_input, out_features=depth)
            )
            match act_func_type:
                case ACT_FUNCT_TYPE.RELU:
                    self.add_module(name=str(i) + "-ReLU", module=nn.ReLU())
                case ACT_FUNCT_TYPE.TANH:
                    self.add_module(name=str(i) + "-Tanh", module=nn.Tanh())
                case ACT_FUNCT_TYPE.EXP:
                    self.add_module(name=str(i) + "-Exp", module=Exp())
                case _:
                    self.add_module(name=str(i) + "-Tanh", module=nn.Tanh())
            n_input = depth
        self.add_module(
            name="out", module=nn.Linear(in_features=n_input, out_features=out_features)
        )

    def forward(self, input):
        return super().forward(input)


class PinnCondConv1D:
    """
    Creates a pinn model for the equation

    u dφ/dx = k d^2φ/dx^2 + q,

    on a 1D domain. k is diffusivity, u is velocity and φ is the conserved quantity.
    """

    def __init__(
        self,
        depths: list[int],
        Lx: float,
        N_train: int,
        kappa: float = 1,
        u: float = 0,
        q: float = 0,
        phi0: float = 0,
        phi1: float = 1,
    ) -> None:
        self.net = SeqModel(in_features=1, depths=depths, out_features=1)
        self.X = torch.linspace(0, Lx, N_train).reshape(-1, 1)
        self.X.requires_grad = True
        self.kappa = kappa
        self.losses = []
        self.epoch = 0
        self.u = u
        self.q = q
        self.x_bc = torch.FloatTensor([[0.0], [Lx]])
        self.phi_bc = torch.FloatTensor([[phi0], [phi1]])

    def __pde_residual(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Returns k d^2φ/dx^2 - u dφ/dx

        """
        phi_x = torch.autograd.grad(
            phi, self.X, torch.ones_like(phi), create_graph=True, retain_graph=True
        )[0]
        phi_xx = torch.autograd.grad(
            phi_x, self.X, torch.ones_like(phi_x), create_graph=True, retain_graph=True
        )[0]
        residual = self.kappa * phi_xx - self.u * phi_x + self.q
        return residual

    def __loss_pinn(self) -> torch.Tensor:
        """
        Returns Loss function for training: residual of PDE + loss of deciation from the given boundary condions
        """
        self.epoch += 1
        phi = self.net(self.X)
        residual = self.__pde_residual(phi=phi)
        loss_pde = self.criterion(residual, torch.zeros_like(residual))
        phi_pred_boundary = self.net(self.x_bc)
        loss_bc = self.criterion(phi_pred_boundary, self.phi_bc)
        loss = loss_pde + loss_bc
        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        if (self.epoch) % 100 == 0:
            print(
                f"Epoch = {self.epoch}, loss = {loss.item()} = {loss_pde.item()} (PDE) + {loss_bc.item()} (BC)"
            )
        return loss

    def train_pinn(self, lr: float, epochs: int) -> None:
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        for _ in range(epochs):
            self.optimizer.step(self.__loss_pinn)
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.optimizer.step(self.__loss_pinn)
        if len(self.losses) > 0:
            plt.plot(self.losses)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.show()


def gen_square_domain(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    boundary_value: Callable[[float, float], float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates PyTorch domain nodes and boundary condition training data for a PINN for a
    rectangular domain.
    Inputs:
        - Lx: Horizontal side length of the rectangular domain
        - Ly: Vertical side length of the rectangular domain
        - Nx: Number of nodes on a horizntal row of nodes
        - Ny: Number of nodes on a vertical column of nodes
        - boundary_value: a function (x ,y) returning the Dirichlet boundary condition on the
            boundary node (x, y). This function should be numpy based.
    Outputs: (X, Y, XY, XY_boundary, phi_boundary)
        - X: A torch tensor with shape (Nx * Ny , 1) containing x coordinate of domain nodes
        - Y: A torch tensor with shape (Nx * Ny , 1) containing y coordinate of domain nodes
        - XY: A torch tensor with shape (Nx * Ny , 2) containing (x,y) coordinate of domain nodes,
            this is just X and Y outputs concatenated.
        - XY_boundary: A torch tensor with shape (2 * Nx + 2 * Ny , 2) containing (x , y) coordinate
            of boundary
        - phi_boundary: A torch tensor with shape (2 * Nx + 2 * Ny , 1) containing Dirichlet boundary
            condition values for points in XY_boundary.

    """
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    xx = np.repeat(x, Ny)
    yy = np.tile(y, Nx)
    X = torch.FloatTensor(xx).reshape(-1, 1)
    X.requires_grad = True
    Y = torch.FloatTensor(yy).reshape(-1, 1)
    Y.requires_grad = True
    XY = torch.cat([X, Y], dim=1)

    xe = np.zeros(Ny) + Lx
    ye = np.linspace(0, Ly, Ny)
    xw = np.zeros(Ny)
    yw = np.linspace(Ly, 0, Ny)
    xn = np.linspace(Lx, 0, Nx)
    yn = np.zeros(Nx) + Ly
    xs = np.linspace(0, Lx, Nx)
    ys = np.zeros(Nx)
    x_boundary = np.concatenate((xs[:-1], xe[:-1], xn[:-1], xw[:-1])).reshape(-1, 1)
    y_boundary = np.concatenate((ys[:-1], ye[:-1], yn[:-1], yw[:-1])).reshape(-1, 1)
    xy_boundary = np.concatenate((x_boundary, y_boundary), axis=1)
    XY_boundary = torch.FloatTensor(xy_boundary)
    if boundary_value is None:
        return X, Y, XY, XY_boundary
    phi_boundary_np = boundary_value(x_boundary, y_boundary)
    phi_boundary = torch.FloatTensor(phi_boundary_np)
    return X, Y, XY, XY_boundary, phi_boundary


class PinnCondConv2D:
    """
    Creates a pinn model for the equation

    u ⋅ ∇φ = k Δφ,

    on a 2D rectangular domain. Here Δ is the Laplacian operator, k is diffusivity,
    and φ is the conserved quantity, ∇ is the graident operator and u is the 2D velocity field.
    """

    def __init__(
        self,
        depths: list[int],
        Lx: float,
        Ly: float,
        Nx_train: int,
        Ny_train: int,
        phi_theory_np: Callable[[float, float], float],
        kappa: float,
        ux: float = 0,
        uy: float = 0,
    ) -> None:
        self.net = SeqModel(in_features=2, depths=depths, out_features=1)
        self.X, self.Y, self.XY, self.XY_boundary, self.phi_boundary = (
            gen_square_domain(
                Nx=Nx_train, Ny=Ny_train, Lx=Lx, Ly=Ly, boundary_value=phi_theory_np
            )
        )
        self.kappa = kappa
        self.ux = ux
        self.uy = uy
        self.losses = []
        self.epoch = 0

    def __pde_residual(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Returns k Δφ - u ⋅ ∇φ

        """
        phi_x = torch.autograd.grad(
            phi, self.X, torch.ones_like(phi), create_graph=True, retain_graph=True
        )[0]
        phi_xx = torch.autograd.grad(
            phi_x, self.X, torch.ones_like(phi_x), create_graph=True, retain_graph=True
        )[0]
        phi_y = torch.autograd.grad(
            phi, self.Y, torch.ones_like(phi), create_graph=True, retain_graph=True
        )[0]
        phi_yy = torch.autograd.grad(
            phi_y, self.Y, torch.ones_like(phi_y), create_graph=True, retain_graph=True
        )[0]
        residual = self.kappa * (phi_xx + phi_yy) - (self.ux * phi_x + self.uy * phi_y)
        return residual

    def __loss_pinn(self) -> torch.Tensor:
        """
        Returns Loss function for training: residual of PDE + loss of deciation from the given boundary condions
        """
        self.epoch += 1
        phi = self.net(self.XY)
        residual = self.__pde_residual(phi=phi)
        loss_pde = self.criterion(residual, torch.zeros_like(residual))
        phi_pred_boundary = self.net(self.XY_boundary)
        loss_bc = self.criterion(phi_pred_boundary, self.phi_boundary)
        loss = loss_pde + loss_bc
        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        if (self.epoch) % 100 == 0:
            print(
                f"Epoch = {self.epoch}, loss = {loss.item()} = {loss_pde.item()} (PDE) + {loss_bc.item()} (BC)"
            )
        return loss

    def train_pinn(self, lr: float, epochs: int) -> None:
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        for _ in range(epochs):
            self.optimizer.step(self.__loss_pinn)
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.optimizer.step(self.__loss_pinn)
        if len(self.losses) > 0:
            plt.plot(self.losses)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.show()
