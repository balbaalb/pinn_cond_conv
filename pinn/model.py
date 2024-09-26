import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Because torch.nn.funcation doesn't have an Exp() function, here one is created for the SeqModel class.
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
