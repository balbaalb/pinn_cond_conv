from pinn.model import *


def test_model_exp_function():
    # f(x) = 1 - exp(-5x)
    m = Model(in_features=1, depths=[1], out_features=1)
    m.set_act_func_type(ACT_FUNCT_TYPE.EXP)
    m.layers[0].weight = nn.Parameter(torch.FloatTensor([[-5.0]]))
    m.layers[0].bias = nn.Parameter(torch.FloatTensor([0.0]))
    m.out.weight = nn.Parameter(torch.FloatTensor([[-1.0]]))
    m.out.bias = nn.Parameter(torch.FloatTensor([1.0]))
    x = torch.linspace(0.0, 1.0, 101).reshape(-1, 1)
    x.requires_grad = True
    y_test = 1.0 - torch.exp(-5.0 * x)
    y_pred = m(x)
    criterion = nn.MSELoss()
    loss = criterion(y_test, y_pred)
    assert loss.item() == 0
    yx_pred = torch.autograd.grad(
        y_pred, x, torch.ones_like(y_pred), create_graph=True, retain_graph=True
    )[0]
    yx_test = 5.0 * torch.exp(-5.0 * x)
    loss_x = criterion(yx_test, yx_pred)
    assert loss_x.item() == 0
