import torch

from gatr.primitives.gmconv_general import GMConvGeneralBase


def make_layer(error: bool) -> GMConvGeneralBase:
    group_elements = list(range(4))
    space_elements = list(range(4))

    def action(g, x):
        return (g + x) % len(space_elements)

    def inverse(g):
        return (-g) % len(group_elements)

    return GMConvGeneralBase(
        group_elements=group_elements,
        space_elements=space_elements,
        group_action=action,
        group_inverse_func=inverse,
        nbr_elements=group_elements,
        out_channels=2,
        in_channels=3,
        error=error,
    )


def test_forward_shape_without_error():
    layer = make_layer(error=False)
    x = torch.randn(5, 3, 4)
    out = layer(x)
    assert out.shape == (5, 2, 4)


def test_forward_shape_with_error():
    layer = make_layer(error=True)
    x = torch.randn(5, 3, 4)
    out = layer(x)
    assert out.shape == (5, 2, 4)
