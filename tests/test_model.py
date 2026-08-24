import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import get_model


@pytest.mark.parametrize("architecture", ["resnet18", "simple_cnn"])
def test_forward_output_shape(architecture):
    model = get_model(architecture=architecture, num_classes=10)
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10)


def test_unknown_architecture_raises():
    with pytest.raises(ValueError):
        get_model(architecture="does_not_exist", num_classes=10)


def test_probabilities_sum_to_one():
    model = get_model(architecture="simple_cnn", num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_model_is_trainable():
    model = get_model(architecture="simple_cnn", num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([1, 3])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
