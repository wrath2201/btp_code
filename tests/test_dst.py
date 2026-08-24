"""
test_dst.py -- the Differentiable Stockwell Transform must reproduce the
fixed numpy S-transform (src/features.py) exactly at its initialization
(c=1, p=1, delta=0), and its parameters must receive gradients.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.dst import DifferentiableSTransform, numpy_reference

FS, F0, N, F_MAX = 6400.0, 50.0, 1280, 1600.0


def _signal(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    x = (np.sin(2 * np.pi * F0 * t)
         + 0.3 * np.sin(2 * np.pi * 3 * F0 * t + 1.0)
         + 0.1 * rng.normal(size=N))
    return x


def test_matches_fixed_stransform_at_init():
    x = _signal()
    ref = numpy_reference(x, FS, F_MAX)                      # rows 1..320

    dst = DifferentiableSTransform(N, FS, F_MAX)
    with torch.no_grad():
        out = dst(torch.from_numpy(x[None]).float())[0].numpy()

    assert out.shape == ref.shape == (320, N)
    err = np.abs(out - ref).max()
    assert err < 5e-5, f"max |DST - ST| = {err}"


def test_gradients_flow_to_window_law():
    x = torch.from_numpy(_signal(1)[None]).float()
    dst = DifferentiableSTransform(N, FS, F_MAX)
    loss = dst(x).mean()
    loss.backward()
    for name, p in dst.named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, \
            f"no gradient reached {name}"


def test_law_changes_output():
    x = torch.from_numpy(_signal(2)[None]).float()
    dst = DifferentiableSTransform(N, FS, F_MAX)
    with torch.no_grad():
        base = dst(x)
        dst.delta[9] = -1.0            # shrink sigma_t at the 50 Hz row
        mod = dst(x)
    row50 = int(round(50.0 / dst.df)) - 1
    assert not torch.allclose(base[0, row50], mod[0, row50])
    other = torch.cat([base[0, :row50], base[0, row50 + 1:]])
    other2 = torch.cat([mod[0, :row50], mod[0, row50 + 1:]])
    assert torch.allclose(other, other2)


if __name__ == "__main__":
    test_matches_fixed_stransform_at_init()
    test_gradients_flow_to_window_law()
    test_law_changes_output()
    print("all DST tests passed")
