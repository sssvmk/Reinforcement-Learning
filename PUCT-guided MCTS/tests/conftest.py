import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "submission"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="session")
def cfg():
    from config import load_config
    return load_config("quick")


@pytest.fixture(scope="session")
def random_net(cfg):
    import torch
    from network import build_net
    torch.manual_seed(123)
    net = build_net(cfg)
    net.train()
    for _ in range(3):   # give BatchNorm non-trivial running statistics
        net(torch.randint(0, 2, (64, 2, 6, 7)).float())
    return net.eval()
