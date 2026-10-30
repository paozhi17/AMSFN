"""Print parameters (M) and GFLOPs (640x640) for each variant."""
import sys

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

from amsfn_modules import apply_amsfn

cfg = sys.argv[1] if len(sys.argv) > 1 else "yolo11s.yaml"
nc = int(sys.argv[2]) if len(sys.argv) > 2 else 6

for tag, flags in [("base", None), ("A", dict(use_a=1, use_b=0, use_c=0)), ("B", dict(use_a=0, use_b=1, use_c=0)),
                   ("C", dict(use_a=0, use_b=0, use_c=1)), ("ABC", dict(use_a=1, use_b=1, use_c=1))]:
    m = DetectionModel(cfg, nc=nc, verbose=False)
    if flags:
        apply_amsfn(m, **{k: bool(v) for k, v in flags.items()})
    p = sum(x.numel() for x in m.parameters()) / 1e6
    print(f"{tag:5s} params {p:6.2f} M   GFLOPs {get_flops(m, 640):6.1f}")
