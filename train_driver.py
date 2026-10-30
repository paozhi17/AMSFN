"""Driver-behavior training on State Farm / AUC.

200 epochs, batch 2, SGD (lr0 0.01, momentum 0.937, wd 5e-4), cosine LR with
3 warm-up epochs, Mosaic / flip / HSV / scale augmentation, input long side 640
(State Farm frames are 640x480).

  python train_driver.py --data data_statefarm.yaml --variant abc          # YOLO11s + ABC
  python train_driver.py --data data_statefarm.yaml --variant none         # YOLO11s
  python train_driver.py --data data_statefarm.yaml --variant a            # single module
  python train_driver.py --data data_statefarm.yaml --variant abc --seeds 42 43 44
"""
import argparse
import csv
import os
import statistics as st

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from amsfn_modules import apply_amsfn


class AMSFNTrainer(DetectionTrainer):
    flags = dict(use_a=True, use_b=True, use_c=True)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if any(self.flags.values()):
            apply_amsfn(model, **self.flags)
        return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11s.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--variant", default="abc", help="none | a | b | c | any combination, e.g. ab")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--ckpt", choices=["last", "best"], default="last")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--project", default="runs_driver")
    a = ap.parse_args()

    v = a.variant.lower()
    AMSFNTrainer.flags = dict(use_a="a" in v, use_b="b" in v, use_c="c" in v)
    tag = "yolo11s" if v == "none" else f"yolo11s_{v.upper()}"

    rows = []
    for seed in a.seeds:
        model = YOLO(a.weights or a.model)
        model.train(
            trainer=AMSFNTrainer, data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
            optimizer="SGD", lr0=0.01, momentum=0.937, weight_decay=5e-4,
            cos_lr=True, warmup_epochs=3,
            mosaic=1.0, fliplr=a.fliplr, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5,
            seed=seed, deterministic=True, amp=True, device=a.device, workers=a.workers,
            project=a.project, name=f"{tag}_seed{seed}", exist_ok=True,
        )
        ckpt = model.trainer.last if a.ckpt == "last" else model.trainer.best
        m = YOLO(str(ckpt)).val(data=a.data, imgsz=a.imgsz, batch=8, device=a.device).box
        rows.append(dict(seed=seed, AP50=m.map50 * 100, AP75=m.map75 * 100,
                         AP50_95=m.map * 100, Recall=m.mr * 100))
        print(rows[-1])

    os.makedirs(a.project, exist_ok=True)
    with open(f"{a.project}/summary_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    if len(rows) > 1:
        for k in ("AP50", "AP75", "AP50_95", "Recall"):
            xs = [r[k] for r in rows]
            print(f"{k}: {st.mean(xs):.1f} ± {st.stdev(xs):.1f}")


if __name__ == "__main__":
    main()
