"""COCO training.

train2017 / val2017, letterbox 640x640, horizontal flip p=0.5 only.
300 epochs, AdamW, weight decay 1e-4; lr 1e-5 for pre-existing backbone parameters,
1e-4 for neck, head and added modules; both x0.1 after epoch 200.
Micro-batch 2 x 8 accumulation steps = 16 images per update, AMP, gradient
norm clipping 0.1 applied after the accumulated gradient is divided by the
number of images in the window. The final (epoch 300) checkpoint is evaluated.

  python train_coco.py --model yolo11m.yaml
"""
import argparse

import torch
from torch import optim
from ultralytics import YOLO

from train_driver import AMSFNTrainer


class COCOTrainer(AMSFNTrainer):
    BACKBONE_LAST = 10
    LR_DROP = 200
    CLIP = 0.1

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        backbone, rest = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            idx = int(n.split(".")[1]) if n.startswith("model.") else 999
            (backbone if idx <= self.BACKBONE_LAST and ".block." not in n else rest).append(p)
        return optim.AdamW([{"params": backbone, "lr": 1e-5}, {"params": rest, "lr": 1e-4}],
                           weight_decay=1e-4)

    def _setup_scheduler(self):
        self.lf = lambda e: 1.0 if e < self.LR_DROP else 0.1
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self._win_imgs = getattr(self, "_win_imgs", 0) + batch["img"].shape[0]
        return batch

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        n = max(getattr(self, "_win_imgs", 1), 1)
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.div_(n)                      # normalise by images in the accumulation window
        self._win_imgs = 0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.CLIP)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    COCOTrainer.flags = dict(use_a=True, use_b=True, use_c=True)
    model = YOLO(a.weights or a.model)
    model.train(
        trainer=COCOTrainer, data=a.data, imgsz=640, epochs=a.epochs,
        batch=2, nbs=16,
        warmup_epochs=0, close_mosaic=0,
        mosaic=0.0, mixup=0.0, scale=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.5, flipud=0.0,
        amp=True, seed=42, device=a.device, workers=a.workers, project="runs_coco", name="amsfn",
        exist_ok=True,
    )
    m = YOLO(str(model.trainer.last)).val(data=a.data, imgsz=640, batch=16, device=a.device).box
    print(f"AP50:95={m.map*100:.1f}  AP50={m.map50*100:.1f}  AP75={m.map75*100:.1f}")


if __name__ == "__main__":
    main()
