# AMSFN

Adaptive Multi-Stage Feature Alignment Network for driver-behavior object detection with YOLO11.

The project includes three feature-processing modules:

- **A — ARFAM:** adaptive receptive-field processing with input-dependent expert routing.
- **B — CSSAM:** cross-scale semantic recalibration of spatial features.
- **C — DFRM:** feature refinement using EMA, squeeze-and-excitation and DySample.

## Installation

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CUDA installations, select the appropriate PyTorch build using the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v210).

## Project files

| File | Purpose |
|---|---|
| `amsfn_modules.py` | ARFAM, CSSAM, DFRM and insertion into the YOLO11 model |
| `train_driver.py` | Detection training, module ablation and multi-seed evaluation |
| `train_coco.py` | COCO training and evaluation |
| `check_model.py` | Parameter counts and GFLOPs for the base, A, B, C and ABC variants |
| `data_example.yaml` | Example dataset paths and class-name configuration |
| `requirements.txt` | Python dependencies |
| `LICENSE` | Project license |

## Dataset configuration

Prepare an annotated detection dataset in YOLO format:

```text
your_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each label file uses the same stem as its image and contains one object per line:

```text
class_id x_center y_center width height
```

Bounding-box coordinates are normalized to the image dimensions. Edit `path`, `train`, `val` and `names` in `data_example.yaml` to match your own images, annotations and partitions. The example uses six generic names, `class_0` through `class_5`; replace the names and class count to match your annotation scheme.

Dataset sources:

- [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)
- [AUC Distracted Driver Dataset](https://heshameraqi.github.io/distraction_detection)

Follow each provider's access and licensing conditions.

## Training

Train YOLO11s with all three modules:

```bash
python train_driver.py --data data_example.yaml --variant abc
```

The default settings use 200 epochs, batch size 2, input size 640 and seed 42. Use `--model`, `--epochs`, `--batch`, `--imgsz`, `--device` and `--workers` to configure a run. An optional pretrained checkpoint can be supplied with `--weights`.

### Module ablation and multiple seeds

Supported variants are `none`, `a`, `b`, `c`, `abc`, or other combinations such as `ab`.

```bash
python train_driver.py --data data_example.yaml --variant none --seeds 42 43 44
python train_driver.py --data data_example.yaml --variant a --seeds 42 43 44
python train_driver.py --data data_example.yaml --variant b --seeds 42 43 44
python train_driver.py --data data_example.yaml --variant c --seeds 42 43 44
python train_driver.py --data data_example.yaml --variant abc --seeds 42 43 44
```

Runs are saved under `runs_driver/`. The summary CSV contains one row per seed with AP50, AP75, AP50:95 and Recall. For multiple seeds, the script also prints each metric's mean and sample standard deviation.

### Checkpoint selection

The driver-training script evaluates the final checkpoint (`last.pt`) by default. Select `best.pt` with:

```bash
python train_driver.py --data data_example.yaml --variant abc --ckpt best
```

### COCO

The COCO script uses the `coco.yaml` configuration supplied by Ultralytics:

```bash
python train_coco.py --model yolo11m.yaml --data coco.yaml
```

Its default settings use 300 epochs, AdamW, a microbatch of 2 and an effective batch size of 16. The script evaluates the final checkpoint and saves runs under `runs_coco/`.

## Evaluate a trained model

Import the custom modules before loading a checkpoint:

```python
from ultralytics import YOLO
import amsfn_modules

model = YOLO("runs_driver/yolo11s_ABC_seed42/weights/last.pt")
metrics = model.val(data="data_example.yaml", imgsz=640)
print(metrics.box.map50, metrics.box.map75, metrics.box.map, metrics.box.mr)
```

Change the checkpoint path to the run and checkpoint you want to evaluate.

## Model complexity

Print parameter counts and GFLOPs at a 640 × 640 input size:

```bash
python check_model.py yolo11s.yaml 6
```

The second argument is the number of detection classes. Set it to match your dataset. The script reports the base, A, B, C and ABC variants.

## License

This project is distributed under AGPL-3.0-or-later; see [LICENSE](LICENSE). It uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and the components identified in the source files.
