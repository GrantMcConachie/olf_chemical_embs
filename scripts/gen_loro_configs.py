"""
Generate one training config per LORO receptor for an SGE array-job sweep.

Each receptor under data/CC/LORO/ is a self-contained dataset (train/val/test
CSVs). We stamp out one config per receptor from a base config, overriding only
the three fields that must differ between runs:

  * data_path    -> the receptor's leaf dir (train_lorax treats a leaf with no
                    subdirs as a single split)
  * results_path -> a per-receptor folder, so concurrent tasks don't race on the
                    shared saved_representations/ and config.yaml writes
  * log_path     -> a per-receptor tensorboard dir

A manifest.txt (one config path per line, sorted) lets the array job map
$SGE_TASK_ID -> config with `sed -n "${SGE_TASK_ID}p"`.

Usage:
    uv run python scripts/gen_loro_configs.py [base_config.yaml]
"""

import copy
import os
import sys

import yaml

BASE = sys.argv[1] if len(sys.argv) > 1 else "configs/config_default.yaml"
LORO_DIR = "data/CC/LORO"
OUT_DIR = "configs/loro_sweep_bp3"

with open(BASE) as f:
    base = yaml.safe_load(f)

receptors = sorted(
    d for d in os.listdir(LORO_DIR) if os.path.isdir(os.path.join(LORO_DIR, d))
)

os.makedirs(OUT_DIR, exist_ok=True)
manifest = []
for r in receptors:
    cfg = copy.deepcopy(base)
    cfg["training"]["data_path"] = f"{LORO_DIR}/{r}"
    cfg["training"]["save_path"] = (
        f"results/LORO_sweep_bp/{r}/seed{cfg['training']['seed']}"
    )

    path = os.path.join(OUT_DIR, f"{r}.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    manifest.append(path)

with open(os.path.join(OUT_DIR, "manifest.txt"), "w") as f:
    f.write("\n".join(manifest) + "\n")

print(f"Wrote {len(manifest)} configs to {OUT_DIR}/ (base: {BASE})")
print(f"Array range for qsub:  -t 1-{len(manifest)}")
