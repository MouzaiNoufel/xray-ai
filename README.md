# X-ray Baggage Screening Assistant: local training and evaluation

This is the Phase 0 Colab notebook turned into a project you can run on your own GPU.
It has one evaluation protocol shared by every experiment, config files instead of
copy-pasted cells, automatic resume, and the Pfa (false-alarm) measurement the
notebook could not do.

```
xray-ai/
  .vscode/                  editor settings, debug configs (F5), tasks, recommended extensions
  configs/
    local.example.yaml      -> copied to local.yaml by setup: YOUR dataset paths (only file to edit per machine)
    policy.yaml             risk tiers + thresholds (Safe / Manual / High Risk), per-class overrides
    taxonomy.yaml           unified class list + name aliases for merging datasets
    policy_unified.yaml     risk tiers for the unified classes
    experiments/            one YAML per experiment; each changes ONE thing vs _base.yaml
  xray/                     the library: data, policy, evaluate, audit, config
  scripts/
    check_gpu.py            is the GPU visible, how fast, which models fit
    smoke_test.py           whole pipeline on fake data in ~1 min: run after any install change
    check_dataset.py        run FIRST on real data: paths, class balance, image sizes, train/val LEAK check
    convert_coco.py         COCO JSON datasets (PIDray original, STCray, CLCXray...) -> YOLO format
    merge_datasets.py       several datasets -> one, on configs/taxonomy.yaml's 22 unified classes
    prepare_negatives.py    clean bags -> training backgrounds + held-out Pfa set
    train.py                config-driven training, auto-resume, auto-eval
    evaluate.py             evaluate any weights; --from-cache re-scores instantly
    compare.py              leaderboard of every experiment -> results/LEADERBOARD.md
    export_dashboard.py     JSON for the officer dashboard
    export_model.py         ONNX / TensorRT / OpenVINO export, with a correctness check
  tests/                    fast unit tests (no GPU): policy, metrics, labels, audit log
  setup_env.ps1 / .sh       one-command environment setup (Windows / Linux)
```

## 1. Setup in VS Code (once)

Install first: **NVIDIA driver** (latest Game Ready or Studio), **Python 3.12** from python.org (tick "Add python.exe to PATH"),
**Git**, and **VS Code**. You do not need to install the CUDA Toolkit: the PyTorch wheels bring their own CUDA.

1. Unzip the project to a short path without spaces, e.g. `D:\dev\xray-ai`. Keep datasets outside it, e.g. `D:\datasets\`.
2. VS Code > File > Open Folder > `xray-ai`. Click **Install All** on the recommended-extensions popup.
3. Open a terminal (Ctrl+`) and run:

   ```powershell
   # Windows
   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
   ```
   ```bash
   # Linux / WSL2
   bash setup_env.sh
   ```

   This creates `.venv`, installs PyTorch with CUDA (tries cu130, then cu128, then cu126), installs everything else,
   creates `configs/local.yaml`, installs the git hooks, and runs the GPU check.
   Older GPU (GTX 10xx) or kernel errors: re-run with `-Cuda cu126` (Windows) or `CUDA=cu126 bash setup_env.sh`.
4. Ctrl+Shift+P > **Python: Select Interpreter** > the one in `.venv`. New terminals now activate it automatically.
5. Verify: Terminal > Run Task > **Smoke test (whole pipeline)**. It must end with `SMOKE TEST PASSED on device=0`.
   (`device=cpu` means PyTorch cannot see the GPU: run the **GPU check** task and follow what it says.)

**Security:** the Colab notebook had a Roboflow API key hard-coded in it. Revoke that key in Roboflow settings
and never commit keys. Use an environment variable instead.

## 2. Daily workflow in VS Code

| Want to | Do |
|---|---|
| Run or debug a script | Run and Debug (Ctrl+Shift+D) > pick a config > F5. Breakpoints work in `scripts/` and `xray/` |
| Train an experiment | Terminal > Run Task > **Train experiment** > pick the config (or the debug config of the same name) |
| Watch training live | Run Task > **TensorBoard**, then open http://localhost:6006 |
| See results | `results/<experiment>/report.md` (Ctrl+Shift+V previews it) and Run Task > **Leaderboard** |
| Run tests | Testing panel (flask icon), or Run Task > **Run tests**. Run them before every commit |
| Lint / format | Automatic on save (Ruff). Also runs on every `git commit` via pre-commit |

Git: initialise once with `git init -b main`, commit, then push to a **private** GitHub repo. `.gitignore` already
keeps out datasets, weights, runs and `configs/local.yaml`; the pre-commit hook blocks files over 2 MB and private keys.
Never commit X-ray images from a real site.

Long trainings: start them from a terminal (`python scripts/train.py configs/experiments/e11_y26s_640.yaml`), not the
debugger, so closing VS Code or a debugger hiccup cannot kill a 10-hour run. If it stops anyway, run the same command
again: it resumes. Windows: set Power settings so the PC does not sleep during training.

## 3. Real data

```bash
# unzip PIDray (the Roboflow export you used on Colab) somewhere, e.g. D:/datasets/pidray
python scripts/check_dataset.py --data D:/datasets/pidray/data.yaml --phash 300
```

Read the output carefully:

* **Leak check.** If the same `xray_NNNNN` source image appears in train and valid, the Phase 0 numbers are optimistic. Fix the split before anything else.
* **Image sizes.** If they are around 448 px, the Roboflow export downscaled PIDray. That explains why 832 px gave nothing, and it hurts Lighter and Bullet directly. Get the **original full-resolution PIDray** (github.com/bywang2018/security-dataset). It also ships the official *easy / hard / hidden* test splits, which are the numbers papers and evaluators compare against.

Then copy `configs/local.example.yaml` to `configs/local.yaml` and set `data:` to the generated `data_local.yaml`.

## 3b. More data: STCray + original PIDray (recommended)

The strongest open setup for an airport product is two datasets merged on one class list (`configs/taxonomy.yaml`, 22 classes):

| Dataset | Why | Get it |
|---|---|---|
| **STCray** (2025) | Real airport scanner (ANER K8065), 46,642 scans, 21 threat classes incl. explosives, 3D-printed guns, syringes, batteries, screwdrivers; boxes + masks; Apache-2.0 on its card | huggingface.co/datasets/Naoufel555/STCray-Dataset (log in, accept terms, ~24 GB) |
| **PIDray original** | 47,677 images at full resolution, masks, official easy/hard/hidden test sets | github.com/bywang2018/security-dataset (academic use only) |
| SIXray negatives | ~1M clean bags for Pfa (step 6) | github.com/MeioJane/SIXray (academic use only) |

```powershell
# 1. Convert each to YOLO (COCO JSON in -> YOLO seg labels out). Check the real file names after download.
python scripts/convert_coco.py --out D:/datasets/yolo/pidray --mode seg `
    --split train D:/datasets/pidray/annotations/xray_train.json D:/datasets/pidray/train `
    --split val   D:/datasets/pidray/annotations/xray_test_hard.json D:/datasets/pidray/hard
# (same for STCray; if its annotations are not COCO JSON, inspect one file and adapt)

# 2. Merge on the unified classes; also writes data_val_<source>.yaml for per-scanner evaluation
python scripts/merge_datasets.py --out D:/datasets/merged `
    --source stcray D:/datasets/yolo/stcray/data.yaml --source pidray D:/datasets/yolo/pidray/data.yaml

# 3. Check, set data_merged in configs/local.yaml, train
python scripts/check_dataset.py --data D:/datasets/merged/data.yaml
python scripts/train.py configs/experiments/e30_y26s_merged.yaml
```

Merged experiments use `configs/policy_unified.yaml`. Unlabelled images in the eval split (STCray's ~950 clean scans)
are counted as clean bags automatically, so Pfa gets a first real-airport estimate even before SIXray.

**Licensing:** PIDray, SIXray, HiXray, CLCXray and OPIXray are academic-use only. Use them for research and demos.
The model you SELL must be trained on data you have rights to: STCray (verify its terms with the authors in writing)
plus your own Phase 1 data.

## 4. Re-establish the baseline (no training)

Copy your Colab weights from Drive (`xray_prototype_weights/`) into `weights/`:

```bash
python scripts/evaluate.py --name e00_phase0_448 --weights weights/best.pt --imgsz 448
python scripts/evaluate.py --name e01_phase0_640 --weights weights/exp1_res640_best.pt --imgsz 640
python scripts/compare.py
```

Every new model now has to beat these numbers under the exact same protocol.

## 5. Experiment plan (run in order, each answers one question)

| config | question | decide |
|---|---|---|
| `e10_v8n_640_long` | Was 15 epochs the bottleneck? (same model, 100 epochs) | If it gains a lot, Phase 0 conclusions about multi-scale need re-testing |
| `e11_y26s_640` | Newer + larger detector (YOLO26-s) | New default if better at similar latency |
| `e12_y26s_p2_640` | Extra stride-4 head for small objects | Watch Lighter, Bullet, **tiny/small recall** |
| `e13_y26s_seg_640` | Segmentation + copy-paste for overlap classes | Watch Wrench, Hammer, Pliers, Gun, Scissors |
| `e14_y26m_640` | Is capacity the limit? | Only if s >> n |
| `e20_best_bg` | Clean bags in training vs false alarms | Watch Pfa and Powerbank FP without Pd loss |

```bash
python scripts/train.py configs/experiments/e11_y26s_640.yaml
# interrupted? run the exact same command again -> resumes
python scripts/compare.py
```

Rough GPU guide at 640 px with `batch: -1` (auto): 8 GB VRAM is fine for n/s and tight for m. 12 GB or more handles m and seg.
One epoch on 61k images takes very roughly 5-15 min on a modern RTX card, so 100 epochs is an overnight-to-weekend run. Do a
`--fraction 0.1 --epochs 5` dry run first to measure it.

## 6. Pfa: closing the biggest gap

PIDray has zero clean bags, so Pfa cannot be measured from it. Steps:

1. Download clean images: the SIXray negative set (about 1M images, image-level labels, so take only the negatives), or any clean-bag set you can obtain.
2. `python scripts/prepare_negatives.py --src <clean folder> --out D:/datasets/negatives --data <data_local.yaml> --train 5000 --eval 5000`
3. Add `clean_dirs` and `data_with_backgrounds` to `configs/local.yaml`. Every evaluation now reports Pfa with a 95% confidence interval, the false alarms by class, and **operating points** (best Pd for a 2/5/10/20% Pfa budget).

SIXray comes from a different scanner (subway, not airport), so this is a *proxy* Pfa. It is still far more meaningful than synthetic shapes. The real number only comes from the partner airport's own machines in Phase 1.

## 7. Two Pd numbers: report both

* **Pd any-alarm** is what Phase 0 reported. A bag counts as caught if *anything* fired, even a wrong-class detection somewhere else in the bag.
* **Pd correct-alarm** counts a bag as caught only if a real threat was found with the right class and location (IoU ≥ 0.3).

Operationally the officer opens the bag either way. But in front of an evaluator, the correct-alarm number is the defensible one, and the gap between the two shows how much "luck" the model gets.

## 8. Tuning the policy without retraining

Edit `configs/policy.yaml` (for example `class_review_conf: {Bullet: 0.12}`) and run:

```bash
python scripts/evaluate.py --name e11_y26s_640 --from-cache
```

Predictions are cached at a 0.05 confidence floor, so re-scoring takes seconds.

## 9. Deployment export

```bash
python scripts/export_model.py --weights weights/e11_y26s_640.pt --format onnx
python scripts/export_model.py --weights weights/e11_y26s_640.pt --format engine --half   # on the target NVIDIA box
```

## License note

Ultralytics YOLO is AGPL-3.0. That is fine for research. **Selling** a closed product built on it needs an Ultralytics
Enterprise license, or a switch to a permissively licensed detector (for example RT-DETR implementations under Apache-2.0).
Settle this before the first commercial contract, not after.
