"""Check that PyTorch sees your GPU, measure it, and suggest settings.

python scripts/check_gpu.py
"""

from __future__ import annotations

import platform
import sys
import time


def main():
    print(f"Python      {sys.version.split()[0]}  ({platform.system()} {platform.release()})")
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch is not installed. Run the setup script first.")
    print(f"PyTorch     {torch.__version__}  (built for CUDA {torch.version.cuda})")
    try:
        import ultralytics

        print(f"Ultralytics {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics NOT installed")

    if not torch.cuda.is_available():
        print("\nCUDA available: NO")
        print("  -> Training would run on the CPU, about 50x slower. Usual causes:")
        print("     1. PyTorch CPU build installed: reinstall with the CUDA index URL (see README, step 3)")
        print("     2. NVIDIA driver missing or too old: run 'nvidia-smi' in a terminal")
        sys.exit(1)

    i = 0
    p = torch.cuda.get_device_properties(i)
    vram = p.total_memory / 1024**3
    print(f"\nCUDA available: yes   GPU: {p.name}   VRAM: {vram:.1f} GB   compute capability {p.major}.{p.minor}")

    # quick speed test (fp16 matmul), also proves kernels actually run on this card
    try:
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"\nGPU found but kernels fail: {e}")
        print("  -> This PyTorch build does not support your GPU (common on GTX 10xx and older).")
        print("     Re-run setup with an older CUDA build: -Cuda cu126 (Windows) or CUDA=cu126 (Linux).")
        sys.exit(1)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        a @ b
    torch.cuda.synchronize()
    tflops = 2 * 4096**3 * n / (time.perf_counter() - t0) / 1e12
    print(f"FP16 matmul: {tflops:.1f} TFLOPS (works)")

    print("\nSuggested settings at imgsz 640 (batch: -1 picks the batch size automatically):")
    if vram < 6:
        print("  <6 GB: yolo26n / yolov8n only, batch 8. Use imgsz 512 if you run out of memory.")
    elif vram < 10:
        print("  6-10 GB: yolo26n and yolo26s (e10-e13). Skip e14 (m), or set batch: 8 for it.")
    elif vram < 16:
        print("  10-16 GB: every experiment, including e14 (m) and segmentation.")
    else:
        print("  16 GB+: every experiment. You can also try yolo26l, or cache: ram if you have 64 GB of system RAM.")
    if platform.system() == "Windows":
        print("  Windows: if the dataloader is slow or crashes, set workers: 4 in configs/experiments/_base.yaml")


if __name__ == "__main__":
    main()
