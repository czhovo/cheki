# cheki

Small experimental workspace for extracting and cleaning up cheki / polaroid
photos.

Main batch entry point:

```powershell
python extract_polaroid_batch.py imgs -o outs --debug
```

The current pipeline is:

```text
paper-frame detection -> quadrilateral fit -> perspective warp -> white balance
-> LAB denoise -> reduced USM low sharpen
```

`pipeline1.ipynb` is kept as old notebook context. The temporary comparison
outputs, super-resolution experiments, deconvolution experiments, and scoring
tools were removed.

Postprocessing details are documented in `POSTPROCESSING.md`.

## Environment

The workspace-local Conda environment lives at `.conda`.

Activate it with:

```powershell
conda activate C:\Users\20888\Desktop\cheki\.conda
```

Key pinned dependency:

```text
transformers==5.8.1
```

PyTorch is installed from the CUDA 13.0 wheel index:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

The current environment detects one CUDA device: NVIDIA GeForce RTX 3070 Ti
Laptop GPU.
