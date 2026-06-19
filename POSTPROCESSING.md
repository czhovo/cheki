# Postprocessing

The final postprocessing keeps only two conservative steps after perspective
warp and white balance.

## 1. LAB denoise

Algorithm: OpenCV `fastNlMeansDenoising` on LAB channels.

Parameters:

```text
L channel h = 3.5
A/B channels h = 6.0
templateWindowSize = 7
searchWindowSize = 21
```

This reduces fine luminance and chroma noise while avoiding the stronger
smearing seen in heavier NLM settings.

## 2. Reduced USM low sharpen

Algorithm: unsharp mask on the LAB `L` channel only.

Parameters:

```text
sigma = 1.0
amount = 0.45
threshold = 3.0
```

The sharpen step only reinforces medium-strength luminance edges. Color
channels are left unchanged, and `threshold` suppresses weak residual noise.
