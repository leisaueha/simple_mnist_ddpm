Code written by chatGPT, for educational purpose, see the [blog post](https://leisaueha.github.io/posts/ddpm). `checkpoints/mnist_ddpm.pt` was trained for 100 epochs, using default settings. It took about 22s/epoch on MBP M3 Pro with 18GB RAM.

# Minimal MNIST DDPM

A deliberately small implementation of a class-conditional, classifier-free
noise-prediction DDPM:

1. Sample a clean MNIST image `x0`.
2. Sample a timestep `t`.
3. Add Gaussian noise to obtain `xt`.
4. Train a small U-Net to predict that noise.
5. Generate images by repeatedly denoising random Gaussian noise.

The model uses MNIST labels during training and randomly replaces 10% of them
with a learned null label. At sampling time, conditional and unconditional
noise predictions are combined to provide classifier-free guidance. There is
no EMA, attention, cosine schedule, mixed precision, or accelerated sampler.

## Setup

```bash
cd simple_mnist_ddpm

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Download and preprocess MNIST

```bash
python prepare_mnist.py
```

This creates:

```text
data/mnist_train.pt
```

The file contains `images` with shape `[60000, 1, 28, 28]` and values in
`[-1, 1]`, plus integer `labels`. Rerun this command if you have the older
image-only data file.

## Train

```bash
python train.py
```

On an Apple Silicon Mac, the script automatically uses MPS.

For a quick test:

```bash
python train.py --epochs 1 --timesteps 200
```

For the normal simple setup:

```bash
python train.py --epochs 10 --timesteps 1000
```

Change the classifier-free label dropout rate with
`--label-drop-prob` (the default is `0.1`).

Resume an interrupted run from the checkpoint and train to a total of 100
epochs:

```bash
python train.py --resume --epochs 100
```

Resume uses the path supplied by `--checkpoint`, restores the optimizer when
its state is available, and requires matching `--timesteps` and
`--num-classes`. Older checkpoints without optimizer state resume with a fresh
optimizer.

The latest checkpoint is written to:

```text
checkpoints/mnist_ddpm.pt
```

## Sample

```bash
python sample.py
```

This generates 8 unconditional digits using the learned null label:

```text
samples/grid.png
```

You can change the number of images:

```bash
python sample.py --num-images 16 --out-dir samples_16
```

Generate only selected digits or adjust guidance:

```bash
python sample.py --labels 3 --guidance-scale 3
python sample.py --labels 0,1,2 --num-images 12
```

Use a fixed seed to compare settings with the same random noise:

```bash
python sample.py --labels 1,3,5,7 --guidance-scale 2 --seed 2026
```

`--guidance-scale 0` samples from the unconditional prediction, `1` uses the
ordinary conditional prediction, and values above `1` strengthen conditioning.
When `--labels` is omitted, every image uses the null label (`-1`) and the
guidance scale has no effect.
Checkpoints created before classifier-free support remain sampleable without
labels or guidance.

Run the complete 36-image sampling suite:

```bash
bash run_sampling_cases.sh
```

The suite writes each case to `samples/<case-name>` with its grid, forward and
reverse animations, and a case-specific `readme.md`. It includes unconditional
sampling, each individual digit, all digits in one grid, and labels 1/3/5/7 at
guidance scales 0, 0.5, 1, 2, 4, and 8. The guidance sweep uses a shared seed so
its grids are directly comparable. Override the Python executable or checkpoint
when needed:

```bash
PYTHON=.venv/bin/python CHECKPOINT=checkpoints/custom.pt bash run_sampling_cases.sh
```

To also save five-second, 24 fps MP4s of the grid's generation process in
forward and reverse (`grid.mp4` and `grid_reverse.mp4`):

```bash
python sample.py --save-animation --out-dir samples_with_video
```

## Files

```text
prepare_mnist.py  Download MNIST and save normalized images and labels.
model.py          Class-conditional U-Net with timestep and label embeddings.
ddpm.py           Forward noising and guided reverse sampling equations.
train.py          Noise prediction with random label dropout.
sample.py         Generate class-guided grids of MNIST samples.
```
