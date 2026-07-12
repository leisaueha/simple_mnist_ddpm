Code written by chatGPT, for educational purpose, see the [blog post](https://leisaueha.github.io/posts/ddpm). `checkpoints/mnist_ddpm.pt` was trained for 100 epochs, using default settings. It took about 22s/epoch on MBP M3 Pro with 18GB RAM.

# Minimal MNIST DDPM

A deliberately small implementation of the original noise-prediction DDPM idea:

1. Sample a clean MNIST image `x0`.
2. Sample a timestep `t`.
3. Add Gaussian noise to obtain `xt`.
4. Train a small U-Net to predict that noise.
5. Generate images by repeatedly denoising random Gaussian noise.

There is no conditioning, classifier-free guidance, EMA, attention, cosine
schedule, mixed precision, or accelerated sampler.

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

The tensor has shape `[60000, 1, 28, 28]` and values in `[-1, 1]`.

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

The latest checkpoint is written to:

```text
checkpoints/mnist_ddpm.pt
```

## Sample

```bash
python sample.py
```

This generates 8 digits and writes them as a grid:

```text
samples/grid.png
```

You can change the number of images:

```bash
python sample.py --num-images 16 --out-dir samples_16
```

To also save five-second, 24 fps MP4s of the grid's generation process in
forward and reverse (`grid.mp4` and `grid_reverse.mp4`):

```bash
python sample.py --save-animation --out-dir samples_with_video
```

## Files

```text
prepare_mnist.py  Download MNIST and save one normalized tensor.
model.py          Small U-Net with sinusoidal timestep embeddings.
ddpm.py           Forward noising and reverse sampling equations.
train.py          Noise-prediction training loop.
sample.py         Generate and save a grid of MNIST samples.
```
