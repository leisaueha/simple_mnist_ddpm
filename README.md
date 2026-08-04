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
no EMA, attention, cosine schedule, or mixed precision. DDIM sampling is
available for faster generation and deterministic latent comparisons.

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

Use DDIM with fewer denoising steps (deterministic when `--eta 0`, the
default):

```bash
python sample.py --ddim --num-steps 50 --eta 0 --seed 2026
```

`--num-steps` defaults to 1000 and must not exceed the timestep count stored
in the checkpoint. It respaces either the DDPM or DDIM sampling schedule.
Increasing `--eta` makes DDIM stochastic, while `0` uses its deterministic
update.

To compare DDIM latent endpoints with their linear midpoint interpolations:

```bash
python sample.py --ddim --num-steps 50 --interpolation \
  --num-interpolations 8 --seed 2026 \
  --out-dir samples_interpolation
```

Interpolation mode writes a three-row grid. Each column is one experiment: the
top and middle images come from independently sampled endpoint latents and the
bottom image comes from their midpoint. `--num-interpolations` controls the
number of columns and defaults to one. `--num-images` is ignored in this mode.
Labels, guidance, and animation options still apply.

## Comparing DDPM and DDIM

Run the complete experiment suite and build all labeled comparison images with:

```bash
bash run_ddim_experiments.sh
```

The script uses both the 1000-timestep and 50-timestep checkpoints, generates
64 unconditional samples per setting, and uses seed 2026 so corresponding runs
begin from the same random noise. Each setting writes to its own folder under
`ddim_vs_ddpm/`, and the combined images are written to
`ddim_vs_ddpm/comparisons/`.

Override the Python executable, checkpoint, output directory, sample count, or
seed when needed:

```bash
PYTHON=.venv/bin/python CHECKPOINT=checkpoints/custom_t1000.pt \
  CHECKPOINT_T50=checkpoints/custom_t50.pt \
  EXPERIMENT_DIR=my_experiments NUM_IMAGES=36 SEED=7 \
  bash run_ddim_experiments.sh
```

The script performs the following experiments.

Compare ancestral DDPM with deterministic DDIM using models trained with
T=1000 and T=50. Both models use a 10-step sampling schedule, and the T=50
model is additionally sampled with its complete 50-step schedule:

```bash
python sample.py --checkpoint checkpoints/mnist_ddpm.pt --num-images 64 \
  --num-steps 10 --seed 2026 --out-dir ddim_vs_ddpm/ddpm_10_steps
python sample.py --checkpoint checkpoints/mnist_ddpm.pt --num-images 64 \
  --ddim --num-steps 10 --eta 0 --seed 2026 \
  --out-dir ddim_vs_ddpm/ddim_10_steps_eta_0
python sample.py --checkpoint checkpoints/mnist_ddpm_t50.pt --num-images 64 \
  --num-steps 10 --seed 2026 --out-dir ddim_vs_ddpm/ddpm_t50_10_steps
python sample.py --checkpoint checkpoints/mnist_ddpm_t50.pt --num-images 64 \
  --ddim --num-steps 10 --eta 0 --seed 2026 \
  --out-dir ddim_vs_ddpm/ddim_t50_10_steps_eta_0
python sample.py --checkpoint checkpoints/mnist_ddpm_t50.pt --num-images 64 \
  --num-steps 50 --seed 2026 --out-dir ddim_vs_ddpm/ddpm_t50_50_steps
python sample.py --checkpoint checkpoints/mnist_ddpm_t50.pt --num-images 64 \
  --ddim --num-steps 50 --eta 0 --seed 2026 \
  --out-dir ddim_vs_ddpm/ddim_t50_50_steps_eta_0
python combine_results.py --columns 2 \
  --item "T=1000 model: 10-step DDPM=ddim_vs_ddpm/ddpm_10_steps/grid.png" \
  --item "T=1000 model: 10-step DDIM=ddim_vs_ddpm/ddim_10_steps_eta_0/grid.png" \
  --item "T=50 model: 10-step DDPM=ddim_vs_ddpm/ddpm_t50_10_steps/grid.png" \
  --item "T=50 model: 10-step DDIM=ddim_vs_ddpm/ddim_t50_10_steps_eta_0/grid.png" \
  --item "T=50 model: 50-step DDPM=ddim_vs_ddpm/ddpm_t50_50_steps/grid.png" \
  --item "T=50 model: 50-step DDIM=ddim_vs_ddpm/ddim_t50_50_steps_eta_0/grid.png" \
  --output ddim_vs_ddpm/comparisons/ddpm_vs_ddim_10_steps.png
```

Compare DDIM stochasticity at 10 steps:

```bash
for eta in 0 0.25 0.5 0.75 1; do
  python sample.py --checkpoint checkpoints/mnist_ddpm.pt --num-images 64 \
    --ddim --num-steps 10 --eta "$eta" --seed 2026 \
    --out-dir "ddim_vs_ddpm/ddim_10_steps_eta_${eta}"
done
python combine_results.py --columns 3 \
  --item "eta=0=ddim_vs_ddpm/ddim_10_steps_eta_0/grid.png" \
  --item "eta=0.25=ddim_vs_ddpm/ddim_10_steps_eta_0.25/grid.png" \
  --item "eta=0.5=ddim_vs_ddpm/ddim_10_steps_eta_0.5/grid.png" \
  --item "eta=0.75=ddim_vs_ddpm/ddim_10_steps_eta_0.75/grid.png" \
  --item "eta=1=ddim_vs_ddpm/ddim_10_steps_eta_1/grid.png" \
  --output ddim_vs_ddpm/comparisons/ddim_eta.png
```

Compare deterministic DDIM across inference step counts:

```bash
for steps in 10 50 100 250 500 1000; do
  python sample.py --checkpoint checkpoints/mnist_ddpm.pt --num-images 64 \
    --ddim --num-steps "$steps" --eta 0 --seed 2026 \
    --out-dir "ddim_vs_ddpm/ddim_${steps}_steps_eta_0"
done
python combine_results.py --columns 3 \
  --item "10 steps=ddim_vs_ddpm/ddim_10_steps_eta_0/grid.png" \
  --item "50 steps=ddim_vs_ddpm/ddim_50_steps_eta_0/grid.png" \
  --item "100 steps=ddim_vs_ddpm/ddim_100_steps_eta_0/grid.png" \
  --item "250 steps=ddim_vs_ddpm/ddim_250_steps_eta_0/grid.png" \
  --item "500 steps=ddim_vs_ddpm/ddim_500_steps_eta_0/grid.png" \
  --item "1000 steps=ddim_vs_ddpm/ddim_1000_steps_eta_0/grid.png" \
  --output ddim_vs_ddpm/comparisons/ddim_steps.png
```

Generate an unconditional interpolation strip from two random latents and
their midpoint:

```bash
python sample.py --checkpoint checkpoints/mnist_ddpm.pt --ddim \
  --num-steps 10 --eta 0 --interpolation --num-interpolations 12 --seed 2026 \
  --out-dir ddim_vs_ddpm/ddim_interpolation_10_steps
python combine_results.py --columns 1 \
  --item "Rows: endpoint 1, endpoint 2, midpoint=ddim_vs_ddpm/ddim_interpolation_10_steps/grid.png" \
  --output ddim_vs_ddpm/comparisons/ddim_interpolation.png
```

No `--labels` argument is passed in these experiments, so the checkpoint's
learned null label is used and interpolation is free to move between digits.

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
ddpm.py           Forward noising plus guided DDPM and DDIM sampling equations.
train.py          Noise prediction with random label dropout.
sample.py         Generate class-guided grids of MNIST samples.
combine_results.py Combine saved grids into one labeled comparison image.
run_ddim_experiments.sh Run and combine the DDPM/DDIM experiment suite.
```
