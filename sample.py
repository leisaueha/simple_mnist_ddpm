#!/usr/bin/env python3
import argparse
import math
import shutil
import subprocess
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from ddpm import DDPM
from model import SimpleUNet


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_animation(frames, path, fps=24):
    """Write grayscale tensor frames in [0, 1] to an MP4 using ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("--save-animation requires ffmpeg to be installed")

    height, width = frames.shape[-2:]
    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    video = (
        frames.mul(255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .contiguous()
        .numpy()
    )
    try:
        subprocess.run(command, input=video.tobytes(), check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to write {path}") from exc


def parse_labels(value):
    try:
        return [int(label.strip()) for label in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--labels must be comma-separated integers"
        ) from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/mnist_ddpm.pt")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="number of DDPM or DDIM denoising steps (default: 1000)",
    )
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--ddim", action="store_true", help="use DDIM sampling")
    parser.add_argument(
        "--interpolation",
        action="store_true",
        help="denoise latent pairs and their midpoints as a comparison grid",
    )
    parser.add_argument(
        "--num-interpolations",
        type=int,
        default=1,
        help="number of latent pairs in an interpolation grid (default: 1)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("samples"))
    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument("--animation-seconds", type=float, default=5.0)
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for reproducible sampling",
    )
    parser.add_argument(
        "--labels",
        type=parse_labels,
        help="comma-separated classes to repeat (default: unconditional)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="0 is unconditional, 1 is conditional, >1 strengthens the class",
    )
    args = parser.parse_args()

    if args.num_images < 1:
        parser.error("--num-images must be at least 1")
    if args.num_steps < 1:
        parser.error("--num-steps must be at least 1")
    if args.num_interpolations < 1:
        parser.error("--num-interpolations must be at least 1")
    if args.num_interpolations != 1 and not args.interpolation:
        parser.error("--num-interpolations requires --interpolation")
    if args.eta < 0:
        parser.error("--eta must be non-negative")
    if args.eta != 0 and not args.ddim:
        parser.error("--eta only applies when --ddim is set")
    if args.animation_seconds <= 0:
        parser.error("--animation-seconds must be greater than 0")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    num_classes = checkpoint.get("num_classes")
    model = SimpleUNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model"])
    sample_count = (
        3 * args.num_interpolations
        if args.interpolation
        else args.num_images
    )
    labels = None
    if num_classes is None:
        if args.labels is not None:
            parser.error("--labels requires a classifier-free checkpoint")
        if args.guidance_scale != 2.0:
            parser.error("--guidance-scale requires a classifier-free checkpoint")
    else:
        if args.labels is None:
            labels = torch.full(
                (sample_count,), -1, device=device, dtype=torch.long
            )
        else:
            if not args.labels:
                parser.error("--labels cannot be empty")
            if any(label < 0 or label >= num_classes for label in args.labels):
                parser.error(f"--labels must be between 0 and {num_classes - 1}")
            labels = torch.tensor(
                [
                    args.labels[index % len(args.labels)]
                    for index in range(sample_count)
                ],
                device=device,
            )

    diffusion = DDPM(
        timesteps=checkpoint["timesteps"],
        device=device,
    )
    if args.num_steps > diffusion.timesteps:
        parser.error(
            "--num-steps cannot exceed the checkpoint timestep count "
            f"({diffusion.timesteps})"
        )

    sample_args = {}
    if args.save_animation:
        frame_count = max(1, round(args.animation_seconds * 24))
        sample_args["capture_steps"] = torch.linspace(
            0, args.num_steps - 1, frame_count
        ).round().long().tolist()

    if args.interpolation:
        endpoints = torch.randn(
            (args.num_interpolations, 2, 1, 28, 28), device=device
        )
        midpoint = (endpoints[:, 0] + endpoints[:, 1]) / 2.0
        # Row-major ordering produces rows [z1, z2, midpoint], with one
        # interpolation experiment per column.
        sample_args["initial_noise"] = torch.cat(
            (endpoints[:, 0], endpoints[:, 1], midpoint), dim=0
        )

    sampler = diffusion.sample_ddim if args.ddim else diffusion.sample
    sample_args["num_steps"] = args.num_steps
    if args.ddim:
        sample_args["eta"] = args.eta
    result = sampler(
        model,
        shape=(sample_count, 1, 28, 28),
        labels=labels,
        guidance_scale=args.guidance_scale,
        **sample_args,
    )
    if args.save_animation:
        images, frames = result
    else:
        images = result

    # Convert [-1, 1] back to [0, 1].
    images = (images + 1.0) / 2.0
    nrow = (
        args.num_interpolations
        if args.interpolation
        else max(1, math.ceil(math.sqrt(sample_count)))
    )
    grid_path = args.out_dir / "grid.png"
    save_image(images.cpu(), grid_path, nrow=nrow)

    if args.save_animation:
        frames = (frames + 1.0) / 2.0
        grid_frames = torch.stack([
            make_grid(frame, nrow=nrow)[0]
            for frame in frames
        ])
        save_animation(grid_frames, args.out_dir / "grid.mp4")
        save_animation(grid_frames.flip(0), args.out_dir / "grid_reverse.mp4")

    print(f"Saved samples to {args.out_dir}")


if __name__ == "__main__":
    main()
