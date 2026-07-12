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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/mnist_ddpm.pt")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("samples"))
    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument("--animation-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if args.num_images < 1:
        parser.error("--num-images must be at least 1")
    if args.animation_seconds <= 0:
        parser.error("--animation-seconds must be greater than 0")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = SimpleUNet().to(device)
    model.load_state_dict(checkpoint["model"])

    diffusion = DDPM(
        timesteps=checkpoint["timesteps"],
        device=device,
    )

    sample_args = {}
    if args.save_animation:
        frame_count = max(1, round(args.animation_seconds * 24))
        sample_args["capture_steps"] = torch.linspace(
            0, diffusion.timesteps - 1, frame_count
        ).round().long().tolist()

    result = diffusion.sample(
        model,
        shape=(args.num_images, 1, 28, 28),
        **sample_args,
    )
    if args.save_animation:
        images, frames = result
    else:
        images = result

    # Convert [-1, 1] back to [0, 1].
    images = (images + 1.0) / 2.0
    nrow = max(1, math.ceil(math.sqrt(args.num_images)))
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
