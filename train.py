#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ddpm import DDPM
from model import SimpleUNet


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mnist_train.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--checkpoint", default="checkpoints/mnist_ddpm.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    images = torch.load(args.data, map_location="cpu")
    loader = DataLoader(
        TensorDataset(images),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = SimpleUNet().to(device)
    diffusion = DDPM(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for (x0,) in progress:
            x0 = x0.to(device)
            t = torch.randint(
                0, args.timesteps, (x0.size(0),), device=device
            )
            noise = torch.randn_like(x0)
            xt = diffusion.add_noise(x0, t, noise)

            predicted_noise = model(xt, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x0.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(images)
        print(f"Epoch {epoch}: loss={average_loss:.6f}")

        checkpoint = Path(args.checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "timesteps": args.timesteps,
                "epoch": epoch,
            },
            checkpoint,
        )
        print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
