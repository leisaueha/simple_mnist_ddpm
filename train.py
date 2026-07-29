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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--checkpoint", default="checkpoints/mnist_ddpm.pt")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from --checkpoint; --epochs is the total target epoch",
    )
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--label-drop-prob",
        type=float,
        default=0.3,
        help="probability of replacing a label with the learned null class",
    )
    args = parser.parse_args()
    if args.num_classes < 1:
        parser.error("--num-classes must be at least 1")
    if not 0.0 <= args.label_drop_prob <= 1.0:
        parser.error("--label-drop-prob must be between 0 and 1")

    device = get_device()
    print(f"Using device: {device}")

    data = torch.load(args.data, map_location="cpu")
    if not isinstance(data, dict) or not {"images", "labels"} <= data.keys():
        parser.error(
            f"{args.data} is an old image-only dataset; rerun "
            "python prepare_mnist.py to add labels"
        )
    images, labels = data["images"], data["labels"]
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = SimpleUNet(num_classes=args.num_classes).to(device)
    diffusion = DDPM(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1

    if args.resume:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            parser.error(f"cannot resume: checkpoint not found: {checkpoint_path}")

        saved = torch.load(checkpoint_path, map_location=device)
        saved_timesteps = saved.get("timesteps")
        saved_num_classes = saved.get("num_classes")
        if saved_timesteps != args.timesteps:
            parser.error(
                "cannot resume with a different timestep count "
                f"(checkpoint: {saved_timesteps}, requested: {args.timesteps})"
            )
        if saved_num_classes != args.num_classes:
            parser.error(
                "cannot resume with a different class count "
                f"(checkpoint: {saved_num_classes}, requested: {args.num_classes})"
            )

        model.load_state_dict(saved["model"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        else:
            print("Checkpoint has no optimizer state; using a fresh optimizer")

        start_epoch = saved.get("epoch", 0) + 1
        print(f"Resuming {checkpoint_path} from epoch {start_epoch}")

    if start_epoch > args.epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch - 1}; "
            f"target is {args.epochs}, nothing to do"
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x0, labels in progress:
            x0 = x0.to(device)
            labels = labels.to(device)
            dropped = torch.rand(labels.shape, device=device) < args.label_drop_prob
            model_labels = labels.masked_fill(dropped, -1)
            t = torch.randint(
                0, args.timesteps, (x0.size(0),), device=device
            )
            noise = torch.randn_like(x0)
            xt = diffusion.add_noise(x0, t, noise)

            predicted_noise = model(xt, t, model_labels)
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
                "optimizer": optimizer.state_dict(),
                "timesteps": args.timesteps,
                "epoch": epoch,
                "num_classes": args.num_classes,
                "label_drop_prob": args.label_drop_prob,
            },
            checkpoint,
        )
        print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
