#!/usr/bin/env python3
from pathlib import Path

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    dataset = MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Save one tensor file so the training script has no dataset logic.
    images = torch.stack([image for image, _ in dataset])
    labels = torch.tensor([label for _, label in dataset], dtype=torch.long)

    # DDPM convention: map pixels from [0, 1] to [-1, 1].
    images = images * 2.0 - 1.0

    output = data_dir / "mnist_train.pt"
    torch.save({"images": images, "labels": labels}, output)
    print(
        f"Saved {len(images)} labeled images with shape "
        f"{tuple(images.shape)} to {output}"
    )


if __name__ == "__main__":
    main()
