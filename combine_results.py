#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_item(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--item must have the form 'label=path/to/image.png'"
        )
    # Split at the final equals sign so labels such as "eta=0" work.
    label, path = value.rsplit("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "--item label and path must both be non-empty"
        )
    return label.strip(), Path(path.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Combine experiment grids into one labeled image."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--item", type=parse_item, action="append", required=True)
    parser.add_argument(
        "--columns",
        type=int,
        help="number of panels per row (default: approximately square)",
    )
    args = parser.parse_args()

    if args.columns is not None and args.columns < 1:
        parser.error("--columns must be at least 1")

    panels = []
    for label, path in args.item:
        if not path.is_file():
            parser.error(f"image not found: {path}")
        panels.append((label, Image.open(path).convert("RGB")))

    columns = args.columns or math.ceil(math.sqrt(len(panels)))
    rows = math.ceil(len(panels) / columns)
    padding = 12
    title_height = 28
    panel_width = max(image.width for _, image in panels)
    panel_height = max(image.height for _, image in panels)
    canvas = Image.new(
        "RGB",
        (
            padding + columns * (panel_width + padding),
            padding + rows * (title_height + panel_height + padding),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (label, panel) in enumerate(panels):
        row, column = divmod(index, columns)
        x = padding + column * (panel_width + padding)
        y = padding + row * (title_height + panel_height + padding)
        draw.text((x, y + 7), label, fill="black", font=font)
        canvas.paste(panel, (x, y + title_height))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
