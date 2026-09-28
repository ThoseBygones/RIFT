"""Prepare the image and triplet lists expected by the RIFT trainers."""

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def find_identities(image_root):
    identities = []
    for identity_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in identity_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) >= 2:
            identities.append((identity_dir.name, images))
    if len(identities) < 2:
        raise ValueError("At least two identities with two images each are required.")
    return identities


def relative_image_path(identity_name, image_path):
    return (Path(identity_name) / image_path.name).as_posix()


def write_image_list(output_path, identities, label_offset):
    lines = [str(len(identities))]
    for local_label, (identity_name, images) in enumerate(identities):
        label = label_offset + local_label
        lines.extend(
            f"{relative_image_path(identity_name, image_path)} {label}"
            for image_path in images
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_triplets(output_path, identities):
    lines = []
    for index, (identity_name, images) in enumerate(identities):
        other_name, other_images = identities[(index + 1) % len(identities)]
        first = relative_image_path(identity_name, images[0])
        second = relative_image_path(identity_name, images[1])
        other_first = relative_image_path(other_name, other_images[0])
        other_second = relative_image_path(other_name, other_images[1])
        lines.extend(
            [
                f"{first} {second} {other_first}",
                f"{first} {second} {other_second}",
                f"{other_first} {other_second} {first}",
                f"{other_first} {other_second} {second}",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def prepare(data_root, output_dir=None):
    data_root = Path(data_root).expanduser().resolve()
    image_root = data_root / "celeba_classified"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    output_root = (
        Path(output_dir).expanduser().resolve() if output_dir else data_root
    )
    output_root.mkdir(parents=True, exist_ok=True)

    identities = find_identities(image_root)
    train_end = int(0.8 * len(identities))
    val_end = int(0.9 * len(identities))
    split_identities = {
        "train": identities[:train_end],
        "val": identities[train_end:val_end],
        "test": identities[val_end:],
    }

    offset = 0
    for split, split_items in split_identities.items():
        output_path = output_root / f"list_celeba_{split}.txt"
        write_image_list(output_path, split_items, offset)
        offset += len(split_items)
        print(f"Wrote {output_path} ({len(split_items)} identities)")

    triplet_path = output_root / "triples_celeba.txt"
    triplet_count = write_triplets(triplet_path, identities)
    print(f"Wrote {triplet_path} ({triplet_count} triplets)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CelebA lists for the three-stage RIFT training pipeline."
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Dataset root containing celeba_classified/.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Metadata output directory (defaults to data_root).",
    )
    args = parser.parse_args()
    prepare(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
