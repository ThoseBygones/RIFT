import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class SimpleDataset(Dataset):
    """Load every image below ``<data_root>/celeba_classified``."""

    def __init__(self, data_root, transform, randomized=False):
        self.image_root = Path(data_root) / "celeba_classified"
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")

        self.transform = transform
        self.images = [
            image_path.relative_to(self.image_root)
            for identity_dir in sorted(self.image_root.iterdir())
            if identity_dir.is_dir()
            for image_path in sorted(identity_dir.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if randomized and self.images:
            random.shuffle(self.images)
            self.images = self.images[: max(1, int(len(self.images) * 0.01))]

        print(f"Number of images: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.image_root / self.images[index]).convert("RGB")
        return self.transform(image)


class SimpleCelebA(Dataset):
    """Load one CelebA split from ``list_celeba_<mode>.txt``."""

    def __init__(self, data_root, transform, mode="train", metadata_root=None):
        self.image_root = Path(data_root) / "celeba_classified"
        metadata_root = Path(metadata_root) if metadata_root else Path(data_root)
        self.list_path = metadata_root / f"list_celeba_{mode}.txt"
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")
        if not self.list_path.is_file():
            raise FileNotFoundError(f"Image list not found: {self.list_path}")

        self.transform = transform
        lines = self.list_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"Image list is empty: {self.list_path}")
        self.num_classes = int(lines[0].split()[0])
        self.samples = [line.split()[0] for line in lines[1:] if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = Image.open(self.image_root / self.samples[index]).convert("RGB")
        return self.transform(image)
