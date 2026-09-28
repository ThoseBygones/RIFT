from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """Load the headerless triplet list used to fine-tune RIFT.

    Each line in ``triples_celeba.txt`` contains three relative image paths:
    two images of the same identity followed by an image of another identity.
    The original 80/10/10 train/validation/test split is retained.
    """

    def __init__(
        self,
        root,
        transform=None,
        preprocess=None,
        dataset_name="celeba",
        mode="full",
        metadata_root=None,
    ):
        if dataset_name != "celeba":
            raise ValueError("The public training pipeline supports CelebA only.")
        if mode not in {"train", "val", "test", "full"}:
            raise ValueError(f"Unsupported split: {mode}")

        self.image_root = Path(root) / "celeba_classified"
        metadata_root = Path(metadata_root) if metadata_root else Path(root)
        self.triplets_path = metadata_root / "triples_celeba.txt"
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")
        if not self.triplets_path.is_file():
            raise FileNotFoundError(f"Triplet list not found: {self.triplets_path}")

        self.transform = transform
        self.preprocess = preprocess
        triplets = [
            tuple(line.split())
            for line in self.triplets_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(len(triplet) != 3 for triplet in triplets):
            raise ValueError(
                f"Every line must contain exactly three image paths: {self.triplets_path}"
            )

        train_end = int(0.8 * len(triplets))
        val_end = int(0.9 * len(triplets))
        splits = {
            "train": triplets[:train_end],
            "val": triplets[train_end:val_end],
            "test": triplets[val_end:],
            "full": triplets,
        }
        self.triplets = splits[mode]

    def _load_image(self, relative_path):
        image_path = self.image_root / relative_path
        image = (
            self.preprocess(str(image_path))
            if self.preprocess is not None
            else Image.open(image_path).convert("RGB")
        )
        return self.transform(image) if self.transform is not None else image

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        first, second, different = self.triplets[index]
        return (
            self._load_image(first),
            self._load_image(second),
            self._load_image(different),
        )
