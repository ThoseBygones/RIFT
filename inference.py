"""Generate RIFT virtual-identity images from one image or a directory."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models.cflow_attention import CFlow
from models.flow import Glow
from models.vid_generator import VIDGenerator
from util import require_paths, set_random_seed


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def is_within(path, directory):
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


class InputImages(Dataset):
    def __init__(self, input_path, image_size, exclude_path=None):
        self.input_path = Path(input_path).expanduser().resolve()
        exclude_path = Path(exclude_path).resolve() if exclude_path is not None else None
        if self.input_path.is_file():
            self.root = self.input_path.parent
            self.paths = [self.input_path]
        elif self.input_path.is_dir():
            self.root = self.input_path
            self.paths = sorted(
                path
                for path in self.input_path.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                and (exclude_path is None or not is_within(path, exclude_path))
            )
        else:
            raise FileNotFoundError(f"Input path not found: {self.input_path}")

        if not self.paths:
            raise ValueError(f"No supported images found in: {self.input_path}")

        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(160),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path.relative_to(self.root).as_posix()


class RIFTInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        args.device = self.device

        require_paths(
            flow_path=args.flow_path,
            cflow_path=args.cflow_path,
            rift_path=args.rift_path,
            arcface_path=args.arcface_path,
        )

        self.flow = Glow(
            bits=args.n_bits,
            width=args.width,
            depth=args.depth,
            n_levels=args.n_levels,
            input_dims=(3, args.image_size, args.image_size),
            lu_factorize=True,
        ).to(self.device)
        self.flow.load_state_dict(
            torch.load(args.rift_path, map_location=self.device)["net"]
        )
        self.flow.eval()

        self.cflow = CFlow(
            bits=args.n_bits,
            width=args.width,
            depth=args.depth,
            n_levels=args.n_levels,
            input_dims=(3, args.image_size, args.image_size),
            n_y=args.feature_size,
            lu_factorize=True,
        ).to(self.device)
        self.cflow.load_state_dict(
            torch.load(args.cflow_path, map_location=self.device)["net"]
        )
        self.cflow.eval()

        self.identity_generator = VIDGenerator(args, requires_grad=False).to(self.device)
        self.identity_generator.eval()

    def process_images(self, images):
        bins = 2**self.args.n_bits
        images = torch.floor(images * 255 / 2 ** (8 - self.args.n_bits))
        return images / bins + torch.rand_like(images) / bins

    def noise_shapes(self):
        channels = 3
        size = self.args.image_size
        shapes = []
        for _ in range(self.args.n_levels - 1):
            size //= 2
            channels *= 2
            shapes.append((channels, size, size))
        size //= 2
        shapes.append((channels * 4, size, size))
        return shapes

    def identity_noise(self, batch_size):
        return [
            torch.randn(batch_size, *shape, device=self.device)
            * self.args.identity_noise_std
            for shape in self.noise_shapes()
        ]

    @staticmethod
    def output_path(root, relative_path):
        path = root / Path(relative_path).with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @torch.no_grad()
    def run(self):
        output_root = Path(self.args.output_dir).expanduser().resolve()
        dataset = InputImages(
            self.args.input,
            self.args.image_size,
            exclude_path=output_root,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=self.device.type == "cuda",
        )

        original_root = output_root / "original"
        generated_root = output_root / "anonymized"

        for images, relative_paths in tqdm(loader, desc="Generating"):
            images = images.to(self.device, non_blocking=True)
            model_inputs = self.process_images(images)
            latents, _ = self.flow(model_inputs)

            for index, relative_path in enumerate(relative_paths):
                save_image(
                    images[index].clamp(0, 1),
                    self.output_path(original_root, relative_path),
                )

            for output_index in range(self.args.num_outputs):
                condition = self.identity_generator(self.identity_noise(images.size(0)))
                generated, _ = self.cflow.inverse(
                    latents,
                    self.args.feature_size,
                    condition,
                    z_std=self.args.temperature,
                )
                variant_root = (
                    generated_root
                    if self.args.num_outputs == 1
                    else generated_root / f"{output_index:02d}"
                )
                for index, relative_path in enumerate(relative_paths):
                    save_image(
                        generated[index].clamp(0, 1),
                        self.output_path(variant_root, relative_path),
                    )

        print(f"Processed {len(dataset)} image(s). Results: {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate virtual-identity images with pretrained RIFT models."
    )
    parser.add_argument("--input", required=True, help="Input image or directory.")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory.")
    parser.add_argument("--flow_path", default="./pretrained/flow.pt")
    parser.add_argument("--cflow_path", default="./pretrained/cflow_attention.pt")
    parser.add_argument("--rift_path", default="./pretrained/rift.pt")
    parser.add_argument("--arcface_path", default="./pretrained/arcface.pt")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_outputs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--identity_noise_std", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_bits", type=int, default=5)
    parser.add_argument("--n_levels", type=int, default=4)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--feature_size", type=int, default=512)
    args = parser.parse_args()
    if args.num_outputs < 1:
        parser.error("--num_outputs must be at least 1")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available; use --device cpu if needed.")
    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)
    RIFTInference(args).run()


if __name__ == "__main__":
    main()
