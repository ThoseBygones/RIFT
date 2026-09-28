from argparse import ArgumentParser
import os


def parse_bool(value):
    """Parse a command-line boolean while keeping old ``--resume True`` calls valid."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Invalid boolean value: {value}')


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # losses
        self.parser.add_argument('--cos_margin', default=0.4, type=float)
        self.parser.add_argument('--lambda_anony', default=0, type=float)
        self.parser.add_argument('--lambda_div', default=0, type=float)
        self.parser.add_argument('--lambda_intra', default=0, type=float)
        self.parser.add_argument('--lambda_inter', default=0, type=float)
        self.parser.add_argument('--lambda_forgery', default=0, type=float)
        self.parser.add_argument('--lambda_nll', default=0, type=float)

        # Training parameters
        self.parser.add_argument('--dataset_name', default='celeba', choices=['celeba'], help='Training dataset (the final pipeline uses CelebA only)')
        self.parser.add_argument('--data_path', default='./data/celeba', type=str, help='Path of the dataset')
        self.parser.add_argument('--metadata_path', default=None, type=str, help='Optional directory containing CelebA list and triplet files')
        self.parser.add_argument('--epochs', default=20, type=int, help='Number of epochs to train')
        self.parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU')
        self.parser.add_argument('--test_batch_size', default=32, type=int, help='Test batch size')
        self.parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')
        self.parser.add_argument('--image_size', default=64, type=int, help='training image size')

        # Pretrained Models
        self.parser.add_argument('--arcface_path', default='./pretrained/arcface.pt', type=str, help='Path of pretrained arcface')
        self.parser.add_argument('--cflow_path', default='./pretrained/cflow_attention.pt', type=str, help='Path of the pretrained attention-CFlow checkpoint')
        self.parser.add_argument('--flow_path', default='./pretrained/flow.pt', type=str, help='Path of the pretrained Flow checkpoint')
        self.parser.add_argument(
            '--identity_backbone',
            default='facenet',
            choices=['facenet', 'arcface'],
            help='Frozen face recognizer used by the Stage 3 identity losses; the paper configuration uses FaceNet',
        )

        # Network Parameters
        self.parser.add_argument('--n_levels', default=4, type=int, help='Levels of of the network, \
                            n_levels=5 for input_dim=128, n_levels=4 for input_dim=64, n_levels=3 for input_dim=32')
        self.parser.add_argument('--depth', default=32, type=int, help='Depth of the network')
        self.parser.add_argument('--width', default=256, type=int, help='Dimension of the hidden layers')
        self.parser.add_argument('--max_grad_norm', default=5, type=float, help='Max gradient norm for clipping')
        self.parser.add_argument('--feature_size', default=512, type=int, help='Dimension of feature size (the condition)')        
        self.parser.add_argument(
            '--cflow_variant',
            default='attention',
            choices=['attention', 'no_attention'],
            help='Conditional Flow variant; the paper uses attention',
        )
        # Other parameters
        self.parser.add_argument('--n_bits', default=5, type=int, help='Bits in dequant 1-8')
        self.parser.add_argument('--n_samples', default=10, type=int, help='Number of random generated image samples for validation')
        self.parser.add_argument('--warmup_epochs', default=1, type=int, help='Number of learning-rate warm-up epochs')
        self.parser.add_argument('--resume', nargs='?', const=True, default=False, type=parse_bool, help='Warm-start weights and epoch index from a checkpoint')
        self.parser.add_argument('--resume_ckpt_path', default=None, type=str, help='Path of a previously trained checkpoint')
        self.parser.add_argument('--exp_dir', default='./experiments/celeba/run', type=str, help='Directory of experiments')
        self.parser.add_argument("--temperature", default=0.7, type=float)
        self.parser.add_argument('--workers', default=8, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--device', default='cuda', type=str, help='Device used for non-DDP training')
        self.parser.add_argument('--seed', default=0, type=int, help='Random seed for Python, NumPy, and PyTorch')

        #distribution related
        self.parser.add_argument('--ddp', default=False,action="store_true")
        self.parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    def parse(self):
        opts = self.parser.parse_args()
        return opts
