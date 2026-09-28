from dataset.simple_dataset import SimpleCelebA
from dataset.triplet_dataset import TripletDataset
from models.arcface import ResNet
from models.cflow import CFlow as CFlowWithoutAttention
from models.cflow_attention import CFlow as AttentionCFlow
from models.flow import Glow
from models.face_recognizer import FaceRecognizer
from models.vid_generator import VIDGenerator
from options.train_options import TrainOptions
from skimage.metrics import structural_similarity as SSIM, peak_signal_noise_ratio as PSNR
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
from util import AverageMeter, require_paths, set_random_seed
import json
import math
import numpy as np
import os
import pprint
import torch
import torch.distributions as D
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


def build_identity_backbone(opts, device):
    """Build the frozen recognizer used by the Stage 3 identity losses."""
    if opts.identity_backbone == 'facenet':
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError as error:
            raise ImportError(
                "FaceNet was selected but facenet-pytorch is not installed. "
                "Install it with `pip install facenet-pytorch`."
            ) from error

        backbone = InceptionResnetV1(
            classify=False,
            pretrained='vggface2',
        ).to(device)
        input_size = 160
        standardize_input = True
    else:
        backbone = ResNet().to(device)
        backbone.load_state_dict(
            torch.load(opts.arcface_path, map_location='cpu')
        )
        input_size = 112
        standardize_input = False

    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone, input_size, standardize_input


@torch.no_grad()
def cal_psnr_ssim(x_ori, x_rec):
    batch_size = x_ori.size(0)

    original_pil = x_ori.clone().cpu().detach().numpy().transpose(0, 2, 3, 1)
    recover_pil = x_rec.cpu().detach().numpy().transpose(0, 2, 3, 1)

    # PSNR of original images and recovered images
    psnr = np.zeros(batch_size)
    for i in range(batch_size):
        psnr[i] = PSNR(recover_pil[i], original_pil[i])
    psnr = psnr.mean()

    # SSIM of cover images and container images
    ssim = np.zeros(batch_size)
    for i in range(batch_size):
        ssim[i] = SSIM(recover_pil[i], original_pil[i], channel_axis=2)
    ssim = ssim.mean()
    
    return psnr, ssim


class RIFTTrainer:
    def __init__(self, opts):
        self.opts = opts
        self.global_steps = 0
        require_paths(
            data_path=self.opts.data_path,
            arcface_path=self.opts.arcface_path,
            cflow_path=self.opts.cflow_path,
            flow_path=self.opts.flow_path,
        )
        
        print('local rank', self.opts.local_rank)
        if not self.opts.ddp:
            self.device = torch.device(self.opts.device)
            self.opts.device = self.device

        if self.opts.ddp:
            self.device = torch.device("cuda", self.opts.local_rank)
            self.opts.device = self.device
            dist.init_process_group(backend='nccl')
            print('dist related')
            print('dist world size', dist.get_world_size())
            print('dist rank', dist.get_rank())

        (
            self.identity_backbone,
            identity_input_size,
            standardize_identity_input,
        ) = build_identity_backbone(self.opts, self.device)
        self.face_recognizer = FaceRecognizer(
            opts,
            backbone=self.identity_backbone,
            input_size=identity_input_size,
            standardize_input=standardize_identity_input,
            requires_grad=False,
        )
        
        cflow_class = (
            AttentionCFlow
            if self.opts.cflow_variant == 'attention'
            else CFlowWithoutAttention
        )
        self.cflow = cflow_class(
            bits=self.opts.n_bits, 
            width=self.opts.width, 
            depth=self.opts.depth, 
            n_levels=self.opts.n_levels, 
            input_dims=(3, self.opts.image_size, self.opts.image_size), 
            n_y=self.opts.feature_size, 
            lu_factorize=True
        ).to(self.device)
        self.cflow.load_state_dict(torch.load(self.opts.cflow_path, map_location=self.device)['net'])
        self.cflow.eval()

        self.flow = Glow(
            bits=self.opts.n_bits, 
            width=self.opts.width, 
            depth=self.opts.depth, 
            n_levels=self.opts.n_levels, 
            input_dims=(3, self.opts.image_size, self.opts.image_size), 
            lu_factorize=True
        ).to(self.device)
        self.flow.load_state_dict(torch.load(self.opts.flow_path, map_location=self.device)['net'])

        self.vid_generator = VIDGenerator(
            opts=self.opts,
            requires_grad=False
        ).to(self.device)

        for param in self.cflow.parameters():
            param.requires_grad = False

        if self.opts.ddp:
            self.cflow = nn.parallel.DistributedDataParallel(self.cflow, device_ids=[self.opts.local_rank], output_device=self.opts.local_rank, broadcast_buffers=False, find_unused_parameters=True)
            self.flow = nn.parallel.DistributedDataParallel(self.flow, device_ids=[self.opts.local_rank], output_device=self.opts.local_rank, broadcast_buffers=False, find_unused_parameters=True)
            self.vid_generator = nn.parallel.DistributedDataParallel(self.vid_generator, device_ids=[self.opts.local_rank], output_device=self.opts.local_rank, broadcast_buffers=False, find_unused_parameters=True)
            
        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.val_dataset, self.simple_dataset = self.configure_datasets()

        # Sampler
        train_sampler, val_sampler, simple_sampler = None, None, None
        if self.opts.ddp:
            train_sampler = dist.DistributedSampler(self.train_dataset)
            val_sampler = dist.DistributedSampler(self.val_dataset)
            simple_sampler = dist.DistributedSampler(self.simple_dataset)
        
        self.train_dataloader = DataLoader(self.train_dataset,
                                            batch_size=self.opts.batch_size,
                                            shuffle=(train_sampler is None),
                                            num_workers=int(self.opts.workers),
                                            sampler=train_sampler,
                                            drop_last=False)
        self.val_dataloader = DataLoader(self.val_dataset,
                                            batch_size=self.opts.test_batch_size,
                                            shuffle=False,
                                            num_workers=int(self.opts.workers),
                                            sampler=val_sampler,
                                            drop_last=False)
        self.simple_dataloader = DataLoader(self.simple_dataset,
                                            batch_size=self.opts.test_batch_size,
                                            shuffle=False,
                                            num_workers=int(self.opts.workers),
                                            sampler=simple_sampler,
                                            drop_last=False)
        
        self.loss_meter = AverageMeter()

        # Initialize tensorboard
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.logger = SummaryWriter(log_dir=log_dir)

        self.save_img_dir = os.path.join(opts.exp_dir, 'val_images')
        os.makedirs(self.save_img_dir, exist_ok=True)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_parameter_number(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_num, trainable_num
    
    def configure_optimizers(self):
        params = [{'params': self.flow.parameters()}]
        optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        return optimizer
        
    def configure_datasets(self):
        # Load train, validation and test images
        trans = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize((self.opts.image_size, self.opts.image_size)),
            transforms.ToTensor(),
        ])
        
        train_dataset = TripletDataset(
            self.opts.data_path, 
            transform=trans, 
            dataset_name=self.opts.dataset_name, 
            mode='train',
            metadata_root=self.opts.metadata_path,
        )
        val_dataset = TripletDataset(
            self.opts.data_path, 
            transform=trans, 
            dataset_name=self.opts.dataset_name, 
            mode='val',
            metadata_root=self.opts.metadata_path,
        )
        simple_dataset = SimpleCelebA(
            self.opts.data_path,
            transform=trans,
            mode='val',
            metadata_root=self.opts.metadata_path,
        )
        
        print("Number of training triplets: {}".format(len(train_dataset)), flush=False)
        print("Number of validation triplets: {}".format(len(val_dataset)), flush=False)
        print("Number of validation samples: {}".format(len(simple_dataset)), flush=False)
        return train_dataset, val_dataset, simple_dataset
    
    def load_prev_checkpoint(self):
        print(f'Loading previously trained model from path: {self.opts.resume_ckpt_path}...')
        
        prev_checkpoint = torch.load(self.opts.resume_ckpt_path, map_location='cpu')
        self.flow.load_state_dict(prev_checkpoint['net'])

        current_epoch = prev_checkpoint['epoch'] + 1
        prev_checkpoint = None
        print(f'Resuming training from step {current_epoch}')
        return current_epoch
    
    def save_checkpoint(self, epoch, log_dict, is_best=False):
        if is_best:
            save_name = 'best.pt'
        else:
            save_name = 'epoch_{}.pt'.format(epoch)
        state_dict = {
            'net': self.flow.state_dict(),
            'opts': vars(self.opts), 
            'epoch': epoch
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(state_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write('**Best**: Epoch {}: Metrics - {}\n'.format(epoch, log_dict))
            else:
                f.write('Epoch {} - {}\n'.format(epoch, log_dict))

    def process_img(self, x):    # De-quantize and convert to logits
        n_bins = 2**self.opts.n_bits
        x = torch.floor(x * 255 / 2 ** (8 - self.opts.n_bits))
        x = x / n_bins
        x = x + torch.rand_like(x) / n_bins
        return x
    
    def calc_z_shapes(self, n_channel, input_size, n_levels):
        z_shapes = []
        for i in range(n_levels - 1):
            input_size //= 2
            n_channel *= 2
            z_shapes.append((n_channel, input_size, input_size))
        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))
        return z_shapes
    
    def gen_noise(self, b_size, sigma=1.0):
        z_sample = []
        z_shapes = self.calc_z_shapes(3, self.opts.image_size, self.opts.n_levels)
        for z in z_shapes:
            z_new = torch.randn(b_size, *z) * sigma
            z_sample.append(z_new.to(self.device))
        return z_sample
    
    def lr_schedule(self, global_steps):
        if global_steps <= self.opts.warmup_epochs * len(self.train_dataloader):
            self.optimizer.param_groups[0]['lr'] = (self.opts.learning_rate - 1e-6) * min(1, (global_steps + 1) / (self.opts.warmup_epochs * len(self.train_dataloader))) + 1e-6
        elif global_steps > self.opts.warmup_epochs * 3 * len(self.train_dataloader):
            self.optimizer.param_groups[0]['lr'] = max(self.opts.learning_rate * 0.9999**(global_steps - self.opts.warmup_epochs * 3 * len(self.train_dataloader)), 1e-6)
        else:
            self.optimizer.param_groups[0]['lr'] = self.opts.learning_rate
    
    def calc_flow_loss(self, x, bits_per_pixel=False):
        zs, logdet = self.flow(x)
        base_dist_mean, base_dist_var = torch.zeros(1).to(self.device), torch.ones(1).to(self.device)
        base_dist = D.Normal(base_dist_mean, base_dist_var)
        log_prob = sum(base_dist.log_prob(z).sum([1,2,3]) for z in zs) + logdet
        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel())
        return zs, -log_prob
    
    def calc_loss(self, x1, x2, y, x1k1, x1k2, x2k, yk1, yk2):
        loss_dict = {
            'loss_anonymity': 0.0,
            'loss_diversity': 0.0,
            'loss_intra-id': 0.0,
            'loss_inter-id': 0.0,
            'loss_id-forgery': 0.0,
        }
        loss = 0.0

        # NLL loss of flow
        _, loss_nll_x1 = self.calc_flow_loss(x1, bits_per_pixel=True)
        _, loss_nll_x2 = self.calc_flow_loss(x2, bits_per_pixel=True)
        _, loss_nll_y = self.calc_flow_loss(y, bits_per_pixel=True)
        loss_nll_x1, loss_nll_x2, loss_nll_y = loss_nll_x1.mean(), loss_nll_x2.mean(), loss_nll_y.mean()
        loss_nll = (loss_nll_x1 + loss_nll_x2 + loss_nll_y) / 3
        loss_dict['loss_nll'] = float(loss_nll) * self.opts.lambda_nll
        loss += loss_nll * self.opts.lambda_nll

        if torch.isnan(x1k1).any() or torch.isnan(x1k2).any() or torch.isnan(x2k).any() or torch.isnan(yk1).any() or torch.isnan(yk2).any():
            loss = loss * 5.0

            loss_dict['loss'] = float(loss)
            return loss, loss_dict

        # Anonymity loss
        if self.opts.lambda_anony > 0:
            _, loss_anony_x1v1 = self.face_recognizer(x1, x1k1)
            _, loss_anony_x1v2 = self.face_recognizer(x1, x1k2)
            loss_anonymity = loss_anony_x1v1 + loss_anony_x1v2
            loss_anonymity = loss_anonymity / 2
            loss_dict['loss_anonymity'] = float(loss_anonymity) * self.opts.lambda_anony
            loss += loss_anonymity * self.opts.lambda_anony

        # Diversity loss
        if self.opts.lambda_div > 0:
            _, loss_diversity = self.face_recognizer(x1k1, x1k2)
            loss_dict['loss_diversity'] = float(loss_diversity) * self.opts.lambda_div
            loss += loss_diversity * self.opts.lambda_div

        # Intra-identity loss
        if self.opts.lambda_intra > 0:
            loss_id_intra, _ = self.face_recognizer(x1k1, x2k)
            loss_dict['loss_intra-id'] = float(loss_id_intra) * self.opts.lambda_intra
            loss += loss_id_intra * self.opts.lambda_intra

        # Inter-identity loss
        if self.opts.lambda_inter > 0:
            _, loss_id_inter_x1y = self.face_recognizer(x1k1, yk2)
            _, loss_id_inter_x2y = self.face_recognizer(x2k, yk2)
            loss_id_inter = loss_id_inter_x1y + loss_id_inter_x2y
            loss_id_inter = loss_id_inter / 2
            loss_dict['loss_inter-id'] = float(loss_id_inter) * self.opts.lambda_inter
            loss += loss_id_inter * self.opts.lambda_inter

        # Forgery-resistance loss
        if self.opts.lambda_forgery > 0:
            _, loss_id_forgery_x1y = self.face_recognizer(x1k1, yk1)
            _, loss_id_forgery_x2y = self.face_recognizer(x2k, yk1)
            loss_id_forgery = loss_id_forgery_x1y + loss_id_forgery_x2y
            loss_id_forgery = loss_id_forgery / 2
            loss_dict['loss_id-forgery'] = float(loss_id_forgery) * self.opts.lambda_forgery
            loss += loss_id_forgery * self.opts.lambda_forgery
        
        loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def train(self):
        start_epoch = 0
        min_eer = 1

        if self.opts.resume_ckpt_path is not None and self.opts.resume:
            start_epoch = self.load_prev_checkpoint()
        
        for epoch in range(start_epoch, start_epoch + self.opts.epochs):
            self.loss_meter.reset()
            self.flow.train()
            validation_interval = max(1, len(self.train_dataloader) // 2)

            with tqdm(total=len(self.train_dataloader.dataset)) as progress_bar:
                for index, (x1, x2, y) in enumerate(self.train_dataloader):
                    self.global_steps = epoch * len(self.train_dataloader) + index

                    self.flow.train()

                    # image preprocess
                    x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                    x1, x2, y = self.process_img(x1), self.process_img(x2), self.process_img(y)

                    z_k1 = self.gen_noise(x1.size(0), sigma=0.7)
                    z_k2 = self.gen_noise(x2.size(0), sigma=0.7)
                    
                    # Generate latent z using flow model
                    z_x1, _ = self.flow(x1)
                    z_x2, _ = self.flow(x2)
                    z_y, _ = self.flow(y)

                    with torch.no_grad():
                        c1 = self.vid_generator(z_k1)
                        c2 = self.vid_generator(z_k2)

                    # Generate image x using cflow model
                    x1_v1, _ = self.cflow.inverse(z_x1, self.opts.feature_size, c1, z_std=self.opts.temperature)
                    x1_v2, _ = self.cflow.inverse(z_x1, self.opts.feature_size, c2, z_std=self.opts.temperature)
                    x2_v1, _ = self.cflow.inverse(z_x2, self.opts.feature_size, c1, z_std=self.opts.temperature)
                    y_v1, _ = self.cflow.inverse(z_y, self.opts.feature_size, c1, z_std=self.opts.temperature)
                    y_v2, _ = self.cflow.inverse(z_y, self.opts.feature_size, c2, z_std=self.opts.temperature)

                    loss, loss_dict = self.calc_loss(x1, x2, y, x1_v1, x1_v2, x2_v1, y_v1, y_v2)
                    self.loss_meter.update(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.opts.max_grad_norm > 0:
                         torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.opts.max_grad_norm)
                    self.optimizer.step()
                    loss_str = 'Loss - Avg: {:.4f} | Cur: {:.4f} | L_nll:{:.4f} | L_ano: {:.4f} | L_div: {:.4f} | L_intra: {:.4f} | L_forgery: {:.4f}\
                        '.format(self.loss_meter.avg, loss, loss_dict['loss_nll'], loss_dict['loss_anonymity'], loss_dict['loss_diversity'], loss_dict['loss_intra-id'], loss_dict['loss_id-forgery'])
                    progress_bar.set_description(loss_str)
                    progress_bar.update(x1.size(0))

                    if (index + 1) % validation_interval == 0:
                        print(f'[Epoch {epoch+1}]')
                        eer, auc, thres, val_result_dict = self.validation()
                        print(f'Normal - EER: {eer} | AUC: {auc} | threshold: {thres}')
                        with open(os.path.join(self.checkpoint_dir, 'best_log.txt'), 'a') as f:
                            f.write('Normal - EER: {:.3f} | AUC: {:.3f} | threshold: {:.2f}\n'.format(eer, auc, thres))
                        print(f'Saving models of epoch {epoch}...')
                        self.save_checkpoint(epoch, loss_dict)
                        if eer < min_eer:
                            print('Saving best model...')
                            min_eer = eer
                            self.save_checkpoint(epoch, val_result_dict, is_best=True)
                            with open(os.path.join(self.checkpoint_dir, 'best_log.txt'), 'a') as f:
                                f.write('**Best** => Epoch {} - EER: {:.3f} | AUC: {:.3f} | threshold: {:.2f}\n'.format(epoch+1, eer, auc, thres))
                            print(f'Best model updated at epoch {epoch}, EER: {eer}. AUC: {auc}')
    # Log metrics: add all the losses into tensorboard
    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_steps)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {} - step {}: '.format(prefix, self.global_steps), flush=False,end="")
        for key, value in metrics_dict.items():
            print('{} = {:.4f} |'.format(key, value), flush=False, end="")
        print()
    
    @torch.no_grad()
    def cal_normal_eer(self, images_dir, step=100):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        TP = [0 for _ in range(step)]
        FP = [0 for _ in range(step)]
        
        total = 0
        batch_num = 0

        for x1, x2, y in tqdm(self.val_dataloader):
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

            x1, x2, y = self.process_img(x1), self.process_img(x2), self.process_img(y)

            z_k1 = self.gen_noise(x1.size(0), sigma=0.7)
            z_k2 = self.gen_noise(x2.size(0), sigma=0.7)
            
            # Generate latent z using flow model
            z_x1, _ = self.flow(x1)
            z_x2, _ = self.flow(x2)
            z_y, _ = self.flow(y)
                    
            c1 = self.vid_generator(z_k1)
            c2 = self.vid_generator(z_k2)

            # Generate image x using cflow model
            x1_vir, _ = self.cflow.inverse(z_x1, self.opts.feature_size, c1, z_std=self.opts.temperature)
            x2_vir, _ = self.cflow.inverse(z_x2, self.opts.feature_size, c1, z_std=self.opts.temperature)
            y_vir, _ = self.cflow.inverse(z_y, self.opts.feature_size, c2, z_std=self.opts.temperature)

            if (batch_num + 1) % 10 == 0:
                idx = min(8, x1.shape[0])
                images = torch.cat([x1[:idx], x1_vir[:idx], x2[:idx], x2_vir[:idx], y[:idx], y_vir[:idx]], dim=0)
                # os.makedirs(os.path.join(root, 'images'), exist_ok=True)
                utils.save_image(
                    images, 
                    os.path.join(images_dir, f'tri_{batch_num+1}.png'), 
                    nrow=min(8, x1.shape[0])
                )

            feat_x1 = self.face_recognizer.extract_feats(x1_vir)
            feat_x2 = self.face_recognizer.extract_feats(x2_vir)
            feat_y = self.face_recognizer.extract_feats(y_vir)

            dis1 = cos(feat_x1, feat_x2)
            dis1[dis1 < 0] = 0
            dis2 = cos(feat_x1, feat_y)
            dis2[dis2 < 0] = 0
        
            for i in range(step):
                thres = i / step

                dis1_positive = dis1[dis1 >= thres]
                TP[i] += dis1_positive.shape[0]
                dis2_positive = dis2[dis2 >= thres]
                FP[i] += dis2_positive.shape[0]
        
            total += x1.shape[0]
            batch_num += 1
        print('Total number of tested images:', total)
        
        TPR = [tp / total for tp in TP]
        FPR = [fp / total for fp in FP]

        min_val = 999
        min_idx = 999
        for i in range(step):
            dis = abs(1 - TPR[i] - FPR[i])
            if dis < min_val:
                min_val = dis
                min_idx = i
        thres = min_idx / step
        EER = (1 - TPR[min_idx] + FPR[min_idx]) / 2
        AUC = metrics.auc(np.array(FPR), np.array(TPR))
        return EER, AUC, thres
    
    @torch.no_grad()
    def reverse_generation(self, images_dir):
        result_dict = {}

        avg_psnr_g, avg_ssim_g = [], []
        avg_psnr_r, avg_ssim_r = [], []

        trans = transforms.Resize((112, 112))

        for index, (x) in enumerate(tqdm(self.simple_dataloader)):
            # image preprocess
            x = x.to(self.device)
            x = self.process_img(x)

            feat_emb = self.vid_generator.encoder(trans(x))

            # Generate latent z using flow model
            z, _ = self.flow(x)
            # Generate image x using cflow model
            x_new, _ = self.cflow.inverse(z, self.opts.feature_size, feat_emb, z_std=self.opts.temperature)

            # Recover latent z_rec using cflow model
            z_rec, _ = self.cflow(x_new, self.opts.feature_size, feat_emb)
            # z_rec = [z_r / self.opts.temperature for z_r in z_rec]
            # Recover original images x using flow model
            x_rec, _ = self.flow.inverse(z_rec)
            
            # x = torch.clamp(x * 255, 0, 255).byte() / 255
            x_new = torch.clamp(x_new * 255, 0, 255).byte() / 255
            x_rec = torch.clamp(x_rec * 255, 0, 255).byte() / 255

            psnr_g, ssim_g = cal_psnr_ssim(x, x_new)
            psnr_r, ssim_r = cal_psnr_ssim(x, x_rec)
            
            avg_psnr_g.append(psnr_g)
            avg_ssim_g.append(ssim_g)
            avg_psnr_r.append(psnr_r)
            avg_ssim_r.append(ssim_r)

            images = torch.cat([x, x_new, x_rec], dim=0)
            utils.save_image(
                images, 
                os.path.join(images_dir, f'concat_batch_{str(index).zfill(4)}.png'),
                nrow=x.size(0),  
                pad_value=255,
            )
        
        psnr_g, ssim_g = np.mean(avg_psnr_g), np.mean(avg_ssim_g)
        psnr_r, ssim_r = np.mean(avg_psnr_r), np.mean(avg_ssim_r)
        result_dict['PSNR_G'], result_dict['SSIM_G'] = float(psnr_g), float(ssim_g)
        result_dict['PSNR_R'], result_dict['SSIM_R'] = float(psnr_r), float(ssim_r)
        
        return result_dict

    @torch.no_grad()
    def validation(self):
        self.flow.eval()

        # Complete (reverse) process image generation directory
        rev_images_dir = os.path.join(self.save_img_dir, 'reverse')
        os.makedirs(rev_images_dir, exist_ok=True)

        # Triplet images generation directory
        tri_images_dir = os.path.join(self.save_img_dir, 'triplet')
        os.makedirs(tri_images_dir, exist_ok=True)

        result_dict = self.reverse_generation(rev_images_dir)

        eer, auc, thres = self.cal_normal_eer(tri_images_dir)
        result_dict['EER'], result_dict['AUC'], result_dict['thres'] = float(eer), float(auc), float(thres)

        self.print_metrics(result_dict, prefix='validation')
        self.log_metrics(result_dict, prefix='validation') 
        return eer, auc, thres, result_dict
        

if __name__ == "__main__":
    opts = TrainOptions().parse() #get parse_args
    set_random_seed(opts.seed)
    print('local rank', opts.local_rank)
    os.makedirs(opts.exp_dir, exist_ok=True)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    train_engine = RIFTTrainer(opts)
    train_engine.train()
    
