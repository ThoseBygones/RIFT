from dataset.simple_dataset import SimpleDataset
from models.arcface import ResNet
from models.cflow import CFlow as CFlowWithoutAttention
from models.cflow_attention import CFlow as AttentionCFlow
from models.flow import Glow
from options.train_options import TrainOptions
from skimage.metrics import structural_similarity as SSIM, peak_signal_noise_ratio as PSNR
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


class CFlowTrainer:
    def __init__(self, opts):
        self.opts = opts
        self.global_steps = 0
        require_paths(
            data_path=self.opts.data_path,
            arcface_path=self.opts.arcface_path,
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

        # Initialize networks
        self.feature_extractor = ResNet().to(self.device)
        self.feature_extractor.load_state_dict(torch.load(self.opts.arcface_path, map_location='cpu'))
        self.feature_extractor.eval()

        self.flow = Glow(
            bits=self.opts.n_bits, 
            width=self.opts.width, 
            depth=self.opts.depth, 
            n_levels=self.opts.n_levels, 
            input_dims=(3, self.opts.image_size, self.opts.image_size), 
            lu_factorize=True
        ).to(self.device)
        self.flow.load_state_dict(torch.load(self.opts.flow_path, map_location=self.device)['net'])
        self.flow.eval()
        
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

        if self.opts.ddp:
            self.cflow = nn.parallel.DistributedDataParallel(self.cflow, device_ids=[self.opts.local_rank], output_device=self.opts.local_rank, broadcast_buffers=False, find_unused_parameters=True)
            
        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.val_dataset = self.configure_datasets()

        # Sampler
        train_sampler, val_sampler = None, None
        if self.opts.ddp:
            train_sampler = dist.DistributedSampler(self.train_dataset)
            val_sampler = dist.DistributedSampler(self.val_dataset)
        
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
        params = [{'params': self.cflow.parameters()}]
        optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        return optimizer
        
    def configure_datasets(self):
        # Load train, validation and test images
        trans = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize((self.opts.image_size, self.opts.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        train_dataset = SimpleDataset(
            self.opts.data_path,
            transform=trans
        )
        val_dataset = SimpleDataset(
            self.opts.data_path, 
            transform=trans, 
            randomized=True
        )
        print("Number of training samples: {}".format(len(train_dataset)), flush=False)
        print("Number of validation samples: {}".format(len(val_dataset)), flush=False)
        return train_dataset, val_dataset
    
    def load_prev_checkpoint(self):
        print(f'Loading previously trained model from path: {self.opts.resume_ckpt_path}...')
        
        prev_checkpoint = torch.load(self.opts.resume_ckpt_path, map_location='cpu')
        self.cflow.load_state_dict(prev_checkpoint['net'])

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
            'net': self.cflow.state_dict(),
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
    
    def calc_cflow_loss(self, x, n_y, y, bits_per_pixel=False):
        zs, logdet = self.cflow(x, n_y, y)
        base_dist_mean, base_dist_var = torch.zeros(1).to(self.device), torch.ones(1).to(self.device)
        base_dist = D.Normal(base_dist_mean, base_dist_var)
        log_prob = sum(base_dist.log_prob(z).sum([1,2,3]) for z in zs) + logdet
        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel())
        return zs, -log_prob
    
    def calc_loss(self, x, condition):
        loss_dict = {}
        loss = 0.0

        # NLL loss of cflow
        zs, loss = self.calc_cflow_loss(x, self.opts.feature_size, condition, bits_per_pixel=True)
        loss = loss.mean()
        # print('loss shape:', loss.shape)
        
        loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def train(self):
        start_epoch = 0
        best_loss = 0

        trans = transforms.Resize((112, 112))

        if self.opts.resume_ckpt_path is not None and self.opts.resume:
            start_epoch = self.load_prev_checkpoint()
        
        for epoch in range(start_epoch, start_epoch + self.opts.epochs):
            self.loss_meter.reset()
            self.cflow.train()
            validation_interval = max(1, len(self.train_dataloader) // 5)

            with tqdm(total=len(self.train_dataloader.dataset)) as progress_bar:
                for index, (x) in enumerate(self.train_dataloader):
                    self.global_steps = epoch * len(self.train_dataloader) + index

                    self.cflow.train()

                    # image preprocess
                    x = x.to(self.device)
                    x = self.process_img(x)

                    with torch.no_grad():
                        feat_emb = self.feature_extractor(trans(x))

                    loss, loss_dict = self.calc_loss(x, feat_emb)
                    self.loss_meter.update(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.opts.max_grad_norm > 0:
                         torch.nn.utils.clip_grad_norm_(self.cflow.parameters(), self.opts.max_grad_norm)
                    self.optimizer.step()
                    loss_str = 'Loss - Avg: {:.4f} | Cur: {:.4f}'.format(self.loss_meter.avg, loss.item())
                    progress_bar.set_description(loss_str)
                    progress_bar.update(x.size(0))

                    if (index + 1) % validation_interval == 0:
                        self.print_metrics(loss_dict, prefix='training')
                        self.log_metrics(loss_dict, prefix='training')    
                        self.validation(epoch)
            
            val_result_dict = self.validation(epoch, united_gen=True)
            print(f'Saving models of epoch {epoch}...')
            self.save_checkpoint(epoch, loss_dict)
            
            # Save checkpoint
            if epoch == 0  or self.loss_meter.avg < best_loss:
                print('Saving best models...')
                self.save_checkpoint(epoch, val_result_dict, is_best=True)
                print(f'Model updated at epoch {epoch}...')
                best_loss = self.loss_meter.avg

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
    def random_generation(self, images_dir, epoch):
        samples = []

        trans = transforms.Resize((112, 112))

        # c = torch.rand(self.opts.n_samples, self.opts.feature_size).to(self.device)
        # c = c * 2.0 - torch.ones_like(c)
        z_c = self.gen_noise(self.opts.n_samples, sigma=0.7)
        x_c, _ = self.flow.inverse(z_c)
        c = self.feature_extractor(trans(x_c))

        for z_std in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            z = self.gen_noise(self.opts.n_samples, z_std)
            x, _ = self.cflow.inverse(z, self.opts.feature_size, c)
            samples.append(x)
            x = torch.clamp(x * 255, 0, 255).byte() / 255 

        images = torch.cat(samples, 0)
        utils.save_image(
            images, 
            os.path.join(images_dir, f'samples_epoch_{epoch}.png'),
            nrow=self.opts.n_samples, 
            pad_value=255,
        )
    
    @torch.no_grad()
    def united_generation(self, images_dir):
        result_dict = {}

        avg_psnr_g, avg_ssim_g = [], []
        avg_psnr_r, avg_ssim_r = [], []

        trans = transforms.Resize((112, 112))

        for index, (x) in enumerate(tqdm(self.val_dataloader)):
            # image preprocess
            x = x.to(self.device)
            x = self.process_img(x)

            feat_emb = self.feature_extractor(trans(x))

            # Generate latent z using flow model
            z, _ = self.flow(x)
            # Generate image x using cflow model
            x_new, _ = self.cflow.inverse(z, self.opts.feature_size, feat_emb)

            # Recover latent z_rec using cflow model
            z_rec, _ = self.cflow(x_new, self.opts.feature_size, feat_emb)
            # z_rec = [z_r / 0.7 for z_r in z_rec]
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
        result_dict['PSNR_G'], result_dict['SSIM_G'] = psnr_g, ssim_g
        result_dict['PSNR_R'], result_dict['SSIM_R'] = psnr_r, ssim_r
        
        return result_dict

    @torch.no_grad()
    def validation(self, epoch, united_gen=False):
        self.cflow.eval()

        rand_images_dir = os.path.join(self.save_img_dir, 'random')
        os.makedirs(rand_images_dir, exist_ok=True)

        concat_images_dir = os.path.join(self.save_img_dir, 'concat')
        os.makedirs(concat_images_dir, exist_ok=True)

        # Testing image generation
        self.random_generation(rand_images_dir, epoch) # cflow

        if united_gen:
            result_dict = self.united_generation(concat_images_dir)
            self.print_metrics(result_dict, prefix='validation')
            self.log_metrics(result_dict, prefix='validation') 
            return result_dict
        

if __name__ == "__main__":
    opts = TrainOptions().parse() #get parse_args
    set_random_seed(opts.seed)
    print('local rank', opts.local_rank)
    os.makedirs(opts.exp_dir, exist_ok=True)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    train_engine = CFlowTrainer(opts)
    train_engine.train()
    
