import torch
from models.arcface import ResNet
from models.flow import Glow
from torchvision import transforms


# Virtual Identity Generator
class VIDGenerator(torch.nn.Module):
    def __init__(self, opts, requires_grad=False):
        super(VIDGenerator, self).__init__()
        self.opts = opts
        self.device = opts.device

        # Flow generator
        self.generator = Glow(
            bits=self.opts.n_bits, 
            width=self.opts.width, 
            depth=self.opts.depth, 
            n_levels=self.opts.n_levels, 
            input_dims=(3, self.opts.image_size, self.opts.image_size), 
            lu_factorize=True
        ).to(self.device)
        self.generator.load_state_dict(torch.load(self.opts.flow_path, map_location=self.device)['net'])
        self.generator.eval()

        # Feature extractor
        self.encoder = ResNet().to(self.device)
        self.encoder.load_state_dict(torch.load(self.opts.arcface_path, map_location='cpu'))
        self.encoder.eval()
        
        self.transform = transforms.Resize((112, 112))
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, z):
        x_cond, _ = self.generator.inverse(z)
        condition = self.encoder(self.transform(x_cond))
        
        del x_cond
        return condition