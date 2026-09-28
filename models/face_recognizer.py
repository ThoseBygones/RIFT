import torch
from torchvision import transforms

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


# a class for calculating id loss (cosine similarity)
class FaceRecognizer(torch.nn.Module):
    def __init__(
        self,
        opts,
        backbone,
        input_size=112,
        standardize_input=False,
        requires_grad=False,
    ):
        self.opts = opts
        super(FaceRecognizer, self).__init__()
        self.backbone = backbone
        self.backbone.eval()
        self.transform = transforms.Resize((input_size, input_size))
        self.standardize_input = standardize_input
        
        self.margin = self.opts.cos_margin if self.opts.cos_margin is not None else 0.1 
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def extract_feats(self, X):
        h = self.transform(X)
        if self.standardize_input:
            # facenet-pytorch expects the fixed-image-standardization range.
            h = (h * 255.0 - 127.5) / 128.0
        h = self.backbone(h)
        h = l2_norm(h)
        return h
    
    def calculate_loss(self, features1, features2):
        assert features1.shape == features2.shape

        rev_id_loss = torch.cosine_similarity(features1, features2).clamp(min=self.margin).mean()
        id_loss = (1 - torch.cosine_similarity(features1, features2).clamp(min=self.margin)).mean()

        return id_loss, rev_id_loss


    def forward(self, img1, img2):
        features1 = self.extract_feats(img1)
        features2 = self.extract_feats(img2)
        
        return self.calculate_loss(features1, features2)
