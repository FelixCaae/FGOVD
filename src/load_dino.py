import torch
import torch.nn.functional as F

def load_dinov2(dino_repo ='dinov2',model_type="dinov2_vitb14_reg", checkpoint_path="./weights/dinov2_vitb14_reg4_pretrain.pth"):
    dino = torch.hub.load(dino_repo, model_type,source='local',pretrained=False).cuda()
    dino.load_state_dict(torch.load(checkpoint_path))
    from torchvision import transforms
    transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
    ])

    for param in dino.parameters():
        param.requires_grad = False
    return dino, transform

@torch.no_grad()
def forward_dino(self, data, output_shape=None, keyword='image_weak', max_length=560, patch_size=14,):
    #data is an image tensor [B,C,H,W] 
    transforms = self.VFM_transform
    model_dino = self.VFM_backbone
    import torch.nn.functional as F
    output = []
    #compute the target size accoding to the original aspect ratio
    b,c,h,w = data.shape
    x = data
    if h <= w:
        s_w = max_length
        s_h = s_w * (h/w)
    else:
        s_h = max_length
        s_w = s_h * (w/h)
    #adjust slightly to fit the patch size
    s_h = (s_h // patch_size) * patch_size
    s_w = (s_w // patch_size) * patch_size
    s_h, s_w = int(s_h), int(s_w)

    #resize image to the target size
    x = F.interpolate(x, (s_h, s_w), mode='bilinear')

    #forward
    feat_dict = model_dino.forward_features(x)
    
    #reshape to B,C,H,W
    feat = feat_dict['x_norm_patchtokens'].reshape(b, s_h//patch_size, s_w//patch_size,-1).permute(0,3,1,2)
    if output_shape is not None:
        #reisze to outshape if given
        feat = F.interpolate(feat, output_shape, mode='bilinear')   
    return feat

from transformers import AutoImageProcessor, AutoModel
import torch

def load_dinov2_hf():
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').eval().cuda()
    
    for p in model.parameters():
        p.requires_grad = False
    
    return model, processor

@torch.no_grad()
def forward_dinov2_hf(model, image_tensor):
    """
    image_tensor: (B,3,H,W) torch tensor, range [0,1]
    output_shape: (H_out, W_out) optional
    return: (B,C,H_p,W_p) feature map
    """

    # 1 resize
    pixel_values = F.interpolate(
        image_tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    # 3 forward
    outputs = model(pixel_values=pixel_values)

    # (B, 1 + N, C)
    token_feats = outputs.last_hidden_state

    B, L, C = token_feats.shape

    # 4 remove CLS token
    patch_feats = token_feats[:, 1:, :]

    # 5 compute patch grid
    num_patches = patch_feats.shape[1]
    H_p = W_p = int(num_patches ** 0.5)

    # reshape -> (B,C,H_p,W_p)
    feat_map = patch_feats.reshape(B, H_p, W_p, C).permute(0,3,1,2).contiguous()

    return feat_map