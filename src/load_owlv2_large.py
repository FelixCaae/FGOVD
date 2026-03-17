import requests
from PIL import Image
import torch
import torch.nn.functional as F

from transformers import Owlv2Processor, Owlv2VisionModel


def load_owlv2_hf():
    processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14")
    model = Owlv2VisionModel.from_pretrained("google/owlv2-large-patch14")
    
    for p in model.parameters():
        p.requires_grad = False
    
    return model, processor

@torch.no_grad()
def forward_owlv2_hf(model, image_tensor):
    """
    image_tensor: (B,3,H,W) torch tensor, range [0,1]
    output_shape: (H_out, W_out) optional
    return: (B,C,H_p,W_p) feature map
    """

    # 1 resize
    pixel_values = F.interpolate(
        image_tensor,
        size=(1008, 1008),
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

def main():
    processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14")
    model = Owlv2Model.from_pretrained("google/owlv2-large-patch14")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = [["a photo of a cat", "a photo of a dog"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")

    with torch.no_grad():
      outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")