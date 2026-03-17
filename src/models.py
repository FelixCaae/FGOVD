import os
import numpy as np
import torch
import torch.nn as nn
import transformers

from copy import deepcopy
from PIL import Image
from torchvision.ops import batched_nms
from transformers import AutoProcessor, OwlViTForObjectDetection, Owlv2ForObjectDetection
from transformers.image_transforms import center_to_corners_format
from src.util import get_processor


# Monkey patched for no in-place ops
class PatchedOwlViTClassPredictionHead(nn.Module):
    def __init__(self, original_cls_head):
        super().__init__()

        self.query_dim = original_cls_head.query_dim

        self.dense0 = original_cls_head.dense0
        # self.dense0 = nn.Linear(768, 512) # random initialization

    def forward(self, image_embeds, query_embeds):
        image_class_embeds = self.dense0(image_embeds)

        # Normalize image and text features
        image_class_embeds = image_class_embeds / (
            torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
        )
        query_embeds = (
            query_embeds / torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
        )

        pred_sims = image_class_embeds @ query_embeds.transpose(1, 2)

        return None, pred_sims


from .clip_visual_extractor import CLIPVisualExtractor
class OwlViT(torch.nn.Module):
    """
    We don't train this that's why it's not an nn.Module subclass.
    We just use this to get to the point where we can use the
    classifier to filter noise.
    """

    def __init__(self, pretrained_model, query_bank=None, processor=None):
        super().__init__()

        # Take the pretrained components that are useful to us
        if isinstance(pretrained_model, OwlViTForObjectDetection):
            self.backbone = pretrained_model.owlvit.vision_model
            self.text_model = pretrained_model.owlvit.text_model
            self.text_projection = pretrained_model.owlvit.text_projection
        elif isinstance(pretrained_model, Owlv2ForObjectDetection):
            self.backbone = pretrained_model.owlv2.vision_model
            self.text_model = pretrained_model.owlv2.text_model
            self.text_projection = pretrained_model.owlv2.text_projection
        else:
            raise ValueError("The loaded pretrained model is invalid")
        self.post_post_layernorm = pretrained_model.layer_norm
        self.class_predictor = PatchedOwlViTClassPredictionHead(
            pretrained_model.class_head
        )
        # with the aim of calculating a loss which helps preserving the original coarse-grained capabilities of the model, we attach the original class_predictor which will remain freezed for the whole training
        self.original_class_predictor = deepcopy(self.class_predictor)
        # PatchedOwlViTClassPredictionHead(
        #     pretrained_model.class_head
        # )
        self.box_head = pretrained_model.box_head
        self.compute_box_bias = pretrained_model.compute_box_bias
        self.box_bias = pretrained_model.compute_box_bias(pretrained_model.sqrt_num_patches)
        self.sigmoid = pretrained_model.sigmoid

        if query_bank is not None:
            self.queries = torch.nn.Parameter(query_bank)
        else:
            # if we have not precomputed the text embeddings, we attach the original model in order to exploit the text pipeline
            # Note: we attach all the model and not only the textual pipeline because the forward function of OwlViTForObjectDetection do not account the possibility of disentangle text and visual forwarding
            # self.pretrained_model = pretrained_model.eval()
            self.processor = processor
            self.dummy_image = Image.new("RGB", (224, 224))
            self.init_distill()
        
    def init_distill(self):
        self.clip = CLIPVisualExtractor('ViT-B/32', 'cuda')
        self.distill_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Linear(768, 768),
        )

    # Copied from transformers.models.clip.modeling_owlvit.OwlViTForObjectDetection.box_predictor
    # Removed some comments and docstring to clear up clutter for now
    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
        feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pred_boxes = self.box_head(image_feats)
        if transformers.__version__ == '4.30.2':
            pred_boxes += self.compute_box_bias(feature_map) 
        else:
            pred_boxes += self.box_bias.to(pred_boxes.device) # self.compute_box_bias(self.pretrained_model.sqrt_num_patches).to(pred_boxes.device)
        pred_boxes = self.sigmoid(pred_boxes)
        return center_to_corners_format(pred_boxes)

    # Copied from transformers.models.clip.modeling_owlvit.OwlViTForObjectDetection.image_embedder
    # Removed some comments and docstring to clear up clutter for now
    def image_embedder(self, pixel_values):
        vision_outputs = self.backbone(pixel_values=pixel_values)
        last_hidden_state = vision_outputs.last_hidden_state
        image_embeds = self.backbone.post_layernorm(last_hidden_state)

        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.post_post_layernorm(image_embeds)

        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        return image_embeds
    
    def text_embedder(self, inputs):
        text_outputs = self.text_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        text_embeds = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)
        return text_embeds

    def get_distill_loss(self, image, feature_map):
        spatial_tokens = self.clip.to(image.device).get_spatial_feats(image, image.dtype, return_cls_token=False)
        _feature_map = feature_map.permute(0, 3, 1, 2).contiguous()
        _feature_map = self.clip.to(image.device).align_to_sizes(_feature_map, [(spatial_tokens.shape[2], spatial_tokens.shape[3])])[0]
        loss_distill = 1 - (_feature_map * spatial_tokens / (_feature_map.norm(2, dim=1, keepdim=True) * spatial_tokens.norm(2, dim=1, keepdim=True) + 1e-6)).sum(dim=1)
        loss_distill = loss_distill.mean(dim=[-1, -2])
        return loss_distill

    def forward(
        self,
        image: torch.Tensor,
        texts: torch.Tensor = None,
        queries: torch.Tensor = None,
        lvis_query_embeds: torch.Tensor = None
    ):
        assert texts is not None or queries is not None, "At least a vocabulary or a set of embeddings should be provided!"
        # Same naming convention as image_guided_detection
        feature_map = self.image_embedder(image)
        new_size = (
            feature_map.shape[0],
            feature_map.shape[1] * feature_map.shape[2],
            feature_map.shape[3],
        )
        feature_map = feature_map + self.distill_proj(feature_map)

        # spatial_tokens = self.clip.to(image.device).get_spatial_feats(image, image.dtype, return_cls_token=False)
        # _feature_map = self.distill_proj(feature_map).permute(0, 3, 1, 2).contiguous()
        # _feature_map = self.clip.to(image.device).align_to_sizes(_feature_map, [(spatial_tokens.shape[2], spatial_tokens.shape[3])])[0]
        # loss_distill = 1 - (_feature_map * spatial_tokens / (_feature_map.norm(2, dim=1, keepdim=True) * spatial_tokens.norm(2, dim=1, keepdim=True) + 1e-6)).sum(dim=1)
        # loss_distill = loss_distill.sum(dim=[-1, -2])
        if self.training:
            loss_distill = self.get_distill_loss(image, feature_map)

        image_feats = torch.reshape(feature_map, new_size)

        # Box predictions
        pred_boxes = self.box_predictor(image_feats, feature_map)

        # if we have not precomputed text embeddings, we calculate the embedding of the vocabulary step by step
        if texts is not None:
            with torch.no_grad():
                queries = self.text_embedder(texts)
                queries = queries.view(image.shape[0], queries.shape[0] // image.shape[0], -1) # BS X N_CAPT X EMBED_DIM
                # queries = self.pretrained_model(**texts).text_embeds
        pred_class_logits, pred_class_sims = self.class_predictor(
            image_feats, queries
        )
        # if we have the LVIS text embeddings, we calculate the original LVIS scores
        if lvis_query_embeds is not None:
            _, lvis_pred_class_sims = self.class_predictor(
                image_feats, lvis_query_embeds
            )
            _, lvis_target_class_sims = self.original_class_predictor(
                image_feats, lvis_query_embeds
            )
            if self.training:
                return (pred_boxes, pred_class_logits, pred_class_sims, None, lvis_pred_class_sims, lvis_target_class_sims, loss_distill)
            else:
                return (pred_boxes, pred_class_logits, pred_class_sims, None, lvis_pred_class_sims, lvis_target_class_sims)
        if self.training:
            return (pred_boxes, pred_class_logits, pred_class_sims, None, loss_distill)
        else:
            return (pred_boxes, pred_class_logits, pred_class_sims, None)

import torch.nn.functional as F
from .load_dino import load_dinov2_hf as load_dinov2, forward_dinov2_hf as forward_dino
class OwlViT_DINO(OwlViT):
    def init_distill(self):
        dino, transform = load_dinov2()
        self.dino = dino
        self.transform = transform
        self.distill_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Linear(768, 768),
        )
    
    def get_distill_loss(self, image, feature_map):
        dino_feat = forward_dino(self.dino, image) # [B, C, H, W]
        student_feat = student_feat.permute(0,3,1,2).contiguous()
        student_feat = F.interpolate(
            student_feat,
            size=dino_feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        import pdb; pdb.set_trace()
        student_feat = F.normalize(student_feat, dim=1)
        dino_feat = F.normalize(dino_feat, dim=1)
        loss = 1 - (student_feat * dino_feat).sum(dim=1)
        loss = loss.mean(dim=[-1, -2])
        return loss

from .load_owlv2_large import load_owlv2_hf as load_owlv2, forward_owlv2_hf as forward_owlv2
class OwlViT_Large(OwlViT):
    def init_distill(self):
        model, transform = load_owlv2()
        self.teacher = model
        self.transform = transform
        self.distill_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Linear(768, 768),
        )
        self.distill_proj_align = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )

    def get_distill_loss(self, image, feature_map):
        teacher_feat = forward_owlv2(self.teacher, image) # [B, C, H, W]
        student_feat = self.distill_proj_align(feature_map).permute(0,3,1,2).contiguous()
        student_feat = F.interpolate(
            student_feat,
            size=teacher_feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        student_feat = F.normalize(student_feat, dim=1)
        teacher_feat = F.normalize(teacher_feat, dim=1)
        loss = 1 - (student_feat * teacher_feat).sum(dim=1)
        loss = loss.mean(dim=[-1, -2])
        return loss


class PostProcess:
    def __init__(self, confidence_threshold=0.75, iou_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, all_pred_boxes, pred_classes):
        # Just support batch size of one for now
        pred_boxes = all_pred_boxes.squeeze(0)
        pred_classes = pred_classes.squeeze(0)

        top = torch.max(pred_classes, dim=1)
        scores = top.values
        classes = top.indices

        idx = scores > self.confidence_threshold
        scores = scores[idx]
        classes = classes[idx]
        pred_boxes = pred_boxes[idx]

        idx = batched_nms(pred_boxes, scores, classes, iou_threshold=self.iou_threshold)
        classes = classes[idx]
        pred_boxes = pred_boxes[idx]
        scores = scores[idx]

        return pred_boxes.unsqueeze_(0), classes.unsqueeze_(0), scores.unsqueeze_(0)


def load_model(device, base_config="google/owlvit-base-patch16"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if 'owlvit' in base_config:
        _model = OwlViTForObjectDetection.from_pretrained(base_config)
    elif 'owlv2' in base_config:
        _model = Owlv2ForObjectDetection.from_pretrained(base_config)
    else:
        raise ValueError("The starting configuration must come from from owlvit or owlv2")
    _processor = get_processor(base_config)
    _model = _model.eval()
    # patched_model = OwlViT_Large(pretrained_model=_model, processor=_processor)
    # patched_model = OwlViT_DINO(pretrained_model=_model, processor=_processor)
    patched_model = OwlViT(pretrained_model=_model, processor=_processor)
        
    for name, parameter in patched_model.named_parameters():
        conditions = [
            "class_predictor" in name and 'original' not in name,
            'distill_proj' in name
        ]
        if any(conditions):
            continue
        parameter.requires_grad = False

    print("Trainable parameters:")
    for name, parameter in patched_model.named_parameters():
        if parameter.requires_grad:
            print(f"  {name}")
    print()
    return patched_model.to(device)