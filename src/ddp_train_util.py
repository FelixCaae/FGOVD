import json
import itertools
import os
import numpy as np
import torch
import torch.distributed as dist

from eval.fgovd.evaluate_map import evaluate_map
from lvis import LVISEval
from PIL import Image
from src.tensorboard_util import log_train, log_validation, log_lvis
from src.util import BoxUtil
try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    # torchmetrics==0.6.0
    from torchmetrics.detection import MAP as MeanAveragePrecision
from tqdm import tqdm


def coco_to_model_input(boxes, metadata):
    """
    absolute xywh -> relative xyxy
    """
    boxes = BoxUtil.box_convert(boxes, "xywh", "xyxy")
    boxes = BoxUtil.scale_bounding_box(
        boxes, metadata["width"], metadata["height"], mode="down"
    )

    return boxes

def prepare_texts_inputs(model, vocabularies):
    to_encode = [[] for _ in range(len(vocabularies[0][0]))]
    expected_len = len(vocabularies[0][0])
    for capts_batch in vocabularies[0]:
        for j, capt in enumerate(capts_batch):
            if j >= expected_len:
                break  # 防止越界
            to_encode[j].append(capt)
    #assume this model is a ddp model
    inputs = model.module.processor(
        text=to_encode,
        images=[model.module.dummy_image] * len(to_encode),
        return_tensors="pt"
    )
    return inputs


def train(model,
          train_dataloader,
          criterion,
          optimizer,
          general_loss,
          epoch,
          n_accumulation_steps=1,
          lvis_query_embeds=None,
          writer=None,
          amp=False):
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()  

    losses = []
    device = next(model.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # 只在主进程显示进度条
    if rank == 0:
        dataloader_iter = tqdm(train_dataloader, ncols=60)
    else:
        dataloader_iter = train_dataloader
    
    for i, (images, labels, boxes, metadata) in enumerate(dataloader_iter):
        # Prep inputs
        images = images.to(device)
        labels = labels.to(device) * 0  # correct label is always the first of the vocabulary
        
        vocabularies = metadata['vocabularies']
        boxes = coco_to_model_input(boxes, metadata).to(device)
        inputs = prepare_texts_inputs(model, vocabularies)
        inputs = inputs.to(device)
        
        # if lvis_query_embeds is None:
        #     # self-distillation disabled
        #     all_pred_boxes, _, pred_sims, _ = model(images, inputs) 
        #     lvis_pred_sims = lvis_target_sims = None
        # else:
        #     # self-distillation enabled
        #     all_pred_boxes, _, pred_sims, _, lvis_pred_sims, lvis_target_sims = model(images, inputs, lvis_query_embeds=lvis_query_embeds) 
            

        if amp:
            with autocast():
                if lvis_query_embeds is None:
                # self-distillation disabled
                    all_pred_boxes, _, pred_sims, _ = model(images, inputs) 
                    lvis_pred_sims = lvis_target_sims = None
                else:
                # self-distillation enabled
                    all_pred_boxes, _, pred_sims, _, lvis_pred_sims, lvis_target_sims = model(images, inputs, lvis_query_embeds=lvis_query_embeds) 
                losses = criterion(pred_sims, labels, all_pred_boxes, boxes, lvis_pred_scores=lvis_pred_sims, lvis_target_scores=lvis_target_sims)
                loss = (
                    losses["loss_triplet"]
                    + losses["loss_bg"]
                    + losses["loss_rank"]
                ) 
                loss = loss / n_accumulation_steps
            scaler.scale(loss).backward()
        else:
            if lvis_query_embeds is None:
            # self-distillation disabled
                all_pred_boxes, _, pred_sims, _ = model(images, inputs) 
                lvis_pred_sims = lvis_target_sims = None
            else:
            # self-distillation enabled
                all_pred_boxes, _, pred_sims, _, lvis_pred_sims, lvis_target_sims = model(images, inputs, lvis_query_embeds=lvis_query_embeds) 
            losses = criterion(pred_sims, labels, all_pred_boxes, boxes, lvis_pred_scores=lvis_pred_sims, lvis_target_scores=lvis_target_sims)
            loss = (
                losses["loss_triplet"]
                + losses["loss_bg"]
                + losses["loss_rank"]
            ) 
            loss = loss / n_accumulation_steps
            loss.backward()
        # 只在主进程记录TensorBoard
        if rank == 0 and writer is not None:
            log_train(writer, losses, epoch * len(train_dataloader) + i)
        
        # 梯度累积

        
        if ((i + 1) % n_accumulation_steps == 0) or ((i + 1) == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()
        
        # 只在主进程更新损失记录
        if rank == 0 and general_loss is not None:
            general_loss.update(losses)
        
        # 同步所有进程
        if dist.is_initialized():
            dist.barrier()
    
    # 只在主进程返回训练指标
    if rank == 0 and general_loss is not None:
        train_metrics = general_loss.get_values()
        general_loss.reset()
        return train_metrics
    else:
        return None

def apply_NMS(preds, iou=0.5):
    boxes = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']
    
    indexes_to_keep = batched_nms(boxes, 
                                  scores, 
                                  labels,
                                  iou)
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    
    for x in range(len(boxes)):
        if x in indexes_to_keep:
            filtered_boxes.append(boxes[x])
            filtered_scores.append(scores[x])
            filtered_labels.append(labels[x])
    
    preds['boxes'] = torch.stack(filtered_boxes, dim=0)
    preds['scores'] = torch.stack(filtered_scores, dim=0)
    preds['labels'] = torch.stack(filtered_labels, dim=0)
    return preds

def mAP_update_metric(metric, preds_per_image, targets, device=None, disable_nms=False, n_neg=5, get_tensors=False):
    import os
    n_neg = n_neg
    metric.update(preds_per_image, targets)
    

def get_image_ground_truth(data, image_id, device):
    """
    Given a dictionary 'data' and an 'image_id', returns a dictionary with 'boxes' and 'categories' information for
    that image.

    Args:
        data (dict): The data dictionary containing 'annotations'.
        image_id (int): The image_id for which to retrieve data.

    Returns:
        dict: A dictionary with 'boxes' and 'categories' information for the given image_id.
    """
    image_data = {'boxes': [], 'labels': []}  # Initialize the dictionary to store image data
    def convert_format(boxes):
        for box in boxes:
            box[2] += box[0]
            box[3] += box[1]
        return boxes

    def assert_box(boxes):
        """Check that the box is in [xmin, ymin, xmax, ymax] format"""
        for box in boxes:
            assert box[0] <= box[2] and box[1] <= box[3]

    # Loop through each annotation in the 'annotations' list
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            # If the 'image_id' in the annotation matches the given 'image_id', append bbox and category_id to the lists
            image_data['boxes'].append(annotation['bbox'])
            image_data['labels'].append(annotation['category_id'])

    image_data['boxes'] = convert_format(image_data['boxes'])
    assert_box(image_data['boxes'])
    # tensorize elements
    image_data['boxes'] = torch.Tensor(image_data['boxes']).to(device)
    image_data['labels'] = torch.IntTensor(image_data['labels']).to(device)
    
    return image_data
def validate(model, val_dataloader, val_data, epoch, writer, n_hardnegatives=10, lvis_evaluation=False):
    from torchvision.ops import batched_nms
    device = next(model.parameters()).device
    metric = MeanAveragePrecision().to(device)
    val_data['annotations'] = [ann for ann in val_data['annotations'] if len(ann['neg_category_ids']) >= n_hardnegatives]

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    with torch.no_grad():
        for i, (image, labels, boxes, metadata) in enumerate(
            tqdm(val_dataloader, ncols=60)
        ):
            # Prep inputs
            image = image.to(device) #B,C,H,W
            labels = labels.to(device)
            
            bs = image.shape[0]
            vocabularies = metadata['vocabularies'] if not lvis_evaluation else [[tuple([cat['name']] * bs) for cat in val_data['categories']]]
            
            
            boxes = coco_to_model_input(boxes, metadata).to(device)
            n_hardnegatives = len(vocabularies[0]) - 1

            # Get predictions and save output 
            inputs = prepare_texts_inputs(model, vocabularies)
            inputs = inputs.to(device)
            pred_boxes, _, pred_class_sims, _ = model(image, inputs)
            
            n_preds = pred_boxes.shape[1]
            scores_filtered = []
            labels_filtered = []
            boxes_filtered = []
            height = metadata['height']
            width = metadata['width']
            
            logits = torch.max(pred_class_sims, dim=-1)
            scores = torch.sigmoid(logits.values)
            pred_labels = (labels + logits.indices) if not lvis_evaluation else logits.indices
            
            pred_boxes[:, :, [0, 2]] = pred_boxes[:, :, [0, 2]] * width.expand(n_preds, bs).expand(2, -1, -1).transpose(0, 2).cuda()
            pred_boxes[:, :, [1, 3]] = pred_boxes[:, :, [1, 3]] * height.expand(n_preds, bs).expand(2, -1, -1).transpose(0, 2).cuda()
            
            # ordering the predictions and keeping only the best 200
            max_preds = 200
            sorted_indices = torch.argsort(scores, dim=1, descending=True)
            scores_filtered = torch.gather(scores, 1, sorted_indices)[:max_preds]
            labels_filtered = torch.gather(pred_labels, 1, sorted_indices)[:max_preds]
            boxes_filtered = pred_boxes.gather(1, sorted_indices.unsqueeze(2).expand(-1, -1, 4))[:max_preds]
             
            
            # iterating over all the images in the batch
            preds = []
            for index, (s, b, l) in enumerate(zip(scores_filtered, boxes_filtered, labels_filtered)):
                keep_ind = batched_nms(b, 
                                  s, 
                                  l,
                                  0.5)
                preds.append({
                    'scores': s[keep_ind].contiguous(),
                    'labels': l[keep_ind].contiguous(),
                    'boxes': b[keep_ind].contiguous(),
                    'image_filepath': '/'.join(metadata['impath'][index].split('/')[-2:])
                })
            targest =[get_image_ground_truth(val_data, int(path.split('/')[-1].split('.')[0]), device) for path in metadata['impath']]
            mAP_update_metric(metric, preds, targest, n_neg=n_hardnegatives,  )
    # print(rank, world_size, dist.is_initialized())
    val_metrics = metric.compute()
    return val_metrics
def get_ids_per_frequencies(lvis_data):
    freqs = {}
    for cat in lvis_data['categories']:
        freqs[cat['frequency']] = freqs.get(cat['frequency'], []) + [cat['id']]
    return freqs

def get_map_per_frequencies(lvis_metrics, ids_per_frequencies):
    for freq in ids_per_frequencies.keys():
        cat_mask = torch.logical_and(torch.isin(lvis_metrics['classes'], torch.tensor(ids_per_frequencies[freq])), lvis_metrics['map_per_class'] >= 0)
        lvis_metrics[f'map_{freq}'] = torch.mean(lvis_metrics['map_per_class'][cat_mask])
    
    return lvis_metrics

def validate_lvis(model, lvis_dataloader, queries, postprocess, ids_per_frequencies, epoch, writer=None):
    device = next(model.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # 只在主进程初始化指标
    if rank == 0:
        metric = MeanAveragePrecision(class_metrics=True).to('cpu')
        maps = []
    else:
        metric = None
        maps = None
    
    with torch.no_grad():
        # 只在主进程显示进度条
        if rank == 0:
            dataloader_iter = tqdm(lvis_dataloader, ncols=60)
        else:
            dataloader_iter = lvis_dataloader
        
        for i, (image, labels, boxes, metadata) in enumerate(dataloader_iter):
            # Prep inputs
            image = image.to(device)
            labels = labels.to(device)
            boxes = coco_to_model_input(boxes, metadata).to(device)

            # Get predictions and save output
            pred_boxes, pred_classes, pred_class_sims, _ = model(image, queries=queries)
            pred_boxes, pred_classes, scores = postprocess(pred_boxes, pred_class_sims)
            pred_classes += 1
        
            # Use only the top 300 boxes to stay consistent with benchmarking
            top = torch.topk(scores, min(300, scores.size(-1)))
            scores = top.values
            inds = top.indices.squeeze(0)
            
            # 只在主进程更新指标
            if rank == 0:
                update_metrics(metric, metadata, pred_boxes[:, inds], pred_classes[:, inds], scores, boxes, labels)
                
                # 计算每个批次的mAP
                oneshot_metric = MeanAveragePrecision().to('cpu')
                update_metrics(oneshot_metric, metadata, pred_boxes[:, inds], pred_classes[:, inds], scores, boxes, labels)
                oneshot_metrics = oneshot_metric.compute()
                maps.append(oneshot_metrics['map'].item())
    
    # 同步所有进程
    if dist.is_initialized():
        dist.barrier()
    
    # 只在主进程计算最终指标
    if rank == 0:
        lvis_metrics = metric.compute()
        lvis_metrics = get_map_per_frequencies(lvis_metrics, ids_per_frequencies)
        if writer is not None:
            log_lvis(writer, epoch, lvis_metrics=lvis_metrics, map_per_image=sum(maps) / len(maps) if maps else 0)
        return lvis_metrics
    else:
        return None
    
# adapted from an answer in https://github.com/huggingface/transformers/issues/21206
def validate_lvis_hf(model, processor, data_cfg, train_cfg, lvis_data, text_query, tmp_base_path, epoch, writer=None):
    batch_size = train_cfg['batch_size']
    dataset_path = data_cfg['images_path']
    device = next(model.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # 只在主进程执行评估
    if rank != 0:
        return None
    
    images = [dataset_path + '/' + '/'.join(img['coco_url'].split('/')[-2:]) for img in lvis_data['images']]
    instances = []
    n = len(images)
    n_batches = n // batch_size + 1 if n % batch_size != 0 else n // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            image_ids = []
            batch_images = []
            target_sizes = []
            for img_path in images[i * batch_size: (i+1) * batch_size]:
                image_ids.append(int(img_path.split("/")[-1].split(".")[0]))
                image = Image.open(img_path).convert("RGB")
                batch_images.append(image)
                target_sizes.append((max(image.size), max(image.size)))
            target_sizes = torch.Tensor(target_sizes)
            target_sizes = target_sizes.to(device)
            texts = [text_query] * len(batch_images)
            inputs = processor(text=texts, images=batch_images, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs)
            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            # Convert outputs (bounding boxes and class logits) to COCO API
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.)
            # results = processor.post_process_object_detection_evaluation(outputs=outputs, target_sizes=target_sizes, pred_per_im=300)
            for image_id, res in zip(image_ids, results):
                # keep first 300 to stay consistent with LVIS evaluation
                _, sorted_indices = torch.sort(res['scores'], descending=True)
                # Sort each field in the dictionary using the sorted indices
                max_preds = 300
                res = {key: value[sorted_indices][:max_preds] for key, value in res.items()}
                for bbox, score, label in zip(res["boxes"], res["scores"], res["labels"]):
                    # tensor to numpy
                    bbox = bbox.cpu().detach().numpy()
                    score = score.cpu().detach().numpy()
                    label = label.cpu().detach().numpy()
                    # bbox format: xyxy -> xywh
                    x1, y1, x2, y2 = bbox
                    bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                    instance = {}
                    instance["image_id"] = image_id
                    instance["bbox"] = bbox
                    instance["score"] = float(score)
                    instance["category_id"] = int(label) + 1
                    instances.append(instance)

    tmp_out = os.path.join(tmp_base_path, 'tmp_results.json')
    with open(tmp_out, 'w') as f:
        json.dump(instances, f)  
          
    lvis_eval = LVISEval(data_cfg['lvis_annotations_file'], tmp_out, 'bbox')
    lvis_eval.run()
    lvis_dict = lvis_eval.get_results()
    
    val_metrics = {
        'map': lvis_dict['AP'],
        'map_c': lvis_dict['APc'],
        'map_f': lvis_dict['APf'],
        'map_r': lvis_dict['APr'],
    }
    
    if writer is not None:
        log_lvis(writer, epoch, lvis_metrics=val_metrics)
    
    return val_metrics

def update_metrics(metric, metadata, pred_boxes, pred_classes, scores, boxes, labels):
    pred_boxes = BoxUtil.scale_bounding_box(
        pred_boxes.cpu(), metadata["width"], metadata["height"], mode="up"
    )
    boxes = BoxUtil.scale_bounding_box(
        boxes.cpu(), metadata["width"], metadata["height"], mode="up"
    )

    preds = []
    for _pred_boxes, _pred_classes, _scores in zip(pred_boxes, pred_classes, scores):
        preds.append(
            {
                "boxes": _pred_boxes.to('cpu'),
                "scores": _scores.to('cpu'),
                "labels": _pred_classes.to('cpu'),
            }
        )

    targets = []
    for _boxes, _classes in zip(boxes, labels):
        targets.append(
            {
                "boxes": _boxes.to('cpu'),
                "labels": _classes.to('cpu'),
            }
        )

    metric.update(preds, targets)