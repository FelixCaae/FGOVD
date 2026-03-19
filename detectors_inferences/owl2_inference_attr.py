import argparse
import torch

from tqdm import tqdm

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from torchvision.ops import batched_nms
from PIL import Image
# import some common libraries
import sys
import numpy as np
import os, cv2, random
from skimage import io as skimage_io

from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTTextConfig
from transformers import Owlv2Processor, Owlv2ForObjectDetection

import numpy as np

import pickle, json

# READ/WRITE UTILITIES

import spacy

nlp = spacy.load("en_core_web_sm")

COLORS = ['black', 'light_blue', 'blue', 'dark_blue', 'light_brown', 'brown', 'dark_brown', 'light_green', 'green', 'dark_green', 'light_grey', 'grey', 'dark_grey', 'light_orange', 'orange', 'dark_orange', 'light_pink', 'pink', 'dark_pink', 'light_purple', 'purple', 'dark_purple', 'light_red', 'red', 'dark_red', 'white', 'light_yellow', 'yellow', 'dark_yellow']
MATERIALS = ['text', 'stone', 'wood', 'rattan', 'fabric', 'crochet', 'wool', 'leather', 'velvet', 'metal', 'paper', 'plastic', 'glass', 'ceramic']
PATTERNS = ['plain', 'striped', 'dotted', 'checkered', 'woven', 'studded', 'perforated', 'floral', 'logo']
TRANSPARENCIES = ['opaque', 'translucent', 'transparent']

ATTRIBUTE_VOCAB = {
    "color": COLORS,
    "material": MATERIALS,
    "pattern": PATTERNS,
    "transparency": TRANSPARENCIES
}

def extract_attributes(sentence):
    doc = nlp(sentence)
    attrs = []
    obj = None

    tokens = list(doc)
    i = 0
    vocab_set = set()
    for v in ATTRIBUTE_VOCAB.values():
        vocab_set.update(v)

    while i < len(tokens):
        token = tokens[i]
        if token.pos_ == "ADJ":
            if token.text.lower() in ["light", "dark"]:
                combined_attrs = token.text.lower() + " " + tokens[i+1].text.lower()
                attrs.append(combined_attrs)
                i+=2
                continue
            else:
                attrs.append(token.text.lower())
        if token.pos_ == "NOUN":
            if token.text.lower() in vocab_set:
                attrs.append(token.text.lower())
                i += 1
                continue
            else:
                obj = token.text.lower()
        i+=1
    return attrs, obj

def build_attribute_prompts(attrs, obj):
    prompts = []
    for a in attrs:
        prompts.append(f"a {a} {obj}")
    return prompts

def build_all_prompts(vocabulary):

    all_prompt_groups = []

    for sent in vocabulary:

        attrs, obj = extract_attributes(sent)

        attr_prompts = build_attribute_prompts(attrs, obj)
        obj_prompt = f"a {obj}"

        # 最终 prompt 组
        prompts = [sent] + attr_prompts + [obj_prompt]

        all_prompt_groups.append(prompts)

    return all_prompt_groups

def ensemble_scores(scores, w0=1.0, w1=0.2, w2=0.):

    # scores: [num_queries, num_prompts]

    s_full = scores[:, 0]              # 原始句子
    s_attrs = scores[:, 1:-1]          # 属性
    s_obj = scores[:, -1]              # object

    if s_attrs.shape[1] > 0:
        s_attr_mean = s_attrs.mean(dim=1)
    else:
        s_attr_mean = 0

    # ⭐ 推荐权重（你可以做 ablation）
    final_score = w0 * s_full + w1 * s_attr_mean + w2 * s_obj

    return final_score

def save_object(obj, path):
    """
    Save an object using the pickle library to a file.
    Automatically creates the directory if it doesn't exist.
    
    :param obj: any python object. Object to save
    :param path: str or Path. File path to save the object
    """
    from pathlib import Path
    # 将路径转换为Path对象以便统一处理
    save_path = Path(path)
    
    # 获取父目录
    save_dir = save_path.parent
    
    # 如果父目录不存在，则创建（包括所有中间目录）
    if not save_dir.exists():
        print(f"Creating directory: {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存对象
    print(f"Saving to: {save_path}")
    with open(save_path, 'wb') as fid:
        pickle.dump(obj, fid)
    print(f"Save completed.")

def load_object(path):
    """"Load an object from a file
    
    :param fileName: str. Name of the file of the object to load
    :return: obj: undefined. Object loaded
    """
    try:
        with open(path, 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except IOError:
        return None   

def read_json(file_name):
    #Read JSON file
    with open(file_name) as infile:
        data = json.load(infile)
    return data

def write_json(data, file_name):
    # Write JSON file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile)

def convert_to_x1y1x2y2(bbox, img_width, img_height):
    """
    Convert bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Args:
        bbox (np.array): NumPy array of bounding boxes in the format [cx, cy, w, h].
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        np.array: NumPy array of bounding boxes in the format [x1, y1, x2, y2].
    """
    cx, cy, w, h = bbox
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return np.array([x1, y1, x2, y2])

def apply_NMS(boxes, scores, labels, total_scores, iou=0.5):
    indexes_to_keep = batched_nms(torch.stack([torch.FloatTensor(box) for box in boxes], dim=0),
                       torch.FloatTensor(scores),
                       torch.IntTensor(labels),
                       iou)
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    filtered_total_scores = []
    deleted_boxes = []
    deleted_scores = []
    deleted_labels = []
    deleted_total_scores = []
    
    for x in range(len(boxes)):
        if x in indexes_to_keep:
            filtered_boxes.append(boxes[x])
            filtered_scores.append(scores[x])
            filtered_labels.append(labels[x])
            filtered_total_scores.append(total_scores[x])
        else:
            deleted_boxes.append(boxes[x])
            deleted_scores.append(scores[x])
            deleted_labels.append(labels[x])
            deleted_total_scores.append(total_scores[x])
    
    return filtered_boxes, filtered_scores, filtered_labels, filtered_total_scores

skipped_categories = 0
def evaluate_image(model, processor, im, vocabulary, MAX_PREDICTIONS=100, nms=False):
    global skipped_categories
    # preparing the inputs
    inputs = processor(text=vocabulary, images=im, return_tensors="pt", padding=True).to(device)
    
    # if the tokens length is above 16, the model can't handle them
    if  inputs['input_ids'].shape[1] > 16:
        skipped_categories += 1
        return None
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        
        
    # Get prediction logits
    logits = torch.max(outputs['logits'][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()
    all_scores = torch.sigmoid(outputs['logits'][0]).cpu().detach().numpy()
    
    # Get prediction labels and boundary boxes
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs['pred_boxes'][0].cpu().detach().numpy()    
        
    scores_filtered = []
    labels_filtered = []
    boxes_filtered = []
    total_scores_filtered = []
    height = max(im.shape)
    width = max(im.shape)
    
    boxes = [convert_to_x1y1x2y2(box, width, height) for box in boxes]
    # Combine the lists into tuples using zip
    if nms:
        # apply NMS
        boxes, scores, labels, all_scores = apply_NMS(boxes, scores, labels, all_scores)
    data = list(zip(scores, boxes, labels, all_scores))

    # Sort the combined data based on the first element of each tuple (score) in decreasing order
    sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
    
    
    # filtering the predictions with low confidence
    for score, box, label, total_scores in sorted_data[:MAX_PREDICTIONS]:
        scores_filtered.append(score)
        labels_filtered.append(label)
        # boxes_filtered.append(convert_to_x1y1x2y2(box, width, height))
        boxes_filtered.append(box)
        total_scores_filtered.append(total_scores)
    
    return {
        'scores': scores_filtered,
        'labels': labels_filtered,
        'boxes': boxes_filtered,
        'total_scores': total_scores_filtered
    }
        
skipped_categories = 0
def evaluate_image_attr(model, processor, im, vocabulary, MAX_PREDICTIONS=100, nms=False):
    global skipped_categories
    
    # 1. 构建增强的prompt组
    prompts_groups = build_all_prompts(vocabulary)  # 二维列表：[num_groups, prompts_per_group]
    # 2. 合并所有prompt并记录索引
    all_prompts = []
    group_indices = []
    group_start_idx = 0
    
    for prompts in prompts_groups:
        all_prompts.extend(prompts)
        group_end_idx = group_start_idx + len(prompts)
        group_indices.append((group_start_idx, group_end_idx))
        group_start_idx = group_end_idx
    
    # 3. 一次性处理所有prompt
    inputs = processor(
        text=all_prompts, 
        images=im, 
        return_tensors="pt", 
        padding=True, 
        # max_length=16
    ).to(device)
    
    # 4. 检查token长度
    if inputs['input_ids'].shape[1] > 16:
        skipped_categories += 1
        return None
    
    # 5. 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 6. 获取所有得分
    all_raw_scores = torch.sigmoid(outputs['logits'][0])  # [num_queries, total_prompts]
    boxes = outputs['pred_boxes'][0].cpu().detach().numpy()  # [num_queries, 4]
    # 7. 按组提取并计算集成得分
    all_scores = []
    for start_idx, end_idx in group_indices:
        group_scores = all_raw_scores[:, start_idx:end_idx]  # [num_queries, group_size]
        final_score = ensemble_scores(group_scores)           # [num_queries]
        all_scores.append(final_score.unsqueeze(-1))
    
    all_scores = torch.cat(all_scores, dim=1)  # [num_queries, num_groups]
    
    # 8. 获取最佳得分和标签
    logits = torch.max(all_scores, dim=-1)
    scores = logits.values.cpu().detach().numpy()
    labels = logits.indices.cpu().detach().numpy()
    
    # 9. 后处理
    scores_filtered = []
    labels_filtered = []
    boxes_filtered = []
    total_scores_filtered = []
    
    # 确保使用正确的高度和宽度
    height = max(im.shape)
    width = max(im.shape)
    
    # 转换边界框格式
    boxes_converted = [convert_to_x1y1x2y2(box, width, height) for box in boxes]
    
    # 应用NMS
    if nms:
        # 注意：这里需要传入正确的total_scores
        # 如果NMS需要使用原始所有得分，可以传入all_scores对应的行
        boxes_converted, scores, labels, all_scores = apply_NMS(
            boxes_converted, 
            scores, 
            labels, 
            all_scores.cpu().numpy()  # 传入集成的总得分
        )
    
    # 排序和过滤
    data = list(zip(scores, boxes_converted, labels, all_scores.cpu().numpy()))
    sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
    
    for score, box, label, total_score_vector in sorted_data[:MAX_PREDICTIONS]:
        scores_filtered.append(score)
        labels_filtered.append(label)
        boxes_filtered.append(box)
        total_scores_filtered.append(total_score_vector)  # 保存整个向量
    
    return {
        'scores': scores_filtered,
        'labels': labels_filtered,
        'boxes': boxes_filtered,
        'total_scores': total_scores_filtered
    }
        
    

def get_category_name(id, categories):
    for category in categories:
        if id == category['id']:
            return category['name']
        
def get_image_filepath(id, images):
    for image in images:
        if id == image['id']:
            return image['file_name']

def create_vocabulary(ann, categories):
    vocabulary_id = [ann['category_id']] + ann['neg_category_ids']
    vocabulary = [get_category_name(id, categories) for id in vocabulary_id]
    
    return vocabulary, vocabulary_id

def adjust_out_id(output, vocabulary_id):
    for i in range(len(output['labels'])):
        output['labels'][i] = vocabulary_id[output['labels'][i]]
    return output

# def convert_to_standard_format(output):
#     return {
#             'labels': output['instances'].pred_classes.cpu().numpy().tolist(),
#             'boxes': output['instances'].pred_boxes.tensor.cpu().numpy().tolist(),
#             'scores': output['instances'].scores.cpu().numpy().tolist(),
#             'annotation_id': output['annotation_id'],
#             'image_filepath': output['image_filepath']
#     }


# def convert_to_standard_format_complete(outputs):
#     std_out = []
    
#     #print("Converting predictions in standard format")
#     for output in outputs:
#         std_out.append({
#             'labels': output['labels'].cpu().numpy().tolist(),
#             'boxes': output['boxes'].tensor.cpu().numpy().tolist(),
#             'scores': output['scores'].cpu().numpy().tolist(),
#             'category_id': output['category_id'],
#             'image_filepath': output['image_filepath']
#         })

#     return std_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process')
    parser.add_argument('--out', type=str, required=True, help='Out path')
    parser.add_argument('--nms', default=False, action='store_true', help='If set it will be applied NMS with iou=0.5')
    parser.add_argument('--large', default=False, action='store_true', help='If set, it will be loaded the large model')
    parser.add_argument('--n_hardnegatives', type=int, default=10, help="Number of hardnegatives in each vocabulary")
    parser.add_argument('--attr', action='store_true')
    args = parser.parse_args()
    global skipped_categories
    
    coco_path = '/gpfsdata/home/yangshuai/data/coco/'
    if args.n_hardnegatives == 0 and '1_attributes' not in args.dataset:
        return
    # data = read_json('/home/lorenzobianchi/PacoDatasetHandling/jsons/captioned_%s.json' % dataset_name)
    data = read_json(args.dataset)
    if args.large:
        processor = Owlv2Processor.from_pretrained("/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/owlv2-large-patch14")
        model = Owlv2ForObjectDetection.from_pretrained("/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/owlv2-large-patch14")
        print("Large model loaded")
    else:
        processor = Owlv2Processor.from_pretrained("/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/owlv2-base-patch16")
        model = Owlv2ForObjectDetection.from_pretrained("/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/owlv2-base-patch16")
        print("Base model loaded")
    model = model.to(device)
    model.eval()
    
    complete_outputs = []
    categories_done = []
    for i, ann in enumerate(tqdm(data['annotations'])):
        # if the category is not done, we add it to the list
        if ann['category_id'] not in categories_done:
            categories_done.append(ann['category_id'])
        else:
            continue
        vocabulary, vocabulary_id = create_vocabulary(ann, data['categories'])
        # check if a number of hardnegatives is setted to non-default values
        # if it is, the vocabulary is clipped and if it is too short, we skip that image
        len_vocabulary = args.n_hardnegatives + 1
        if len(vocabulary) < len_vocabulary:
            continue
        vocabulary = vocabulary[:len_vocabulary]
        vocabulary_id = vocabulary_id[:len_vocabulary]
        image_filepath = coco_path + get_image_filepath(ann['image_id'], data['images'])
        imm = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
        # imm = Image.open(image_filepath)
        if args.attr:
            output = evaluate_image_attr(model, processor, imm, vocabulary, nms=args.nms)
        else:
            output = evaluate_image(model, processor, imm, vocabulary, nms=args.nms)
        if output == None:
            continue
        output['category_id'] = ann['category_id']
        output['vocabulary'] = vocabulary_id
        output['image_filepath'] = get_image_filepath(ann['image_id'], data['images'])
        output = adjust_out_id(output, vocabulary_id)
        complete_outputs.append(output)
        
    save_object(complete_outputs, args.out)
    print("Skipped categories: %d/%d" % (skipped_categories, len(categories_done)))


if __name__ == '__main__':
    main()
