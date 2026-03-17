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

def save_object(obj, path):
    """"Save an object using the pickle library on a file
    
    :param obj: undefined. Object to save
    :param fileName: str. Name of the file of the object to save
    """
    print("Saving " + path)
    with open(path, 'wb') as fid:
        pickle.dump(obj, fid)
        
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
        model = Owlv2ForObjectDetection.from_pretrained("epoch3_baseline/")
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

def visualization(model,
                  processor,
                  image_path,
                  prompt,
                  out,
                  threshold=0.2,
                  use_nms=False,
                  colors=None,
                  display_thr=0.4):
    """
    通用可视化函数（兼容 model(**inputs)）

    Args:
        model: 已加载模型（Owlv2 / OwlViT）
        processor: 对应 processor
        image_path: str（单图 / 文件夹 / list）
        prompt: list[str]
        out: 输出目录
        threshold: 置信度阈值
        use_nms: 是否使用NMS（复用你已有函数）
        colors: 配色
        display_thr: 显示出置信度高于多少的目标
    """

    import os
    import cv2
    import torch
    import numpy as np
    from PIL import Image

    def extract_color_from_prompt(prompt):
        COLOR_NAME_TO_RGB = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "pink": (255, 192, 203),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128),
            "brown": (165, 42, 42),
            "cyan": (0, 255, 255)
        }
        prompt = prompt.lower()

        for color_name in COLOR_NAME_TO_RGB:
            if color_name in prompt:
                return COLOR_NAME_TO_RGB[color_name]

        return None

    COLOR_PALETTE = [
        (230, 159, 0),
        (86, 180, 233),
        (0, 158, 115),
        (240, 228, 66),
        (0, 114, 178),
        (213, 94, 0),
        (204, 121, 167),
        (0, 0, 0)
    ]
    if colors is not None and hasattr(colors, "__iter__"):
        colors = list(colors) + COLOR_PALETTE
    else:
        colors = COLOR_PALETTE
    color_map = {}
    palette_idx = 1
    for p in prompt:
        color = extract_color_from_prompt(p)
        if color is not None:
            color_map[p] = color
        else:
            color_map[p] = colors[palette_idx % len(colors)]
            palette_idx += 1

    os.makedirs(out, exist_ok=True)

    # 🔹 1. 解析图片路径
    if isinstance(image_path, str):
        if os.path.isdir(image_path):
            image_list = [os.path.join(image_path, x) for x in os.listdir(image_path)]
        else:
            image_list = [image_path]
    elif isinstance(image_path, list):
        image_list = image_path
    else:
        raise ValueError("image_path must be str or list")

    model.eval()

    for img_path in image_list:

        # 🔹 2. 读取图片
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)

        # 🔹 3. 构造输入（关键：统一接口）
        inputs = processor(text=prompt,
                           images=image_pil,
                           return_tensors="pt",
                           padding=True).to(device)

        # 🔹 4. inference（核心要求）
        with torch.no_grad():
            outputs = model(**inputs)

        # 🔥 两种后处理方式（你可以选）

        # =========================
        # ✅ 方法1：官方推荐（更标准）
        # =========================
        target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)

        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        # =========================
        # ❗（可选）用你自己的 decode逻辑
        # =========================
        # 如果你想完全复用 evaluate_image：
        # 可以替换成你那一套逻辑

        # 🔹 5. NMS（可选）
        if use_nms and len(boxes) > 0:
            boxes_list = [box for box in boxes]
            scores_list = [float(s) for s in scores]
            labels_list = [int(l) for l in labels]

            boxes_list, scores_list, labels_list, _ = apply_NMS(
                boxes_list, scores_list, labels_list, scores_list
            )

            boxes = np.array(boxes_list)
            scores = np.array(scores_list)
            labels = np.array(labels_list)

        # 🔹 6. 可视化

        used_text_boxes = []
        for box, score, label in sorted(zip(boxes, scores, labels), key=lambda x: x[1], reverse=True):
            if score < display_thr:
                continue

            x1, y1, x2, y2 = map(int, box)
            class_name = prompt[label]
            color = color_map[class_name]

            # 画框
            cv2.rectangle(image_np,
                          (x1, y1),
                          (x2, y2),
                          color,
                          2)

            # 标签
            text = f"{prompt[label]}: {score:.2f}"

            def get_non_overlapping_text_pos(x1, y1, text_w, text_h, image_h, image_w):

                # 候选位置（优先顺序）
                candidates = [
                    (x1, y1 - 5),                      # 上
                    (x1, y1 + text_h + 5),             # 下
                    (x1 + 5, y1),                      # 右
                    (x1 - text_w - 5, y1)              # 左
                ]

                for (tx, ty) in candidates:

                    # 边界修正
                    tx = max(0, tx)
                    ty = max(text_h, ty)
                    tx = min(image_w - text_w - 1, tx)
                    ty = min(image_h - 1, ty)

                    new_box = [tx, ty - text_h, tx + text_w, ty]

                    overlap = False

                    for box in used_text_boxes:
                        # 判断是否重叠（IoU思想）
                        xx1 = max(new_box[0], box[0])
                        yy1 = max(new_box[1], box[1])
                        xx2 = min(new_box[2], box[2])
                        yy2 = min(new_box[3], box[3])

                        if xx2 > xx1 and yy2 > yy1:
                            overlap = True
                            break

                    if not overlap:
                        used_text_boxes.append(new_box)
                        return tx, ty

                # ❗ 全部重叠 → 强行返回
                tx, ty = candidates[0]
                used_text_boxes.append([tx, ty - text_h, tx + text_w, ty])
                return tx, ty

            (text_w, text_h), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                2
            )
            text_x, text_y = get_non_overlapping_text_pos(
                x1, y1, text_w, text_h, image_np.shape[0], image_np.shape[1]
            )
            cv2.putText(image_np,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)

        # 🔹 7. 保存
        save_path = os.path.join(out, os.path.basename(img_path))
        cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        print(f"[Saved] {save_path}")

if __name__ == '__main__':
    # main()
    def visualization_debug():
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16").to(device)

        visualization(
            model=model,
            processor=processor,
            image_path="/irip/liaozhixuan_2023/cz/FGOVD/visualization_debug/input",
            prompt=[
                "carrot", 
                "ox",
                "red apple",
                "green apple",
                "yellow apple",
                "green leaves",
                "yellow leaves",
                "orange leaves",
                "a people with blue clothes",
                "a people with red clothes",
                "a people with long hair and dressed black clothes",
                "a people without long hair and dressed black clothes",
                "round rock",
                "wooden toy",
                "metal toy"
            ],
            use_nms=True,
            out="/irip/liaozhixuan_2023/cz/FGOVD/visualization_debug/output",
            display_thr=0.1,
        )
    visualization_debug()