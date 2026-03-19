import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
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

def ensemble_scores(scores):

    # scores: [num_queries, num_prompts]

    s_full = scores[:, 0]              # 原始句子
    s_attrs = scores[:, 1:-1]          # 属性
    s_obj = scores[:, -1]              # object

    if s_attrs.shape[1] > 0:
        s_attr_mean = s_attrs.mean(dim=1)
    else:
        s_attr_mean = 0

    # ⭐ 推荐权重（你可以做 ablation）
    final_score = 0.6 * s_full + 0.3 * s_attr_mean + 0.1 * s_obj

    return final_score
def parse_args():
    parser = argparse.ArgumentParser(description="OWLv2目标检测可视化调试")
    parser.add_argument("--image_path", type=str, required=True, default='demo',
                       help="输入图像路径（单张图片、文件夹或包含路径的txt文件）")
    parser.add_argument("--output_dir", type=str, required=True, default='vis_output',
                       help="输出结果保存目录")
    parser.add_argument("--model_name", type=str, default="google/owlv2-base-patch16",
                       help="预训练模型名称")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="检测置信度阈值")
    parser.add_argument("--display_thr", type=float, default=0.1,
                       help="可视化显示阈值")
    parser.add_argument('--attr_enhance', action='store_true', help='使用属性增强测试')
    parser.add_argument("--use_nms", action="store_true",
                       help="是否使用NMS进行后处理")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="运行设备（cuda/cpu）")
    return parser.parse_args()

def extract_color_from_prompt(prompt):
    """从提示词中提取颜色信息"""
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
    
    prompt_lower = prompt.lower()
    for color_name, rgb in COLOR_NAME_TO_RGB.items():
        if color_name in prompt_lower:
            return rgb
    return None

def get_text_position(x1, y1, text_w, text_h, image_h, image_w, used_text_boxes):
    """获取非重叠的文本位置"""
    candidates = [
        (x1, y1 - 5),                    # 上方
        (x1, y1 + text_h + 5),           # 下方
        (x1 + 5, y1),                    # 右侧
        (x1 - text_w - 5, y1)            # 左侧
    ]
    
    for tx, ty in candidates:
        # 边界约束
        tx = max(0, min(tx, image_w - text_w - 1))
        ty = max(text_h, min(ty, image_h - 1))
        
        new_box = [tx, ty - text_h, tx + text_w, ty]
        
        # 检查重叠
        overlap = False
        for used_box in used_text_boxes:
            xx1 = max(new_box[0], used_box[0])
            yy1 = max(new_box[1], used_box[1])
            xx2 = min(new_box[2], used_box[2])
            yy2 = min(new_box[3], used_box[3])
            
            if xx2 > xx1 and yy2 > yy1:
                overlap = True
                break
        
        if not overlap:
            used_text_boxes.append(new_box)
            return tx, ty
    
    # 所有候选位置都重叠，返回第一个位置
    tx, ty = candidates[0]
    tx = max(0, min(tx, image_w - text_w - 1))
    ty = max(text_h, min(ty, image_h - 1))
    used_text_boxes.append([tx, ty - text_h, tx + text_w, ty])
    return tx, ty

def load_image_paths(image_path):
    """加载图像路径列表"""
    if isinstance(image_path, list):
        return image_path
    
    if os.path.isfile(image_path):
        # 如果是txt文件，读取每行路径
        if image_path.lower().endswith('.txt'):
            with open(image_path, 'r') as f:
                paths = [line.strip() for line in f if line.strip()]
                return [p for p in paths if os.path.exists(p)]
        # 如果是单张图片
        else:
            return [image_path] if os.path.exists(image_path) else []
    
    elif os.path.isdir(image_path):
        # 获取文件夹中所有图像文件
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_list = []
        for filename in os.listdir(image_path):
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                image_list.append(os.path.join(image_path, filename))
        return image_list
    
    return []

def apply_nms(boxes, scores, labels, iou_threshold=0.5):
    """应用非极大值抑制"""
    if len(boxes) == 0:
        return boxes, scores, labels
    
    import torch
    from torchvision.ops import nms
    
    # 转换为tensor
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    
    # 应用NMS
    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    if keep_indices.numel() == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int64)
    
    keep_indices = keep_indices.tolist()
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]

def create_color_map(prompts):
    """为提示词创建颜色映射"""
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
    
    color_map = {}
    palette_idx = 0
    
    for prompt in prompts:
        color = extract_color_from_prompt(prompt)
        if color is not None:
            color_map[prompt] = color
        else:
            color_map[prompt] = COLOR_PALETTE[palette_idx % len(COLOR_PALETTE)]
            palette_idx += 1
    
    return color_map

def visualize_detections(image_np, boxes, scores, labels, prompts, color_map, display_thr=0.4):
    """在图像上可视化检测结果"""
    if len(boxes) == 0:
        return image_np
    
    used_text_boxes = []
    image_h, image_w = image_np.shape[:2]
    
    # 按置信度降序排序
    sorted_indices = np.argsort(scores)[::-1]
    
    for idx in sorted_indices:
        score = scores[idx]
        if score < display_thr:
            continue
            
        box = boxes[idx]
        label = labels[idx]
        x1, y1, x2, y2 = map(int, box)
        class_name = prompts[label]
        color = color_map[class_name]
        
        # 绘制边界框
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        text = f"{class_name}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # 获取文本位置并绘制
        text_x, text_y = get_text_position(x1, y1, text_w, text_h, image_h, image_w, used_text_boxes)
        cv2.putText(image_np, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image_np
def process_image(image_paths, prompts, args, processor, model):
        
    # 创建颜色映射
    color_map = create_color_map(prompts)
    
     # 处理每张图像
    for img_idx, img_path in enumerate(image_paths, 1):
        print(f"处理图像 {img_idx}/{len(image_paths)}: {os.path.basename(img_path)}")
        try:
            # 读取图像
            image_pil = Image.open(img_path).convert("RGB")
            image_np = np.array(image_pil)
            
            # 预处理

            inputs = processor(
            text=prompts,
            images=image_pil,
            return_tensors="pt",
            padding=True
            ).to(device)
        
        # 推理
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 后处理
            target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)
            results = processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=args.threshold
            )[0]
            
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()
            # 可选：应用NMS
            if args.use_nms and len(boxes) > 0:
                boxes, scores, labels = apply_nms(boxes, scores, labels)
            
            # 可视化
            vis_image = visualize_detections(image_np.copy(), boxes, scores, labels, prompts, color_map, args.display_thr)
            
            # 保存结果
            save_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_result.jpg"
            save_path = os.path.join(args.output_dir, save_name)
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"  结果保存至: {save_path}")
                
        except Exception as e:
            import pdb;pdb.set_trace()
            print(f"  处理失败 {img_path}: {e}")
            continue
def process_images_attr_enhanced(image_paths, prompts, args, processor, model, device):
        
    # 创建颜色映射
    color_map = create_color_map(prompts)
    
     # 处理每张图像
    # 1. 构建增强的prompt组
    prompts_groups = build_all_prompts(prompts)  # 二维列表：[num_groups, prompts_per_group]
    # 2. 合并所有prompt并记录索引
    all_prompts = []
    group_indices = []
    group_start_idx = 0
    print(prompts_groups)
    for group in prompts_groups:
        all_prompts.extend(group )
        group_end_idx = group_start_idx + len(group)
        group_indices.append((group_start_idx, group_end_idx))
        group_start_idx = group_end_idx
    
   
    for img_idx, img_path in enumerate(image_paths, 1):
        print(f"处理图像 {img_idx}/{len(image_paths)}: {os.path.basename(img_path)}")
        # 读取图像
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)
            # 3. 一次性处理所有prompt
        inputs = processor(
            text=all_prompts, 
            images=image_np, 
            return_tensors="pt", 
            padding="max_length", 
            # max_length=16
        ).to(device)
        # 预处理
    
    # 推理
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 后处理
        target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)

        all_raw_scores = torch.sigmoid(outputs['logits'][0])  # [num_queries, total_prompts]
        boxes = outputs['pred_boxes'][0].cpu().detach().numpy()  # [num_queries, 4]

        # 7. 按组提取并计算集成得分
        all_scores_ensemble = []
        for start_idx, end_idx in group_indices:
            group_scores = all_raw_scores[:, start_idx:end_idx]  # [num_queries, group_size]
            final_score = ensemble_scores(group_scores)           # [num_queries]
            all_scores_ensemble.append(final_score.unsqueeze(-1))
        
        all_scores_ensemble = torch.cat(all_scores_ensemble, dim=1)  # [num_queries, num_groups]

        # 8. 获取最佳得分和标签
        logits = torch.max(all_scores_ensemble, dim=-1)
        scores = logits.values.cpu().detach().numpy()
        labels = logits.indices.cpu().detach().numpy()
        filter_idx = scores > args.threshold

        scores = scores[filter_idx, ]
        labels = labels[filter_idx, ]
        boxes = boxes[filter_idx]
        target_sizes = target_sizes.cpu()
        box_list = []
        for box in boxes:
            box_list.append(convert_to_x1y1x2y2(box,max(image_np.shape), max(image_np.shape)))
        boxes = np.array(box_list)
        # boxes = boxes * np.array([[target_sizes[0,1],target_sizes[0,0],target_sizes[0,1],target_sizes[0,0]]])
        # results = processor.post_process_object_detection(
        #     outputs=outputs,
        #     target_sizes=target_sizes,
        #     threshold=args.threshold
        # )[0]
        
        # boxes = results["boxes"].cpu().numpy()
        # scores = results["scores"].cpu().numpy()
        # labels = results["labels"].cpu().numpy()
        # 可选：应用NMS
        if args.use_nms and len(boxes) > 0:
            boxes, scores, labels = apply_nms(boxes, scores, labels)
        
        # 可视化
        vis_image = visualize_detections(image_np.copy(), boxes, scores, labels, prompts, color_map, args.display_thr)
        
        # 保存结果
        save_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_result.jpg"
        save_path = os.path.join(args.output_dir, save_name)
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"  结果保存至: {save_path}")
            
        # except Exception as e:
        #     import pdb;pdb.set_trace()
        #     print(f"  处理失败 {img_path}: {e}")
        #     continue
def main(args):
    """主函数"""
    # 初始化设备和模型
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载处理器和模型
    print("加载模型...")
    processor = Owlv2Processor.from_pretrained(args.model_name)
    model = Owlv2ForObjectDetection.from_pretrained(args.model_name).to(device)
    model.eval()
    
    # 定义检测提示词
    prompts = [
        "carrot", 
        "ox",
        # "person",
        "red apple",
        "green apple",
        "yellow apple",
        "green leaves",
        "yellow leaves",
        "orange leaves",
        "a person with blue clothes",
        "a person with black clothes",
        "a person with red clothes",
        "a person with long hair and dressed black clothes",
        "a person without long hair and dressed black clothes",
        "round rock",
        "square rock",
        "wooden toy",
        "metal toy"
    ]

    # 加载图像路径
    image_paths = load_image_paths(args.image_path)
    if not image_paths:
        print(f"错误：未找到有效图像文件，请检查路径: {args.image_path}")
        return
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.attr_enhance:
        process_images_attr_enhanced(image_paths, prompts, args, processor, model, device)
    else:
        process_images(image_paths, prompts, args, processor, model, device)
    
    print("处理完成！")

if __name__ == "__main__":
    args = parse_args()
    main(args)