import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

def parse_args():
    parser = argparse.ArgumentParser(description="OWLv2目标检测可视化调试")
    parser.add_argument("--image_path", type=str, required=True,
                       help="输入图像路径（单张图片、文件夹或包含路径的txt文件）")
    parser.add_argument("--output_dir", type=str, required=True, default='vis_output',
                       help="输出结果保存目录")
    parser.add_argument("--model_name", type=str, default="google/owlv2-base-patch16",
                       help="预训练模型名称")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="检测置信度阈值")
    parser.add_argument("--display_thr", type=float, default=0.1,
                       help="可视化显示阈值")
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
        "person",
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
    
    # 创建颜色映射
    color_map = create_color_map(prompts)
    
    # 加载图像路径
    image_paths = load_image_paths(args.image_path)
    if not image_paths:
        print(f"错误：未找到有效图像文件，请检查路径: {args.image_path}")
        return
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    print("处理完成！")

if __name__ == "__main__":
    args = parse_args()
    main(args)