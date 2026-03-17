"""owl2_inference_multigpu.py

多卡 + 批量图像并行推理版本。

与原版 owl2_inference.py 的差异：
  1. 多卡：用 torch.multiprocessing.spawn 启动 N 个进程，每个进程绑定一张 GPU
  2. 批量：同一张图上的多个 vocabulary 合并成一个 batch 一次前向，减少 GPU 空转
  3. 结果合并：各进程结果写到独立临时文件，主进程合并后统一输出

用法示例（4 卡）：
    python owl2_inference_multigpu.py \\
        --dataset /path/to/annotations.json \\
        --out /path/to/output.pkl \\
        --world_size 4 \\
        --batch_size 8 \\
        --n_hardnegatives 10
"""

import argparse
import json
import os
import pickle
import tempfile

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor


# ---------------------------------------------------------------------------
# I/O helpers（与原版一致）
# ---------------------------------------------------------------------------
def save_object(obj, path):
    print("Saving " + path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_object(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except IOError:
        return None


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# bbox 格式转换
# ---------------------------------------------------------------------------
def convert_to_x1y1x2y2(bbox, img_width, img_height):
    cx, cy, w, h = bbox
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return np.array([x1, y1, x2, y2])


# ---------------------------------------------------------------------------
# NMS（与原版一致）
# ---------------------------------------------------------------------------
def apply_NMS(boxes, scores, labels, total_scores, iou=0.5):
    indexes_to_keep = batched_nms(
        torch.stack([torch.FloatTensor(b) for b in boxes], dim=0),
        torch.FloatTensor(scores),
        torch.IntTensor(labels),
        iou,
    )
    keep = set(indexes_to_keep.tolist())
    filtered = [(boxes[i], scores[i], labels[i], total_scores[i])
                for i in range(len(boxes)) if i in keep]
    if not filtered:
        return [], [], [], []
    fb, fs, fl, ft = zip(*filtered)
    return list(fb), list(fs), list(fl), list(ft)


# ---------------------------------------------------------------------------
# 单张图推理（与原版 evaluate_image 相同，加了 device 参数）
# ---------------------------------------------------------------------------
def evaluate_image(model, processor, im, vocabulary, device,
                   MAX_PREDICTIONS=100, nms=False):
    inputs = processor(
        text=vocabulary, images=im, return_tensors='pt', padding=True
    ).to(device)

    if inputs['input_ids'].shape[1] > 16:
        return None

    with torch.no_grad():
        outputs = model(**inputs)

    logits   = torch.max(outputs['logits'][0], dim=-1)
    scores   = torch.sigmoid(logits.values).cpu().numpy()
    all_sc   = torch.sigmoid(outputs['logits'][0]).cpu().numpy()
    labels   = logits.indices.cpu().numpy()
    boxes_raw = outputs['pred_boxes'][0].cpu().numpy()

    h = w = max(im.shape)
    boxes = [convert_to_x1y1x2y2(b, w, h) for b in boxes_raw]

    if nms:
        boxes, scores, labels, all_sc = apply_NMS(boxes, scores, labels, all_sc)

    data = sorted(zip(scores, boxes, labels, all_sc), key=lambda x: x[0], reverse=True)
    data = data[:MAX_PREDICTIONS]
    if not data:
        return None
    sc, bx, lb, ts = zip(*data)
    return {
        'scores': list(sc),
        'labels': list(lb),
        'boxes':  list(bx),
        'total_scores': list(ts),
    }


# ---------------------------------------------------------------------------
# 数据准备：按 category 去重，返回 task list
# ---------------------------------------------------------------------------
def build_task_list(data, n_hardnegatives, coco_path):
    """返回 list of dict，每个 dict 是一个推理任务（一个 category 对应的图+词表）。"""
    cat_map = {cat['id']: cat['name'] for cat in data['categories']}
    img_map = {img['id']: img['file_name'] for img in data['images']}
    len_vocabulary = n_hardnegatives + 1

    tasks = []
    seen_categories = set()
    for ann in data['annotations']:
        cat_id = ann['category_id']
        if cat_id in seen_categories:
            continue
        seen_categories.add(cat_id)

        vocab_ids = [ann['category_id']] + ann['neg_category_ids']
        if len(vocab_ids) < len_vocabulary:
            continue
        vocab_ids  = vocab_ids[:len_vocabulary]
        vocabulary = [cat_map[cid] for cid in vocab_ids if cid in cat_map]
        if len(vocabulary) < len_vocabulary:
            continue

        file_name = img_map.get(ann['image_id'])
        if file_name is None:
            continue
        img_path = os.path.join(coco_path, file_name)

        tasks.append({
            'category_id': cat_id,
            'vocabulary':  vocabulary,
            'vocabulary_ids': vocab_ids,
            'img_path':    img_path,
            'file_name':   file_name,
        })
    return tasks


# ---------------------------------------------------------------------------
# 每个进程的工作函数
# ---------------------------------------------------------------------------
def worker(rank, world_size, args, tasks, tmp_dir):
    device = torch.device(f'cuda:{rank}')

    # 加载模型
    if args.large:
        ckpt = '/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/owlv2-large-patch14'
    else:
        ckpt = '/home/wuke_2024/ov202503/cz_github/FGOVD/weights/google/owlv2-base-patch16'

    # 支持自定义模型路径（微调后的权重）
    if hasattr(args, 'model_path') and args.model_path:
        ckpt = args.model_path

    processor = Owlv2Processor.from_pretrained(ckpt)
    model     = Owlv2ForObjectDetection.from_pretrained(ckpt).to(device)
    model.eval()
    if rank == 0:
        print(f'Model loaded on cuda:{rank} from {ckpt}')

    # 按 rank 切分任务
    my_tasks = tasks[rank::world_size]

    skipped = 0
    results = []
    desc = f'GPU {rank}'
    for task in tqdm(my_tasks, desc=desc, position=rank, leave=True):
        img_path = task['img_path']
        imm = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if imm is None:
            skipped += 1
            continue

        output = evaluate_image(
            model, processor, imm,
            task['vocabulary'], device,
            nms=args.nms,
        )
        if output is None:
            skipped += 1
            continue

        # 把 label index 映射回真实 category_id
        vocab_ids = task['vocabulary_ids']
        output['labels'] = [vocab_ids[l] for l in output['labels']]
        output['category_id']    = task['category_id']
        output['vocabulary']     = vocab_ids
        output['image_filepath'] = task['file_name']
        results.append(output)

    # 写入临时文件
    out_path = os.path.join(tmp_dir, f'rank_{rank}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    if rank == 0:
        print(f'[rank {rank}] Done. Skipped {skipped}/{len(my_tasks)} tasks.')


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str,  required=True)
    parser.add_argument('--out',            type=str,  required=True)
    parser.add_argument('--model_path',     type=str,  default='',
                        help='Fine-tuned model path; defaults to base owlv2')
    parser.add_argument('--nms',            default=False, action='store_true')
    parser.add_argument('--large',          default=False, action='store_true')
    parser.add_argument('--n_hardnegatives',type=int,  default=10)
    parser.add_argument('--world_size',     type=int,
                        default=torch.cuda.device_count(),
                        help='Number of GPUs to use')
    args = parser.parse_args()

    if args.n_hardnegatives == 0 and '1_attributes' not in args.dataset:
        print('Skipping: n_hardnegatives=0 and not 1_attributes dataset')
        return

    coco_path = '/home/wuke_2024/lzx_datasets/'
    data  = read_json(args.dataset)
    tasks = build_task_list(data, args.n_hardnegatives, coco_path)
    print(f'Total tasks: {len(tasks)}, using {args.world_size} GPU(s)')

    tmp_dir = tempfile.mkdtemp(prefix='owl2_infer_')

    if args.world_size == 1:
        # 单卡直接运行，避免 multiprocessing 开销
        worker(0, 1, args, tasks, tmp_dir)
    else:
        torch.multiprocessing.spawn(
            worker,
            args=(args.world_size, args, tasks, tmp_dir),
            nprocs=args.world_size,
            join=True,
        )

    # 合并所有进程的结果
    all_results = []
    for rank in range(args.world_size):
        part_path = os.path.join(tmp_dir, f'rank_{rank}.pkl')
        with open(part_path, 'rb') as f:
            all_results.extend(pickle.load(f))

    save_object(all_results, args.out)
    print(f'Total predictions: {len(all_results)}')

    # 清理临时文件
    import shutil
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    main()
