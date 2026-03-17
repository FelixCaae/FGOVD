"""dataset_enhanced.py

在不改动原始 src/dataset.py 的前提下，提供两项数据增强功能：

功能一：expand_negatives_by_sampling
    从同数据集内其他正例的 category_id 中随机采样，把每条 annotation 的
    neg_category_ids 补充到 n_expanded_negatives 个。
    这是纯粹基于已有 category 池的扩充，不需要重新生成任何文本。

功能二：expand_negatives_by_rule
    参考原始负例生成词库（COLORS / MATERIALS / PATTERNS / TRANSPARENCIES），
    在训练数据加载阶段动态扫描正例句子，识别其中包含的属性词，
    再从同类候选池中随机替换，动态生成新的负例句子并注入 neg_category_ids。
    这是规则替换方案，无需 LLM、无需重新运行数据生成脚本。

功能三：mix_attribute_datasets
    从 data_cfg['train_annotations_files']（列表）中加载多个标注文件并合并，
    每个子文件分别调用上述扩充函数，最终合并为一个统一的 data 字典。
    若 data_cfg 中只有单个 train_annotations_file，自动回退到单文件逻辑。

使用方式：
    在 main_ddp_enhanced.py 中 import get_dataloaders_enhanced 替换原始
    get_dataloaders，其余训练代码不做任何修改。
"""

import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, OwlViTProcessor

from src.util import get_processor

# ---------------------------------------------------------------------------
# 属性词库（与原始 creates_hardnegatives.py 保持一致）
# ---------------------------------------------------------------------------
COLORS = [
    'black', 'light blue', 'blue', 'dark blue', 'light brown', 'brown',
    'dark brown', 'light green', 'green', 'dark green', 'light grey', 'grey',
    'dark grey', 'light orange', 'orange', 'dark orange', 'light pink', 'pink',
    'dark pink', 'light purple', 'purple', 'dark purple', 'light red', 'red',
    'dark red', 'white', 'light yellow', 'yellow', 'dark yellow',
]
MATERIALS = [
    'text', 'stone', 'wood', 'rattan', 'fabric', 'crochet', 'wool',
    'leather', 'velvet', 'metal', 'paper', 'plastic', 'glass', 'ceramic',
]
PATTERNS = [
    'plain', 'striped', 'dotted', 'checkered', 'woven',
    'studded', 'perforated', 'floral', 'logo',
]
TRANSPARENCIES = ['opaque', 'translucent', 'transparent']

# 颜色歧义组（近似色不作为负例，与原始代码一致）
AMBIGUITY_GROUPS = [
    {'dark blue', 'dark brown', 'dark grey', 'black'},
    {'white', 'light grey', 'light green', 'light blue', 'light brown'},
    {'brown', 'grey'},
    {'light orange', 'orange', 'light yellow', 'yellow'},
    {'light orange', 'orange', 'light red', 'red'},
    {'orange', 'dark orange', 'red', 'dark red'},
    {'orange', 'dark orange', 'yellow', 'dark yellow'},
    {'light pink', 'pink', 'light purple', 'purple', 'light red', 'red'},
    {'dark pink', 'pink', 'dark purple', 'purple', 'dark red', 'red'},
]


def _ambiguous_color_candidates(true_color: str, candidates: list) -> list:
    """从候选色列表中去除与 true_color 歧义的颜色。"""
    blocked = set()
    for group in AMBIGUITY_GROUPS:
        if true_color in group:
            blocked |= group
    blocked.discard(true_color)
    return [c for c in candidates if c not in blocked]


def _find_attribute_in_text(text: str, attr_list: list):
    """在句子中匹配属性词，返回 (matched_word, attr_value) 或 None。
    attr_value 是 attr_list 中的规范词（可能含空格，如 'light blue'）。
    按词长降序匹配，避免 'blue' 先于 'light blue' 匹配。
    """
    text_lower = text.lower()
    for attr in sorted(attr_list, key=len, reverse=True):
        pattern = r'\b' + re.escape(attr) + r'\b'
        m = re.search(pattern, text_lower)
        if m:
            return m.group(0), attr
    return None


# ---------------------------------------------------------------------------
# 功能一：基于采样的负例扩充
# ---------------------------------------------------------------------------
def expand_negatives_by_sampling(data: dict, training_cfg: dict) -> dict:
    """从同数据集内其他正例 category_id 中随机采样，补充负例到 n_expanded_negatives 个。

    参数：
        data          : 标准 COCO 格式 dict（含 images/annotations/categories）
        training_cfg  : 训练配置，需含 n_hardnegatives；
                        若含 n_expanded_negatives 且 > n_hardnegatives 则生效。
    返回：
        原地修改 data 后返回（不复制）。
    """
    n_target = training_cfg.get('n_expanded_negatives', training_cfg['n_hardnegatives'])
    if n_target <= training_cfg['n_hardnegatives']:
        return data  # 未配置或目标不超过原值，不做扩充

    all_pos_ids = list({ann['category_id'] for ann in data['annotations']})

    for ann in data['annotations']:
        current_negs = ann['neg_category_ids']
        if len(current_negs) >= n_target:
            continue
        excluded = set(current_negs) | {ann['category_id']}
        candidates = [cid for cid in all_pos_ids if cid not in excluded]
        random.shuffle(candidates)
        needed = n_target - len(current_negs)
        ann['neg_category_ids'] = current_negs + candidates[:needed]

    return data


# ---------------------------------------------------------------------------
# 功能二：基于规则替换的动态负例生成
# ---------------------------------------------------------------------------
def expand_negatives_by_rule(data: dict, training_cfg: dict, processor=None) -> dict:
    """扫描每条正例句子，识别属性词（颜色/材质/纹理/透明度），
    动态生成替换了该属性词的新负例句子，并以虚拟 category 形式注入数据。

    新生成的 category 会追加到 data['categories'] 末尾，
    新 category_id 从现有最大 id + 1 开始递增，
    ann['neg_category_ids'] 会被扩充（追加在原有负例之后）。

    参数：
        data          : 标准 COCO 格式 dict
        training_cfg  : 训练配置，需含 n_hardnegatives；
                        若含 n_rule_negatives（每条 ann 通过规则增加的上限数）则生效，
                        默认值为 5。
        processor     : 可选，AutoProcessor 实例，用于预先过滤 token 超长的生成文本。
                        若不传入则跳过长度检查（不推荐）。
    返回：
        原地修改 data 后返回。
    """
    n_rule = training_cfg.get('n_rule_negatives', 5)
    if n_rule <= 0:
        return data

    cat_map = {cat['id']: cat for cat in data['categories']}
    next_cat_id = max(cat['id'] for cat in data['categories']) + 1
    new_categories = []

    # 缓存：对同一 category_id 只生成一次规则负例，其他 ann 复用
    rule_neg_cache: dict = {}  # category_id -> list[new_category_id]

    for ann in data['annotations']:
        pos_cat_id = ann['category_id']

        if pos_cat_id in rule_neg_cache:
            # 复用缓存结果
            ann['neg_category_ids'] = ann['neg_category_ids'] + rule_neg_cache[pos_cat_id]
            continue

        pos_text = cat_map[pos_cat_id]['name']
        generated_ids = []
        existing_negs_texts = {
            cat_map[nid]['name'] for nid in ann['neg_category_ids'] if nid in cat_map
        }

        # 尝试四类属性依次生成
        attr_slots = [
            (COLORS,        'color'),
            (MATERIALS,     'material'),
            (PATTERNS,      'pattern'),
            (TRANSPARENCIES,'transparency'),
        ]

        for attr_list, attr_type in attr_slots:
            if len(generated_ids) >= n_rule:
                break
            result = _find_attribute_in_text(pos_text, attr_list)
            if result is None:
                continue
            matched_word, true_attr = result

            # 排除与正例属性词相同或歧义的候选
            if attr_type == 'color':
                candidates = _ambiguous_color_candidates(true_attr, COLORS)
                candidates = [c for c in candidates if c != true_attr]
            else:
                candidates = [a for a in attr_list if a != true_attr]
            # 排除 'plain' / 'opaque'（视为无属性）
            candidates = [c for c in candidates if c not in ('plain', 'opaque', 'other')]

            random.shuffle(candidates)
            for new_attr in candidates:
                if len(generated_ids) >= n_rule:
                    break
                new_text = re.sub(
                    r'\b' + re.escape(matched_word) + r'\b',
                    new_attr,
                    pos_text,
                    count=1,
                    flags=re.IGNORECASE,
                )
                if new_text == pos_text or new_text in existing_negs_texts:
                    continue
                # 预先检查 token 长度，超过 16 的直接跳过（与 OWLv2 最大序列长度一致）
                if processor is not None:
                    tok_len = processor(
                        text=[new_text], images=None,
                        return_tensors='pt', padding=True
                    )['input_ids'].shape[1]
                    if tok_len > 16:
                        continue
                # 生成新 category
                new_cat = {
                    'id': next_cat_id,
                    'name': new_text,
                    'synset': '',
                    'synonyms': [],
                    'def': '',
                    'frequency': 'f',
                    'instance_count': 0,
                    'image_count': 0,
                    '_rule_generated': True,   # 标记来源，便于调试
                    '_attr_type': attr_type,
                }
                new_categories.append(new_cat)
                cat_map[next_cat_id] = new_cat
                existing_negs_texts.add(new_text)
                generated_ids.append(next_cat_id)
                next_cat_id += 1

        rule_neg_cache[pos_cat_id] = generated_ids
        ann['neg_category_ids'] = ann['neg_category_ids'] + generated_ids

    data['categories'].extend(new_categories)
    print(f"[expand_negatives_by_rule] Generated {len(new_categories)} rule-based "
          f"negative categories across {len(rule_neg_cache)} unique positive categories.")
    return data


# ---------------------------------------------------------------------------
# 功能三：混合多个 attributes 文件
# ---------------------------------------------------------------------------
def mix_attribute_datasets(data_cfg: dict, training_cfg: dict, data_split: str = 'train',
                           processor=None) -> dict:
    """加载并合并多个标注 JSON 文件，每个子文件均执行负例扩充。
    仅供训练集使用（data_split='train'）；验证/测试集请直接读取单文件。

    查找顺序：
      1. data_cfg['{split}_annotations_files']  -> 列表，多文件模式
      2. data_cfg['{split}_annotations_file']   -> 单个字符串，回退模式

    每个子文件的处理流程：
      expand_negatives_by_sampling -> expand_negatives_by_rule

    合并时：
      - images / categories 按 id 去重（以第一次出现为准）
      - annotations 的 id 加全局偏移，保证唯一
    """
    assert data_split == 'train', (
        f"mix_attribute_datasets 仅用于训练集，当前 data_split='{data_split}'。"
        f"验证/测试集请直接读取单文件。"
    )
    multi_key  = f"{data_split}_annotations_files"
    single_key = f"{data_split}_annotations_file"

    if multi_key in data_cfg:
        files = data_cfg[multi_key]
    elif single_key in data_cfg:
        files = [data_cfg[single_key]]
    else:
        raise KeyError(
            f"data_cfg must contain '{multi_key}' or '{single_key}', got: {list(data_cfg.keys())}"
        )

    merged = {'images': [], 'annotations': [], 'categories': []}
    seen_image_ids: set = set()
    seen_cat_ids: set = set()
    ann_id_offset = 0

    for file_path in files:
        with open(file_path) as f:
            data = json.load(f)

        # 功能一：采样扩充
        data = expand_negatives_by_sampling(data, training_cfg)
        # 功能二：规则替换扩充（传入 processor 以过滤超长生成文本）
        data = expand_negatives_by_rule(data, training_cfg, processor=processor)

        # 合并 images（去重）
        for img in data['images']:
            if img['id'] not in seen_image_ids:
                seen_image_ids.add(img['id'])
                merged['images'].append(img)

        # 合并 categories（去重，以第一次出现为准）
        for cat in data['categories']:
            if cat['id'] not in seen_cat_ids:
                seen_cat_ids.add(cat['id'])
                merged['categories'].append(cat)

        # 合并 annotations（id 加偏移保证全局唯一）
        max_ann_id = max((ann['id'] for ann in data['annotations']), default=0)
        for ann in data['annotations']:
            new_ann = ann.copy()
            new_ann['id'] = ann['id'] + ann_id_offset
            merged['annotations'].append(new_ann)
        ann_id_offset += max_ann_id + 1

    print(
        f"[mix_attribute_datasets] Merged {len(files)} file(s): "
        f"{len(merged['images'])} images, "
        f"{len(merged['annotations'])} annotations, "
        f"{len(merged['categories'])} categories"
    )
    return merged


# ---------------------------------------------------------------------------
# 原版 remove_unprocessable_entries（从 dataset.py 复制，避免 import 循环）
# 修改：processor 作为可选参数传入，避免每次调用都重新加载模型文件
# ---------------------------------------------------------------------------
def remove_unprocessable_entries(data, training_cfg, perform_cleaning=False,
                                 keep_short_vocabularies=False, processor=None):
    if not keep_short_vocabularies:
        data['annotations'] = [
            ann for ann in data['annotations']
            if len(ann['neg_category_ids']) >= training_cfg['n_hardnegatives']
        ]
    cats = {cat['id']: cat for cat in data['categories']}

    # 只在未传入 processor 时才加载（避免 8 个进程各加载多次）
    if processor is None:
        processor = AutoProcessor.from_pretrained(training_cfg['base_model'])
    to_remove = []
    for ann in data['annotations']:
        vocabulary = [
            cats[cat_id]['name']
            for cat_id in [ann['category_id']] + ann['neg_category_ids'][:training_cfg['n_hardnegatives']]
            if cat_id in cats
        ]
        if not vocabulary:
            to_remove.append(ann['id'])
            continue
        tok_len = processor(
            text=vocabulary, images=None, return_tensors='pt', padding=True
        )['input_ids'].shape[1]
        if tok_len > 16:
            to_remove.append(ann['id'])

    data['annotations'] = [ann for ann in data['annotations'] if ann['id'] not in to_remove]

    if perform_cleaning:
        cat_ids = {
            cat_id
            for ann in data['annotations']
            for cat_id in [ann['category_id']] + ann['neg_category_ids']
        }
        data['categories'] = [cat for cat in data['categories'] if cat['id'] in cat_ids]
        imm_ids = {ann['image_id'] for ann in data['annotations']}
        data['images'] = [img for img in data['images'] if img['id'] in imm_ids]

    return data


# ---------------------------------------------------------------------------
# 增强版 OwlDataset
# ---------------------------------------------------------------------------
class OwlDatasetEnhanced(Dataset):
    """与原版 OwlDataset 接口完全相同，但仅对训练集（data_split='train'）
    启用三项数据增强；测试/验证集走与原版完全相同的逻辑。

    方案 A：每轮动态随机负例
      训练集的每条 ann 保存完整的 neg_pool（所有候选负例 id + 规则生成负例 id），
      每次 __getitem__ 被调用时从 neg_pool 中随机抽取 n_hardnegatives 个，
      使得每个 epoch 每张图看到的负例组合都不同。
      验证/测试集不做此处理，vocabulary 固定（与原版行为一致）。
    """

    def __init__(self, image_processor, data_cfg, training_cfg, data_split='train'):
        self.images_dir = data_cfg['images_path']
        self.image_processor = image_processor
        self.data_split = data_split
        self.n_hardnegatives = training_cfg['n_hardnegatives']

        # 只加载一次 processor，复用给 remove_unprocessable_entries
        text_processor = AutoProcessor.from_pretrained(training_cfg['base_model'])

        if data_split == 'train':
            # 仅训练集：多文件混合 + 采样扩充 + 规则替换
            data = mix_attribute_datasets(data_cfg, training_cfg, data_split)
        else:
            # 验证/测试集：与原版 OwlDataset 完全相同，走单文件逻辑
            single_key = f"{data_split}_annotations_file"
            with open(data_cfg[single_key]) as f:
                data = json.load(f)

        # 传入已加载的 processor，不再重复加载
        remove_unprocessable_entries(data, training_cfg, processor=text_processor)
        # 保存 category 名称映射，供 __getitem__ 动态构建 vocabulary
        self.categories = {cat['id']: cat['name'] for cat in data['categories']}
        data = self._convert_to_train_format(data)
        self.data = [{k: v} for k, v in data.items() if len(v)]

    # ------------------------------------------------------------------
    # 以下方法与原版 OwlDataset 完全一致
    # ------------------------------------------------------------------
    def _load_image(self, idx: int):
        url = list(self.data[idx].keys()).pop()
        path = os.path.join(
            self.images_dir,
            url.split('/')[-2],
            url.split('/')[-1][:url.split('/')[-1].find('.jpg') + 4],
        )
        image = Image.open(path).convert('RGB')
        return image, path

    def _load_target(self, idx: int):
        annotations = list(self.data[idx].values()).pop()
        labels, boxes, vocabularies = [], [], []
        for ann in annotations:
            labels.append(ann['label'])
            boxes.append(ann['bbox'])
            if self.data_split == 'train' and 'neg_pool' in ann:
                # 方案 A：每次从完整候选池中随机抽取 n_hardnegatives 个负例
                # 使得每个 epoch 的负例组合都不同
                pool = ann['neg_pool']
                # 过滤掉不在 categories 里的 cid（理论上不应发生，但防御性处理）
                valid_pool = [cid for cid in pool if cid in self.categories]
                k = min(self.n_hardnegatives, len(valid_pool))
                chosen_neg_ids = random.sample(valid_pool, k) if k > 0 else []
                neg_names = [self.categories[cid] for cid in chosen_neg_ids]
                # 正例名字，不允许出现在负例中
                pos_name = ann['pos_cat_name']
                # 若候选池不足 n_hardnegatives，从已有负例中随机重复填充
                # 注意：只从真实负例（非正例）中填充，避免正例混入负例导致 nan
                if len(neg_names) == 0:
                    # 极端情况：neg_pool 完全为空，用占位文本填充
                    # 这种情况下 loss 贡献为 0（正负例相同，triplet margin=0）
                    neg_names = [pos_name] * self.n_hardnegatives
                else:
                    while len(neg_names) < self.n_hardnegatives:
                        neg_names.append(random.choice(neg_names))
                vocab = [pos_name] + neg_names[:self.n_hardnegatives]
                assert len(vocab) == self.n_hardnegatives + 1, \
                    f"vocab length {len(vocab)} != {self.n_hardnegatives + 1}, pos={pos_name}, neg_names={neg_names}"
            else:
                # 验证/测试集：使用固定 vocabulary（与原版一致）
                vocab = ann['vocabulary']
            vocabularies.append(vocab)
        return labels, boxes, vocabularies

    def _convert_to_train_format(self, data):
        new_data = {}
        images     = {x['id']: x for x in data['images']}
        categories = {x['id']: x for x in data['categories']}
        count_vocabularies = {}

        for ann in data['annotations']:
            label = ann['category_id']
            neg_ids = ann['neg_category_ids']

            if self.data_split == 'train':
                # 方案 A：保存完整候选池，vocabulary 留空由 __getitem__ 动态构建
                ann_obj = {
                    'bbox': ann['bbox'],
                    'label': label,
                    'vocabulary': [],           # 占位，训练时不使用
                    'pos_cat_name': categories[label]['name'],
                    'neg_pool': [cid for cid in neg_ids if cid in categories],
                }
            else:
                # 验证/测试集：固定 vocabulary，与原版一致
                vocabulary = [
                    categories[cid]['name']
                    for cid in [label] + neg_ids
                    if cid in categories
                ]
                ann_obj = {'bbox': ann['bbox'], 'label': label, 'vocabulary': vocabulary}

            imm_path = images[ann['image_id']]['coco_url']
            count_vocabularies.setdefault(imm_path, {})
            count_vocabularies[imm_path].setdefault(
                label,
                max(count_vocabularies[imm_path].values()) + 1
                if count_vocabularies[imm_path] else 0,
            )
            imm_path += str(count_vocabularies[imm_path][label])

            new_data.setdefault(imm_path, []).append(ann_obj)

        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, path = self._load_image(idx)
        labels, boxes, vocabularies = self._load_target(idx)

        if isinstance(self.image_processor, OwlViTProcessor):
            w, h = image.size
        else:
            w = h = max(image.size)

        metadata = {
            'width': w,
            'height': h,
            'impath': path,
            'vocabularies': vocabularies,
        }
        image = self.image_processor(
            images=image, return_tensors='pt'
        )['pixel_values'].squeeze(0)

        return image, torch.tensor(labels), torch.tensor(boxes), metadata


# ---------------------------------------------------------------------------
# 增强版 get_dataloaders（替换原版 get_dataloaders）
# ---------------------------------------------------------------------------
def get_dataloaders_enhanced(data_cfg, training_cfg, num_workers=0):
    """与原版 get_dataloaders 接口相同，但使用 OwlDatasetEnhanced。

    LVISDataset 不做改动，直接从原版 dataset.py 导入。
    """
    from src.dataset import LVISDataset, get_dataloaders as _orig_get_dataloaders

    lvis_evaluation = 'lvis_annotations_file' in data_cfg
    image_processor = get_processor(training_cfg['base_model'])

    train_dataset = OwlDatasetEnhanced(image_processor, data_cfg, training_cfg, 'train')
    test_dataset  = OwlDatasetEnhanced(image_processor, data_cfg, training_cfg, 'test')
    lvis_dataset  = None
    if lvis_evaluation:
        from src.dataset import LVISDataset
        lvis_dataset = LVISDataset(image_processor, data_cfg)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_cfg['val_batch_size'],
        shuffle=False,
        num_workers=num_workers,
    )
    lvis_dataloader = (
        DataLoader(lvis_dataset, batch_size=1, shuffle=False, num_workers=0)
        if lvis_evaluation else None
    )

    return train_dataloader, test_dataloader, lvis_dataloader
