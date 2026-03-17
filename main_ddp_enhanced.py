"""main_ddp_enhanced.py

与 main_ddp.py 完全相同，唯一差异：
  将 get_dataloaders 替换为 src.dataset_enhanced.get_dataloaders_enhanced，
  从而启用三项数据增强功能（负例采样扩充 / 规则替换扩充 / 多文件混合）。

默认配置文件：configs/base-v2-enhanced.yaml

用法示例（单卡）：
    conda activate ov202503
    python main_ddp_enhanced.py --config configs/base-v2-enhanced.yaml --out result_enhanced

用法示例（多卡，2 GPU）：
    conda activate ov202503
    python main_ddp_enhanced.py --config configs/base-v2-enhanced.yaml --out result_enhanced --world_size 2
"""

import argparse
import json
import os
import pickle
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from copy import deepcopy
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, OwlViTForObjectDetection, Owlv2ForObjectDetection
import yaml

from eval.fgovd.evaluate_map import read_json
from src.dataset import remove_unprocessable_entries          # 原版清洗函数
from src.dataset_enhanced import get_dataloaders_enhanced     # ← 核心替换
from src.losses import PushPullLoss
from src.models import PostProcess, load_model
from src.ddp_train_util import (
    train, validate, validate_lvis, validate_lvis_hf, get_ids_per_frequencies
)
from src.util import (
    BoxUtil, GeneralLossAccumulator, ProgressFormatter,
    ModelUtil, get_processor, process_single_string
)


# ---------------------------------------------------------------------------
# 配置读取（与 main_ddp.py 相同）
# ---------------------------------------------------------------------------
def get_training_config(config_path):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    data['training']['n_accumulation_steps'] = data['training'].get('n_accumulation_steps', 1)
    data['training']['self_distillation']    = data['training'].get('self_distillation', False)
    return data['training']


def get_data_config(config_path):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data['data']


# ---------------------------------------------------------------------------
# DDP 工具函数（与 main_ddp.py 相同）
# ---------------------------------------------------------------------------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_distributed_dataloaders(data_cfg, training_cfg, rank, world_size, args):
    """从主进程预构建的缓存加载数据集，各子进程共享同一份数据，避免重复构建。"""
    import pickle
    from src.dataset_enhanced import OwlDatasetEnhanced
    from src.util import get_processor as _get_proc

    image_processor = _get_proc(training_cfg['base_model'])

    # 从缓存文件恢复预处理好的数据
    with open(args.train_cache, 'rb') as f:
        train_data = pickle.load(f)
    with open(args.test_cache, 'rb') as f:
        test_data = pickle.load(f)

    # 用缓存数据直接构建 Dataset（跳过耗时的数据加载和预处理）
    train_dataset = object.__new__(OwlDatasetEnhanced)
    train_dataset.images_dir      = data_cfg['images_path']
    train_dataset.image_processor = image_processor
    train_dataset.data_split      = 'train'
    train_dataset.n_hardnegatives = training_cfg['n_hardnegatives']
    train_dataset.categories      = {cat['id']: cat['name'] for cat in train_data['categories']}
    formatted = OwlDatasetEnhanced._convert_to_train_format(train_dataset, train_data)
    train_dataset.data = [{k: v} for k, v in formatted.items() if len(v)]

    test_dataset = object.__new__(OwlDatasetEnhanced)
    test_dataset.images_dir      = data_cfg['images_path']
    test_dataset.image_processor = image_processor
    test_dataset.data_split      = 'test'
    test_dataset.n_hardnegatives = 10 #training_cfg['n_hardnegatives']  # 验证集原始负例数固定为 10
    test_dataset.categories      = {cat['id']: cat['name'] for cat in test_data['categories']}
    formatted = OwlDatasetEnhanced._convert_to_train_format(test_dataset, test_data)
    test_dataset.data = [{k: v} for k, v in formatted.items() if len(v)]

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=training_cfg['batch_size'],
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=training_cfg['val_batch_size'],
        sampler=val_sampler, num_workers=4, pin_memory=True, drop_last=False,
    )

    lvis_dl = None
    if 'lvis_annotations_file' in data_cfg:
        from src.dataset import LVISDataset
        lvis_dataset = LVISDataset(image_processor, data_cfg)
        lvis_dl = torch.utils.data.DataLoader(
            lvis_dataset, batch_size=1, shuffle=False, num_workers=0
        )

    return train_dl, val_dl, lvis_dl


# ---------------------------------------------------------------------------
# 每个 GPU 的工作函数（与 main_ddp.py 相同，仅数据加载函数不同）
# ---------------------------------------------------------------------------
def main_worker(rank, world_size, args):
    setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    SEED = 123
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    writer = SummaryWriter(f'runs/{args.out.split("/")[-1]}_enhanced_') if rank == 0 else None

    training_cfg = get_training_config(args.config)
    data_cfg     = get_data_config(args.config)

    train_dataloader, val_dataloader, lvis_dataloader = get_distributed_dataloaders(
        data_cfg, training_cfg, rank, world_size, args
    )

    model     = load_model(device, training_cfg['base_model'])
    model     = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    processor = get_processor(training_cfg['base_model'])
    postprocess = PostProcess(confidence_threshold=0, iou_threshold=0.5)

    lvis_evaluation = 'lvis_annotations_file' in data_cfg

    print('Loading Validation dataset')
    val_data = read_json(data_cfg['test_annotations_file'])
    # 验证集负例数固定为 10，用 keep_short_vocabularies=True 跳过数量过滤
    # 只做 token 长度清洗（perform_cleaning=True 同时清理无效 image/category）
    val_training_cfg = dict(training_cfg)
    val_training_cfg['n_hardnegatives'] = 10
    val_data = remove_unprocessable_entries(val_data, val_training_cfg, perform_cleaning=True)
    #val_data = remove_unprocessable_entries(val_data, training_cfg, perform_cleaning=True)

    queries = None
    if (lvis_evaluation or training_cfg['self_distillation']) and rank == 0:
        lvis_data = read_json('lvis_v1_val.json')

        if args.do_not_use_hf_lvis_evaluation or training_cfg['self_distillation']:
            with torch.no_grad():
                vocabulary = ['a ' + process_single_string(cat['name'])
                              for cat in lvis_data['categories']]
                inputs = processor(
                    text=[vocabulary],
                    images=Image.new('RGB', (224, 224)),
                    return_tensors='pt',
                    padding=True,
                ).to(device)
                queries = (
                    model.module.text_embedder(inputs)
                    if hasattr(model, 'module')
                    else model.text_embedder(inputs)
                ).unsqueeze(0)
            ids_per_frequencies = get_ids_per_frequencies(lvis_data)

        text_query = ['a ' + cat['name'] for cat in lvis_data['categories']]
        tmp_base_path = os.path.join('tmp', args.out.split('/')[-1])
        os.makedirs(tmp_base_path, exist_ok=True)

    criterion = PushPullLoss(
        training_cfg['n_hardnegatives'] + 1,
        margin=training_cfg['margin'],
        self_distillation_loss=training_cfg.get('self_distillation', 'mse'),
        class_ltype=training_cfg.get('ltype', 'triplet'),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg['learning_rate']),
        weight_decay=training_cfg['weight_decay'],
    )

    general_loss    = GeneralLossAccumulator()
    progress_summary = ProgressFormatter()
    best_map = 0

    for epoch in range(-1, training_cfg['n_epochs']):
        if train_dataloader is not None and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        train_metrics = train(
            model, train_dataloader, criterion, optimizer,
            general_loss, epoch, training_cfg['n_accumulation_steps'],
            lvis_query_embeds=queries, writer=writer, amp=args.amp,
        )

        if rank == 0:
            print('start eval')
        model.eval()
        val_metrics = validate(
            model, val_dataloader, deepcopy(val_data), epoch, writer,
            n_hardnegatives=10 #training_cfg['n_hardnegatives']  # 验证集原始负例数固定为 10，与训练集 n_hardnegatives 无关
        )
        print(val_metrics)

        if rank == 0:
            progress_summary.update(epoch, train_metrics, val_metrics)
            progress_summary.print()

            if lvis_evaluation:
                print('Evaluating LVIS...')
                if args.do_not_use_hf_lvis_evaluation:
                    lvis_metrics = validate_lvis(
                        model, lvis_dataloader, queries, postprocess,
                        ids_per_frequencies, epoch, writer
                    )
                else:
                    tmp_model_path = os.path.join(tmp_base_path, 'model_tmp')
                    ModelUtil.create_base_model(model.module, training_cfg['base_model']).save_pretrained(tmp_model_path)
                    if 'owlv2' in training_cfg['base_model']:
                        hf_model = Owlv2ForObjectDetection.from_pretrained(tmp_model_path)
                    elif 'owlvit' in training_cfg['base_model']:
                        hf_model = OwlViTForObjectDetection.from_pretrained(tmp_model_path)
                    hf_model.to(device)
                    lvis_metrics = validate_lvis_hf(
                        hf_model, processor, data_cfg, training_cfg,
                        lvis_data, text_query, tmp_base_path, epoch, writer
                    )
                print('LVIS ' + str(lvis_metrics['map']))

            if val_metrics['map'] > best_map and epoch > -1:
                print('Best validation mAP, saving the weights...')
                best_map = val_metrics['map']
                m = model.module if hasattr(model, 'module') else model
                ModelUtil.create_base_model(m, training_cfg['base_model']).save_pretrained(args.out)

            print(f'Saving model at epoch {epoch}...')
            out_path = args.out + f'_epoch{epoch}'
            m = model.module if hasattr(model, 'module') else model
            ModelUtil.create_base_model(m, training_cfg['base_model']).save_pretrained(out_path)

        dist.barrier()

    cleanup()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/base-v2-enhanced.yaml',
                        help='Enhanced training config file')
    parser.add_argument('--do_not_use_hf_lvis_evaluation', action='store_true')
    parser.add_argument('--out', type=str, default='result_enhanced')
    parser.add_argument('--world_size', type=int,
                        default=torch.cuda.device_count())
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 在主进程预先构建训练/验证数据集，缓存到临时文件
    # 各子进程直接加载缓存，避免 8 个进程重复执行耗时的数据构建
    # ------------------------------------------------------------------
    training_cfg = get_training_config(args.config)
    data_cfg     = get_data_config(args.config)

    from src.util import get_processor as _get_processor
    from src.dataset_enhanced import (
        mix_attribute_datasets, remove_unprocessable_entries,
        OwlDatasetEnhanced
    )
    from transformers import AutoProcessor as _AutoProcessor

    print("[main] Pre-building datasets in main process...")
    image_processor = _get_processor(training_cfg['base_model'])
    text_processor  = _AutoProcessor.from_pretrained(training_cfg['base_model'])

    # 构建训练集
    train_data = mix_attribute_datasets(data_cfg, training_cfg, 'train',
                                        processor=text_processor)
    remove_unprocessable_entries(train_data, training_cfg, processor=text_processor)

    # 构建验证集
    with open(data_cfg['test_annotations_file']) as f:
        test_data = json.load(f)
    val_training_cfg = dict(training_cfg)
    val_training_cfg['n_hardnegatives'] = 10
    remove_unprocessable_entries(test_data, val_training_cfg, processor=text_processor)

    # 序列化到临时文件，供子进程读取
    cache_dir = tempfile.mkdtemp(prefix='fgovd_cache_')
    train_cache = os.path.join(cache_dir, 'train_data.pkl')
    test_cache  = os.path.join(cache_dir, 'test_data.pkl')
    with open(train_cache, 'wb') as f:
        pickle.dump(train_data, f)
    with open(test_cache, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"[main] Dataset cache written to {cache_dir}")

    # 把缓存路径传给子进程
    args.train_cache = train_cache
    args.test_cache  = test_cache

    world_size = args.world_size
    if world_size == 1:
        main_worker(0, 1, args)
    else:
        # 使用 forkserver 避免 spawn 序列化大型 Dataset 对象时的 pickle 截断问题
        torch.multiprocessing.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )


if __name__ == '__main__':
    main()
