import argparse
import json
import os
import shutil
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml

from PIL import Image
from transformers import AutoProcessor, OwlViTForObjectDetection, Owlv2ForObjectDetection

from copy import deepcopy
from eval.fgovd.evaluate_map import read_json
from src.losses import PushPullLoss
from src.dataset import remove_unprocessable_entries, get_dataloaders, keep_only_rare
from src.models import PostProcess, load_model
from src.ddp_train_util import train, validate, validate_lvis, validate_lvis_hf, get_ids_per_frequencies
from src.util import BoxUtil, GeneralLossAccumulator, ProgressFormatter, ModelUtil, get_processor, process_single_string
from torch.utils.tensorboard import SummaryWriter

def get_training_config(config_path):
    with open(config_path, "r") as stream:
        data = yaml.safe_load(stream)
        data['training']['n_accumulation_steps'] = data['training'].get('n_accumulation_steps', 1)
        data['training']['self_distillation'] = data['training'].get('self_distillation', False)
        return data["training"]
    
def get_data_config(config_path):
    with open(config_path, "r") as stream:
        data = yaml.safe_load(stream)
        return data["data"]

def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def get_distributed_dataloaders(data_cfg, training_cfg, rank, world_size):
    """获取分布式数据加载器"""
    train_dataloader, val_dataloader, lvis_dataloader = get_dataloaders(data_cfg, training_cfg,num_workers=4)
    
    # 为训练集创建分布式采样器
    if train_dataloader is not None:
        train_sampler = DistributedSampler(
            train_dataloader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataloader.dataset,
            batch_size=train_dataloader.batch_size,
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )
    
    # 为验证集创建分布式采样器
    if val_dataloader is not None:
        val_sampler = DistributedSampler(
            val_dataloader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataloader.dataset,
            batch_size=val_dataloader.batch_size,
            sampler=val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=False
        )
    
    return train_dataloader, val_dataloader, lvis_dataloader

def main_worker(rank, world_size, args):
    """每个GPU上运行的工作函数"""
    setup(rank, world_size)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # 设置随机种子（确保所有进程有相同的种子）
    SEED = 123
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # 只在主进程上创建TensorBoard写入器
    if rank == 0:
        model_name = args.out.split('/')[-1]
        writer = SummaryWriter(f'runs/{model_name}_')
    else:
        writer = None
    
    # 加载配置
    training_cfg = get_training_config(args.config)
    data_cfg = get_data_config(args.config)
    
    # 获取数据加载器
    train_dataloader, val_dataloader, lvis_dataloader = get_distributed_dataloaders(
        data_cfg, training_cfg, rank, world_size
    )
    
    # 加载模型
    model = load_model(device, training_cfg['base_model'])
    
    # 使用DDP包装模型
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    processor = get_processor(training_cfg['base_model'])
    postprocess = PostProcess(confidence_threshold=0, iou_threshold=0.5)
    
    lvis_evaluation = 'lvis_annotations_file' in data_cfg
    
    print("Loading Validation dataset")
    val_data = read_json(data_cfg['test_annotations_file'])
    val_data = remove_unprocessable_entries(val_data, training_cfg, perform_cleaning=True)
    
    queries = None
    if (lvis_evaluation or training_cfg['self_distillation']) and rank == 0:
        lvis_data = read_json("lvis_v1_val.json")
        
        if args.do_not_use_hf_lvis_evaluation or training_cfg['self_distillation']:
            with torch.no_grad():
                vocabulary = ['a ' + process_single_string(cat['name']) for cat in lvis_data['categories']]
                inputs = processor(
                    text=[vocabulary],
                    images=Image.new("RGB", (224, 224)),
                    return_tensors="pt",
                    padding=True
                )
                inputs = inputs.to(device)
                if hasattr(model, 'module'):
                    queries = model.module.text_embedder(inputs)
                else:
                    queries = model.text_embedder(inputs)
                queries = queries.unsqueeze(0)
            ids_per_frequencies = get_ids_per_frequencies(lvis_data)
        
        text_query = ['a ' + cat['name'] for cat in lvis_data['categories']]
        
        tmp_base_path = os.path.join('tmp', args.out.split('/')[-1])
        os.makedirs(tmp_base_path, exist_ok=True)
    
    # # 同步queries到所有进程
    # if queries is not None:
    #     queries = [queries.to(device) if queries is not None else None]
    #     dist.broadcast_object_list(queries, src=0)
    #     queries = queries[0]
    
    # 损失函数
    criterion = PushPullLoss(
        training_cfg['n_hardnegatives'] + 1,
        margin=training_cfg['margin'],
        self_distillation_loss=training_cfg.get('self_distillation', 'mse'),
        class_ltype=training_cfg.get('ltype', 'triplet')
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=training_cfg["weight_decay"],
    )
    
    # 只在主进程上创建进度跟踪器
    general_loss = GeneralLossAccumulator()
    progress_summary = ProgressFormatter()
    best_map = 0
    # 训练循环
    for epoch in range(-1, training_cfg["n_epochs"]):
        # 设置数据采样器的epoch
        if train_dataloader is not None and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        
        # # 训练
        model.train()
        train_metrics = train(
            model,
            train_dataloader,
            criterion,
            optimizer,
            general_loss,
            epoch,
            training_cfg['n_accumulation_steps'],
            lvis_query_embeds=queries,
            writer=writer,
            amp=False,
        )#if epoch > -1 else {'loss_triplet': 0, 'loss_bg': 0, 'loss_bbox': 0, 'loss_giou': 0}
       
        if rank==0:
            print('start eval')
        # 评估
        model.eval()
        if epoch % 5==0:
            val_metrics = validate(model, val_dataloader, deepcopy(val_data), epoch, writer,n_hardnegatives=training_cfg['n_hardnegatives'])
            print(val_metrics)
        if rank == 0:            
            # 打印训练摘要
            progress_summary.update(epoch, train_metrics, val_metrics)
            progress_summary.print()
            
            # LVIS评估
            if lvis_evaluation:
                print("Evaluating LVIS...")
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
                
                print("LVIS " + str(lvis_metrics['map']))
            
            # 保存最佳模型
            if val_metrics['map'] > best_map and epoch > -1:
                print("Best validation mAP, saving the weights...")
                best_map = val_metrics['map']
                if hasattr(model, 'module'):
                    ModelUtil.create_base_model(model.module, training_cfg['base_model']).save_pretrained(args.out)
                else:
                    ModelUtil.create_base_model(model, training_cfg['base_model']).save_pretrained(args.out)
            
            # 保存当前epoch模型
            print(f"Saving model at epoch {epoch}...")
            out_path = args.out + f'_epoch{epoch}'
            if hasattr(model, 'module'):
                ModelUtil.create_base_model(model.module, training_cfg['base_model']).save_pretrained(out_path)
            else:
                ModelUtil.create_base_model(model, training_cfg['base_model']).save_pretrained(out_path)
        
        # 同步所有进程
        dist.barrier()
    
    # 清理
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/base_config.yaml", help="Training configuration file to load")
    parser.add_argument('--do_not_use_hf_lvis_evaluation', action='store_true', help="If setted, the evaluation on LVIS will be performed without the HuggingFace interface and by using the model as it is trained.") 
    parser.add_argument('--out', type=str, default="result", help="Base OWL model to use")
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    args = parser.parse_args()
    
    # 启动多进程训练
    world_size = args.world_size
    if world_size == 1:
        main_worker(0,1,args)
    else:
        torch.multiprocessing.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )

if __name__ == "__main__":
    main()