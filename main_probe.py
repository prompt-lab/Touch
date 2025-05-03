import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,3'
import torch
from transformers import AutoConfig
from transformers.models.vit.configuration_vit import ViTConfig
from model.linear_probe import TactileProbe
from config_probe import parse_args
import random
import numpy as np
import torch.nn as nn
import sys
from data.downstream_dataset import TAGDataset, OBJ2Dataset, OBJ1Dataset, FeelDataset
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
import timm.optim.optim_factory as optim_factory
from probe_engine import train_one_epoch, evaluate
from peft import get_peft_model, LoraConfig
import argparse
import datetime
import json
import time
from pathlib import Path
import copy

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler


def load_model_from_clip(ckpt, model):
    new_ckpt = {}
    for key,item in ckpt.items():
        if "vision_model" in key and 'position_ids' not in key:
            #new_ckpt[key] = item
            new_ckpt[key.replace("vision_model","touch_model")] = copy.deepcopy(item)
        
        if "visual_projection" in key:
            #new_ckpt[key] = item
            new_ckpt[key.replace("visual","touch")] = copy.deepcopy(item)
    
    for k,v in model.named_parameters():
        if k not in new_ckpt.keys():
            new_ckpt[k] = v
    
    model.load_state_dict(new_ckpt, strict=True)
    
    return model

def load_model_from_mae(ckpt, model, args):
    new_ckpt = {}
    for key,item in ckpt.items():
        if "touch_model" in key or "touch_projection" in key or "sensor_token" in key and "sensor_token_proj" not in key:
            new_ckpt[key] = copy.deepcopy(item)
        if args.use_same_patchemb:
            if "video_patch_embedding" in key:
                new_ckpt[key.replace("video_patch_embedding","touch_model.embeddings.patch_embedding")] = copy.deepcopy(item)
    
    for k,v in model.named_parameters():
        if k not in new_ckpt.keys():
            new_ckpt[k] = v
    
    model.load_state_dict(new_ckpt, strict=True)

    return model

def load_model_from_multi_clip(ckpt, model):
    new_ckpt = {}
    for key,item in ckpt.items():
        if "touch_model" in key or "touch_projection" in key or "sensor_token" in key and "sensor_token_proj" not in key:
            new_ckpt[key.replace('touch_mae_model.','')] = copy.deepcopy(item)
        if args.use_same_patchemb:
            if "video_patch_embedding" in key:
                new_key = key.replace('touch_mae_model.','')
                new_ckpt[new_key.replace("video_patch_embedding","touch_model.embeddings.patch_embedding")] = copy.deepcopy(item)
    
    for k,v in model.named_parameters():
        if k not in new_ckpt.keys():
            new_ckpt[k] = v
    
    model.load_state_dict(new_ckpt, strict=True)

    return model

def load_model(ckpt, model):
    target_modules = ["k_proj", "v_proj", "q_proj", "out_proj"]
    config = LoraConfig(
        r=16,         # 16
        lora_alpha=16,  #  16
        target_modules=target_modules,  # self_attn.out_proj
        lora_dropout=0.1,        # 0.1
        bias="none",
        modules_to_save=[],
    )

    model.touch_model.encoder = get_peft_model(model.touch_model.encoder, config)

    new_ckpt = {}
    for key,item in ckpt.items():
        if "touch_model" in key or "touch_projection" in key:
            if "encoder" in key:
                new_ckpt[key.replace("module.","")] = copy.deepcopy(item)
            else:
                new_ckpt[key.replace("module.","").replace("base_model.model.","")] = copy.deepcopy(item)
    
    for k,v in model.named_parameters():
        if k not in new_ckpt.keys():
            new_ckpt[k] = v
    
    model.load_state_dict(new_ckpt, strict=True)

    return model

def random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.dataset == 'obj2':
        dataset_train = OBJ2Dataset(args, mode = 'train')
        dataset_val = OBJ2Dataset(args, mode = 'test')
    elif args.dataset == 'obj1':
        dataset_train = OBJ1Dataset(args, mode = 'train')
        dataset_val = OBJ1Dataset(args, mode = 'test')
    elif args.dataset == 'feel':
        dataset_train = FeelDataset(args, mode = 'train')
        dataset_val = FeelDataset(args, mode = 'test')
    else:
        dataset_train = TAGDataset(args, mode = 'train')
        dataset_val = TAGDataset(args, mode = 'test')

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers if args.dataset != 'feel' else args.num_workers // 2,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers if args.dataset != 'feel' else args.num_workers // 2,
        pin_memory=True,
        drop_last=False
    )

    config = AutoConfig.from_pretrained('/home/insurance/TLV-Link-main/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/config.json')
    print(args)

    model = TactileProbe(args, config, 1, False, 1)

    if args.load_from_clip:
        load_dir = '../laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/pytorch_model.bin'
        ckpt = torch.load(load_dir, map_location='cpu')
        model = load_model_from_clip(ckpt, model)

    elif args.load_from_align:
        load_dir = args.load_path
        ckpt = torch.load(load_dir, map_location='cpu')['model']
        model = load_model_from_multi_clip(ckpt, model)
    elif args.load_model:
        load_dir = args.load_path
        ckpt = torch.load(load_dir, map_location='cpu')['state_dict']
        model = load_model(ckpt, model)
    else:
        load_dir = args.load_path
        ckpt = torch.load(load_dir, map_location='cpu')['model']
        model = load_model_from_mae(ckpt, model, args)
    # torch.save(model.state_dict(), 'ultratouch_encoder.pth')
    # exit(0)
    print(load_dir)
    if not args.eval:
        model.init_head()

    model.to(device)



    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(model_without_ddp.head.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        test_stats = evaluate(data_loader_val, model, device, args)
        # if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
        if test_stats["acc1"] >= max_accuracy:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=0)
        
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    

if __name__ == "__main__":
    args = parse_args()
    args = args.parse_args()
    main(args)