from __future__ import division
import os.path as osp
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn import sigmoid_focal_loss_star_jit

from config import config

from dataloader import get_train_loader
from datasets.davis import DAVIS
from networks.nets import AnchorDiffNet
from backbone import NaiveSyncBatchNorm
from fpn_config import get_cfg
from losses import *

from utils.init_func import build_optimizer_adnet, init_weights_adnet
from utils.pyt_utils import all_reduce_tensor
from engine.lr_policy import PolyLR
from engine.logger import get_logger
from engine.engine import Engine

try:
    from apex.parallel import DistributedDataParallel
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

logger = get_logger()

torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    if engine.distributed:
        torch.cuda.set_device(engine.local_rank)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, DAVIS)

    # config network and criterion
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    #criterion = BinaryDiceLoss()
    #criterion = sigmoid_focal_loss_star_jit

    if engine.distributed:
        logger.info('Use the Multi-Process-SyncBatchNorm')

    cfg = get_cfg()
    cfg.merge_from_file('./fpn_config/semantic_R_50_FPN_1x.yaml')
    model = AnchorDiffNet(cfg, embedding=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM, batch_mode='sync')

    pretrained_weights = torch.load(config.pretrained_model)
    renamed_pretrained_weights = dict()
    for key in pretrained_weights:
        #if key == 'sem_seg_head.predictor.weight' or key == 'sem_seg_head.predictor.bias':
        #    continue
        if key[:14] == 'sem_seg_head.p':
            continue
        renamed_pretrained_weights['features.' + key] = pretrained_weights[key]
    names = model.load_state_dict(renamed_pretrained_weights, strict=False)
    print(names)

    # group weight and config optimizer
    base_lr = config.lr

    params_list = []
    params_list = build_optimizer_adnet(config, params_list, model.features.backbone, base_lr)
    params_list = build_optimizer_adnet(config, params_list, model.features.sem_seg_head, base_lr)
    params_list = build_optimizer_adnet(config, params_list, model.inter_transformer, base_lr)
    params_list = build_optimizer_adnet(config, params_list, model.intra_transformer, base_lr)
    params_list = build_optimizer_adnet(config, params_list, model.head, base_lr)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)
    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum)
    #optimizer = torch.optim.AdamW(params_list,
    #                            lr=base_lr,
    #                            weight_decay=1e-4,
    #                            )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if engine.distributed:
        model = DistributedDataParallel(model)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    writer = SummaryWriter(log_dir=osp.join("logs", 'davis.fpn.resnet'
                                                    + str(cfg.MODEL.RESNETS.DEPTH)
                                                    + '.'
                                                    + str(base_lr)
                                            )
                          )

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            ref_imgs = minibatch['ref_img']
            cur_imgs = minibatch['cur_img']
            cur_masks = minibatch['cur_mask']

            ref_imgs = ref_imgs.cuda(non_blocking=True)
            cur_imgs = cur_imgs.cuda(non_blocking=True)
            cur_masks = cur_masks.cuda(non_blocking=True)

            preds = model(ref_imgs, cur_imgs)
            preds = F.interpolate(preds, (cur_masks.size()[2], cur_masks.size()[3]), mode='bilinear', align_corners=False)
            loss = criterion(preds, cur_masks)
            #loss = lovasz_hinge(preds.squeeze(), cur_masks.squeeze())
            loss = bootstrapped_ce_loss(loss)
            #loss += 1. * dice_loss(preds[:, 0].flatten(1), cur_masks.flatten(1))

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss,
                                                world_size=engine.world_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

            writer.add_scalar("train/loss", scalar_value=reduce_loss.item(), global_step=epoch*config.niters_per_epoch+idx)
            writer.add_scalar("train/lr", scalar_value=lr, global_step=epoch*config.niters_per_epoch+idx)

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % reduce_loss.item()

            pbar.set_description(print_str, refresh=False)

        if (epoch >= config.nepochs - 20) or (
                epoch % config.snapshot_iter == 0):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
    writer.close()
