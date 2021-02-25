import cv2
import numpy as np

import torch
from torch.utils import data

from config import config
from utils.img_utils import generate_random_common_bbox, random_scale_crop, random_hflip_adnet, random_rotation_adnet, normalize

class TrainPre(object):
    def __init__(self, img_mean, img_std, target_size):
        self.img_mean = img_mean
        self.img_std = img_std
        self.target_size = target_size

    def __call__(self, ref_img, cur_img, ref_mask, cur_mask):

        common_bbox = generate_random_common_bbox(ref_mask, cur_mask)
        ref_img = ref_img[common_bbox[1]:common_bbox[3], common_bbox[0]:common_bbox[2], :]
        cur_img = cur_img[common_bbox[1]:common_bbox[3], common_bbox[0]:common_bbox[2], :]
        cur_mask = cur_mask[common_bbox[1]:common_bbox[3], common_bbox[0]:common_bbox[2]]
        #ref_img, cur_img, ref_mask, cur_mask = random_scale_crop(ref_img, cur_img, ref_mask, cur_mask)

        ref_img = cv2.resize(ref_img, (config.image_width, config.image_height))
        cur_img = cv2.resize(cur_img, (config.image_width, config.image_height))
        cur_mask = cv2.resize(cur_mask, (config.image_width, config.image_height), interpolation=cv2.INTER_NEAREST)

        ref_img, cur_img, cur_mask = random_hflip_adnet(ref_img, cur_img, cur_mask)
        ref_img = normalize(ref_img, self.img_mean, self.img_std)
        cur_img = normalize(cur_img, self.img_mean, self.img_std)
        ref_img, cur_img, cur_mask = random_rotation_adnet(ref_img, cur_img, cur_mask)

        ref_img = ref_img.transpose(2, 0, 1)
        cur_img = cur_img.transpose(2, 0, 1)
        cur_mask = np.expand_dims(cur_mask, 0)

        extra_dict = None

        return ref_img, cur_img, cur_mask, extra_dict


def get_train_loader(engine, dataset):
    data_setting = {'data_train_source': config.data_root_path}
    train_preprocess = TrainPre(config.image_mean, config.image_std,
                                config.target_size)

    train_dataset = dataset(data_setting, "train", 0.2, train_preprocess,
                            config.niters_per_epoch * config.batch_size)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler
