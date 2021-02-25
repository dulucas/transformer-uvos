#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : BaseDataset.py

import os
import time
import json
import cv2
import torch
import pickle as pkl
import numpy as np
from PIL import Image

import torch.utils.data as data

class DAVIS(data.Dataset):
    def __init__(self, setting, split_name, thre, preprocess=None,
                 file_length=None):
        super(DAVIS, self).__init__()
        self._split_name = split_name
        self.data_pairs = self._load_data_file(setting['data_train_source'])
        self._file_length = len(self.data_pairs)
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self.data_pairs)

    def __getitem__(self, index):
        name = self.data_pairs[index]

        ref_img, cur_img, ref_mask, cur_mask = self._fetch_data(name)
        if self.preprocess is not None:
            ref_img, cur_img, cur_mask, extra_dict = self.preprocess(ref_img, cur_img, ref_mask, cur_mask)

        if self._split_name == 'train':
            ref_img = torch.from_numpy(np.ascontiguousarray(ref_img)).float()
            cur_img = torch.from_numpy(np.ascontiguousarray(cur_img)).float()
            cur_mask = torch.from_numpy(np.ascontiguousarray(cur_mask)).float()

            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].float()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(ref_img=ref_img, cur_img=cur_img, cur_mask=cur_mask)

        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, line):
        ref_img = line.split('\t')[0]
        cur_img =  line.split('\t')[1]

        ref_mask = ref_img.replace('JPEGImages', 'Annotations').replace('jpg', 'png')
        cur_mask = cur_img.replace('JPEGImages', 'Annotations').replace('jpg', 'png')

        ref_mask = Image.open(ref_mask)
        ref_mask = np.atleast_3d(ref_mask)[...,0]
        ref_mask = ref_mask.copy()
        ref_mask[ref_mask > 0] = 1
        cur_mask = Image.open(cur_mask)
        cur_mask = np.atleast_3d(cur_mask)[...,0]
        cur_mask = cur_mask.copy()
        cur_mask[cur_mask > 0] = 1

        while cur_mask.sum() == 0 or ref_mask.sum() == 0:
            index = np.random.randint(len(self.data_pairs))
            line = self.data_pairs[index]
            ref_img = line.split('\t')[0]
            cur_img =  line.split('\t')[1]

            ref_mask = ref_img.replace('JPEGImages', 'Annotations').replace('jpg', 'png')
            cur_mask = cur_img.replace('JPEGImages', 'Annotations').replace('jpg', 'png')

            ref_mask = Image.open(ref_mask)
            ref_mask = np.atleast_3d(ref_mask)[...,0]
            ref_mask = ref_mask.copy()
            ref_mask[ref_mask > 0] = 1
            cur_mask = Image.open(cur_mask)
            cur_mask = np.atleast_3d(cur_mask)[...,0]
            cur_mask = cur_mask.copy()
            cur_mask[cur_mask > 0] = 1

        ref_img = cv2.imread(ref_img)[:,:,::-1]
        cur_img = cv2.imread(cur_img)[:,:,::-1]

        return ref_img, cur_img, ref_mask, cur_mask

    def _load_data_file(self, data_source_path):
        data_file = open(data_source_path, 'r').readlines()
        data_file = [i.strip() for i in data_file]
        return data_file

    def get_length(self):
        return self.__len__()


if __name__ == "__main__":
    data_setting = {'data_train_source': '/home/duy/phd/lucasdu/duy/coco_train2017_refine_training/info.txt'}
    bd = COCO(data_setting, 'train', 0.3)
