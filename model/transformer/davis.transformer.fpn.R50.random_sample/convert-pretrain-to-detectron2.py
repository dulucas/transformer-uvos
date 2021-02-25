#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    obj = obj["model"]

    newmodel = {}
    for k, v in obj.items():
        #if not k.startswith("module.encoder_q."):
        #    continue
        old_k = k
        k = k.replace("module.encoder_q.", "")
        if "layer" not in k:
            num = k.find('.')
            k = k[:num] + ".stem" + k[num:]
            #k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        k = k.replace("backbone.0", "backbone.bottom_up")
        k = k.replace("detr.", "")
        k = k.replace("body.", "")
        print(old_k, "->", k)
        newmodel[k] = v

    res = newmodel

    torch.save(res, sys.argv[2])
