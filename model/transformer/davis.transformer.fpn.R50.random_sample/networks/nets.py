import os
import sys
import math
import copy
import numpy as np

from semantic_seg import SemanticSegmentor
from .norm import trunc_normal_

import torch
from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init

#from inplace_abn import InPlaceABNSync
from apex.parallel import SyncBatchNorm as InPlaceABNSync

def downsample_by_channel(features):
    features0 = features[:, :,  ::2,  ::2]
    features1 = features[:, :, 1::2,  ::2]
    features2 = features[:, :, 1::2, 1::2]
    features3 = features[:, :,  ::2, 1::2]
    nfeatures = torch.cat([features0, features1, features3, features2], axis=1)
    return nfeatures

class AnchorDiffNet(nn.Module):
    def __init__(self, cfg, embedding=256, batch_mode='sync'):
        super(AnchorDiffNet, self).__init__()

        self.features = SemanticSegmentor(cfg)

        #self.pos_enc = PositionEmbeddingSine(num_pos_feats=embedding//2)
        self.pos_enc = PositionEmbeddingLearned(num_pos_feats=embedding//2)
        transformer_layer = TransformerEncoderLayer(d_model=embedding, nhead=4, dim_feedforward=embedding)
        self.inter_transformer = TransformerEncoder(transformer_layer, num_layers=1)
        self.intra_transformer = TransformerEncoder(transformer_layer, num_layers=1)
        #self.conv0 = nn.Conv2d(16*embedding, embedding, kernel_size=3, stride=1, padding=1)
        #self.bn0 = nn.GroupNorm(32, embedding)
        #self.conv1 = nn.Conv2d(16*embedding, embedding, kernel_size=3, stride=1, padding=1)
        #self.bn1 = nn.GroupNorm(32, embedding)
        #self.relu = nn.ReLU()

        self.head = AnchorHead(embedding=embedding)

        #weight_init.c2_xavier_fill(self.conv0)
        #weight_init.c2_xavier_fill(self.conv1)

    def forward(self, reference, current):
        ref_features, _ = self.features(reference)
        cur_features, cur_babn_features = self.features(current)

        ref_features = F.interpolate(ref_features, scale_factor=.5, mode='bilinear', align_corners=False)
        cur_features = F.interpolate(cur_features, scale_factor=.5, mode='bilinear', align_corners=False)
        #ref_features = downsample_by_channel(downsample_by_channel(ref_features))
        #ref_features = self.conv0(ref_features)
        #ref_features = self.bn0(ref_features)
        #ref_features = self.relu(ref_features)
        #cur_features = downsample_by_channel(downsample_by_channel(cur_features))
        #cur_features = self.conv1(cur_features)
        #cur_features = self.bn1(cur_features)
        #cur_features = self.relu(cur_features)

        pos_embed = self.pos_enc(cur_features)

        bs, c, h, w = cur_features.shape
        cur_features = cur_features.flatten(2).permute(2, 0, 1)
        ref_features = ref_features.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        feats_0 = self.inter_transformer(cur_features, ref_features, pos_embed)
        feats_1 = self.intra_transformer(cur_features, cur_features, pos_embed)
        feats_0 = feats_0.permute(1, 2, 0).view(bs, c, h, w)
        feats_1 = feats_1.permute(1, 2, 0).view(bs, c, h, w)
        cur_features = cur_features.permute(1, 2, 0).view(bs, c, h, w)

        feats = torch.cat([feats_0, feats_1, cur_features], dim=1)

        pred = self.head(feats)

        return pred

class AnchorHead(nn.Module):
    def __init__(self, embedding=256, batch_mode='sync'):
        super(AnchorHead, self).__init__()

        self.cls_conv0 = nn.Conv2d(3*embedding, embedding, kernel_size=1, stride=1, padding=0)
        self.bn0 = InPlaceABNSync(embedding)
        self.relu = nn.LeakyReLU(0.01)
        self.drop = nn.Dropout2d(0.10)
        self.dconv0 = nn.ConvTranspose2d(embedding, embedding, 3, stride=2, padding=1)
        self.bn1 = InPlaceABNSync(embedding)
        self.dconv1 = nn.ConvTranspose2d(embedding, embedding, 3, stride=2, padding=1)
        self.bn2 = InPlaceABNSync(embedding)
        self.dconv2 = nn.ConvTranspose2d(embedding, embedding, 3, stride=2, padding=1)
        self.bn3 = InPlaceABNSync(embedding)
        self.cls_conv1 = nn.Conv2d(embedding, 1, kernel_size=1, stride=1, padding=0)

        #self.skip_conv0 = nn.Conv2d(2*embedding, embedding, 3, stride=1, padding=1)
        #self.skip_conv1 = nn.Conv2d(2*embedding, embedding, 3, stride=1, padding=1)
        #self.skip_conv2 = nn.Conv2d(2*embedding, embedding, 3, stride=1, padding=1)

        weight_init.c2_xavier_fill(self.cls_conv0)
        weight_init.c2_xavier_fill(self.cls_conv1)
        weight_init.c2_xavier_fill(self.dconv0)
        weight_init.c2_xavier_fill(self.dconv1)
        weight_init.c2_xavier_fill(self.dconv2)
        #weight_init.c2_xavier_fill(self.skip_conv0)
        #weight_init.c2_xavier_fill(self.skip_conv1)
        #weight_init.c2_xavier_fill(self.skip_conv2)

    def forward(self, pred):
        pred = self.cls_conv0(pred)
        pred = self.bn0(pred)
        pred = self.relu(pred)
        #pred += self.skip_conv0(babn_features[2])
        #pred = self.drop(pred)
        pred = self.dconv0(pred)
        pred = self.bn1(pred)
        pred = self.relu(pred)
        #pred += self.skip_conv1(babn_features[1])
        pred = self.dconv1(pred)
        pred = self.bn2(pred)
        pred = self.relu(pred)
        #pred += self.skip_conv2(babn_features[0])
        pred = self.dconv2(pred)
        pred = self.bn3(pred)
        pred = self.relu(pred)
        pred = self.cls_conv1(pred)

        return pred

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        not_mask = x.new_ones((x.size()[0], x.size()[2], x.size()[3]))
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(200, num_pos_feats)
        self.col_embed = nn.Embedding(200, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        #nn.init.uniform_(self.row_embed.weight)
        #nn.init.uniform_(self.col_embed.weight)
        trunc_normal_(self.row_embed.weight, std=.02)
        trunc_normal_(self.col_embed.weight, std=.02)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, cur, ref, pos):
        output = cur

        for layer in self.layers:
            output = layer(output, ref, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        #self._reset_parameters()
        self.apply(self._init_weights)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, cur, ref, pos):
        q = self.with_pos_embed(ref, pos)
        k = self.with_pos_embed(cur, pos)
        src2 = self.self_attn(q, k, value=cur)[0]
        cur = cur + self.dropout1(src2)
        cur = self.norm1(cur)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(cur))))
        cur = cur + self.dropout2(src2)
        cur = self.norm2(cur)
        return cur

    def forward_pre(self, cur, ref, pos):
        src2 = self.norm1(cur)
        q = self.with_pos_embed(ref, pos)
        k = self.with_pos_embed(cur, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        cur = cur + self.dropout1(src2)
        src2 = self.norm2(cur)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        cur = cur + self.dropout2(src2)
        return cur

    def forward(self, cur, ref, pos):
        if self.normalize_before:
            return self.forward_pre(cur, ref, pos)
        return self.forward_post(cur, ref, pos)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


