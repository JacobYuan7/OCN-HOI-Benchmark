# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from scipy.optimize import linear_sum_assignment
import time

import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from torchvision.ops import RoIAlign
import math
from .transformer import TransformerEncoder, TransformerEncoderLayer

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def xyxy_to_feature_scale(bboxes: torch.Tensor, features: NestedTensor):
    b, nq, ncoor = bboxes.shape # xy xy
    for i in range(b):
        ft_mask = features[-1].mask[i] # [ft_w, ft_h]
        # print(ft_mask.shape)
        valid_w = torch.nonzero(ft_mask[:,0], as_tuple = True)[0] 
        # the first 0 is the index for the tuple and the second 0 is the index for the first nonzero term
        valid_h = torch.nonzero(ft_mask[0,:], as_tuple = True)[0]
        valid_w = ft_mask[:,0].shape[0] if valid_w.shape[0] == 0 else valid_w[0]
        valid_h = ft_mask[0,:].shape[0] if valid_h.shape[0] == 0 else valid_h[0]

        # print(ft_mask.shape)
        # print(valid_w)
        # print(valid_h)
        # print(print(~ft_mask[:valid_w-ft_mask.shape[0]-1, :valid_h - ft_mask.shape[1] - 1]))

        # print(bboxes[i, :, 0].shape)
        bboxes[i, :, 0] = bboxes[i, :, 0] * valid_w
        bboxes[i, :, 1] = bboxes[i, :, 1] * valid_h
        bboxes[i, :, 2] = bboxes[i, :, 2] * valid_w
        bboxes[i, :, 3] = bboxes[i, :, 3] * valid_h

    bboxes_roi_align = torch.stack((bboxes[:,:,1], bboxes[:,:,0],
                                    bboxes[:,:,3], bboxes[:,:,2]), dim = -1)

    return bboxes_roi_align #bboxes




class VanillaStochasticDETRHOIauxkl(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.latent_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        self.verb_class_embed = nn.Linear(self.latent_dim, num_verb_classes)
        # self.verb_class_embed = MLP(self.latent_dim, self.latent_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # Stochastic 
        # self.latent_mu = nn.Linear(hidden_dim, self.latent_dim)
        # self.latent_log_var = nn.Linear(hidden_dim, self.latent_dim)
        # self.obj_class_mu = nn.Linear(hidden_dim, self.latent_dim)
        # self.obj_class_log_var = nn.Linear(hidden_dim, self.latent_dim)

    def cal_kl(self, aux_tensor, tensor):
        # ensure the input hasn't used softmax processing
        # tensor1, tensor2 shape [bs, nq, num_class]
        assert aux_tensor.shape == tensor.shape
        bs, nq, num_class = tensor.shape
        aux_tensor = F.softmax(aux_tensor, dim = -1)
        tensor = F.softmax(tensor, dim = -1)
        kl = (tensor * torch.log(tensor/aux_tensor)).sum(dim = -1)
        return kl    # [bs, nq]

        
    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print(len(pos))
        # print(pos[-1].shape)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # self.transformer return decoder info and memory
        # hs: tensor [6, 2, 100, 256]  6 is the #decoder_layers

        outputs_obj_class = self.obj_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
        outputs_verb_class = self.verb_class_embed(hs)
        
        # calculate uncertainty via kl divergence between 4th output and 6th output
        obj_class_kl = self.cal_kl(outputs_obj_class[5,:], outputs_obj_class[4,:]) 
        verb_class_kl = self.cal_kl(outputs_verb_class[5,:], outputs_verb_class[4,:])
        # aux_kl = torch.stack((obj_class_kl, verb_class_kl), dim = 0)


        # verb_mu = self.latent_mu(hs)  # [6, b, nq, latent_dim]
        # verb_log_var = self.latent_log_var(hs)  # [6, b, nq, latent_dim]
        # if self.training:
        #     sampling_num = 5
        # else:
        #     sampling_num = 5
        # stochastic_prior = torch.randn([sampling_num, ] + list(verb_mu.shape), 
        #                                dtype = verb_mu.dtype, 
        #                                device = verb_mu.device)
        # verb_std = torch.exp(0.5*verb_log_var)
        # verb_latent = verb_mu + verb_std * stochastic_prior # [6, b, nq, latent_dim] * [sampling_num, 6, b, nq, latent_dim]
        # outputs_verb_class = self.verb_class_embed(verb_latent)
        # outputs_verb_class = outputs_verb_class.mean(dim = 0)
        # outputs_verb_class += res_outputs_verb_class

        # shared latent variable
        # outputs_obj_class = self.obj_class_embed(verb_latent)
        # outputs_obj_class = outputs_obj_class.mean(dim = 0)
        # outputs_obj_class += res_outputs_obj_class

        # obj_class_mu = self.obj_class_mu(hs)
        # obj_class_log_var = self.obj_class_log_var(hs)
        # obj_class_stochastic_prior = torch.randn([sampling_num, ] + list(obj_class_mu.shape),
        #                                          dtype = obj_class_mu.dtype,
        #                                          device = obj_class_mu.device)
        # obj_class_std = torch.exp(0.5*obj_class_log_var)
        # obj_class_latent = obj_class_mu + obj_class_std * obj_class_stochastic_prior
        # outputs_obj_class = self.obj_class_embed(obj_class_latent).mean(dim = 0)
        # outputs_obj_class += res_outputs_obj_class
        # gaussian_constraint = torch.cat((verb_log_var, obj_class_log_var), dim = -1) # verb_log_var
        
        
        # return KL divergence parameters
        # gaussian_constraint = torch.cat((verb_mu, verb_log_var), dim = -1)
        # out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
        #        'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
        #        'verb_kl_divergence': gaussian_constraint[-1]}

        # return verb_log_var for calculating entropy bound
        
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
               'aux_kl': {'obj_class_kl':obj_class_kl, 'verb_class_kl':verb_class_kl}} 

        if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        # return KL divergence parameters
        # return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d, 'verb_kl_divergence': e}
        #         for a, b, c, d, e in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
        #                               outputs_sub_coord[:-1], outputs_obj_coord[:-1], gaussian_constraint[:-1])]

        # return verb_log_var for calculating entropy bound
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]



class VanillaStochasticDETRHOI(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.latent_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        self.verb_class_embed = nn.Linear(self.latent_dim, num_verb_classes)
        # self.verb_class_embed = MLP(self.latent_dim, self.latent_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # Stochastic 
        self.latent_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, self.latent_dim)
        self.obj_class_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.obj_class_log_var = nn.Linear(hidden_dim, self.latent_dim)

        
    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print(len(pos))
        # print(pos[-1].shape)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # self.transformer return decoder info and memory
        # hs: tensor [6, 2, 100, 256]  6 is the #decoder_layers

        res_outputs_obj_class = self.obj_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
        res_outputs_verb_class = self.verb_class_embed(hs)

        verb_mu = self.latent_mu(hs)  # [6, b, nq, latent_dim]
        verb_log_var = self.latent_log_var(hs)  # [6, b, nq, latent_dim]
        if self.training:
            sampling_num = 5
        else:
            sampling_num = 5
        stochastic_prior = torch.randn([sampling_num, ] + list(verb_mu.shape), 
                                       dtype = verb_mu.dtype, 
                                       device = verb_mu.device)
        verb_std = torch.exp(0.5*verb_log_var)
        verb_latent = verb_mu + verb_std * stochastic_prior # [6, b, nq, latent_dim] * [sampling_num, 6, b, nq, latent_dim]
        outputs_verb_class = self.verb_class_embed(verb_latent)
        outputs_verb_class = outputs_verb_class.mean(dim = 0)
        outputs_verb_class += res_outputs_verb_class

        # shared latent variable
        # outputs_obj_class = self.obj_class_embed(verb_latent)
        # outputs_obj_class = outputs_obj_class.mean(dim = 0)
        # outputs_obj_class += res_outputs_obj_class

        obj_class_mu = self.obj_class_mu(hs)
        obj_class_log_var = self.obj_class_log_var(hs)
        obj_class_stochastic_prior = torch.randn([sampling_num, ] + list(obj_class_mu.shape),
                                                 dtype = obj_class_mu.dtype,
                                                 device = obj_class_mu.device)
        obj_class_std = torch.exp(0.5*obj_class_log_var)
        obj_class_latent = obj_class_mu + obj_class_std * obj_class_stochastic_prior
        outputs_obj_class = self.obj_class_embed(obj_class_latent).mean(dim = 0)
        outputs_obj_class += res_outputs_obj_class
        gaussian_constraint = torch.cat((verb_log_var, obj_class_log_var), dim = -1) # verb_log_var
        
        # return KL divergence parameters
        # gaussian_constraint = torch.cat((verb_mu, verb_log_var), dim = -1)
        # out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
        #        'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
        #        'verb_kl_divergence': gaussian_constraint[-1]}

        # return verb_log_var for calculating entropy bound
        
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
               'verb_log_var': gaussian_constraint[-1]} 

        if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord, gaussian_constraint)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, gaussian_constraint):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        # return KL divergence parameters
        # return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d, 'verb_kl_divergence': e}
        #         for a, b, c, d, e in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
        #                               outputs_sub_coord[:-1], outputs_obj_coord[:-1], gaussian_constraint[:-1])]

        # return verb_log_var for calculating entropy bound
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d, 'verb_log_var': e}
                for a, b, c, d, e in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1], gaussian_constraint[:-1])]



def norm_tensor(tensor):
    norm = torch.norm(tensor, p = 'fro', dim = -1).unsqueeze(dim = -1).expand_as(tensor)
    return tensor/norm


def count_fusion(x, y):
    return F.relu(x + y) - (x - y)*(x - y)

class SemanticGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, attention_type='embedded_dot_pro', head_num = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_type = attention_type

        if attention_type == 'embedded_dot_pro':
            self.relation_dim = hidden_dim
            self.semantic_q = [nn.Linear(input_dim, self.relation_dim),]
            self.semantic_k = [nn.Linear(input_dim, self.relation_dim),]
            self.semantic_v = [nn.Linear(input_dim, hidden_dim),]
            self.semantic_proj_res = nn.Linear(input_dim, hidden_dim)
            for _ in range(num_layers-1):
                self.semantic_q.append(nn.Linear(hidden_dim, hidden_dim))
                self.semantic_k.append(nn.Linear(hidden_dim, hidden_dim))
                self.semantic_v.append(nn.Linear(hidden_dim, hidden_dim))
            self.semantic_q = nn.ModuleList(self.semantic_q)
            self.semantic_k = nn.ModuleList(self.semantic_k)
            self.semantic_v = nn.ModuleList(self.semantic_v)

        elif attention_type == 'multihead_transformer':
            assert self.num_layers == 1
            self.head_num = head_num
            self.bottleneck_dim = int(self.hidden_dim//0.5)
            self.relation_dim = hidden_dim//self.head_num
            self.semantic_q = nn.Linear(input_dim, self.relation_dim)
            self.semantic_k = nn.Linear(input_dim, self.relation_dim)
            self.semantic_q = _get_clones(self.semantic_q, self.head_num)
            self.semantic_k = _get_clones(self.semantic_k, self.head_num)
            self.semantic_v = nn.Linear(input_dim, self.relation_dim)
            # self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(self.head_num)])
            self.semantic_proj_res = nn.Linear(input_dim, hidden_dim)

            self.W_t2 = nn.Linear(hidden_dim, self.bottleneck_dim)
            self.dropout2 = nn.Dropout(0.1)
            self.W_t1 = nn.Linear(self.bottleneck_dim, hidden_dim)
            self.LayerNorm = nn.LayerNorm([self.bottleneck_dim,])

        elif attention_type == 'MLP':
            self.mlp_layers = 3
            self.mlp = nn.ModuleList([nn.Linear(input_dim, hidden_dim) if i==0 else nn.Linear(hidden_dim, hidden_dim) for i in range(self.mlp_layers)])
            # self.nonlinearity = nn.ModuleList([nn.LeakyReLU(negative_slope=0.2, inplace=False) for i in range(self.mlp_layers-1)])
            self.nonlinearity = nn.ModuleList([nn.ReLU() for i in range(self.mlp_layers)])
            self.mlp_ln = nn.LayerNorm([hidden_dim,])
        
        elif attention_type == 'MLP_GNN':
            self.mlp_layers = 2
            self.mlp = nn.ModuleList([nn.Linear(input_dim, hidden_dim) if i==0 else nn.Linear(hidden_dim, hidden_dim) for i in range(self.mlp_layers)])
            # self.nonlinearity = nn.ModuleList([nn.LeakyReLU(negative_slope=0.2, inplace=False) for i in range(self.mlp_layers-1)])
            self.nonlinearity = nn.ModuleList([nn.ReLU() for i in range(self.mlp_layers)])
            self.mlp_ln = nn.ModuleList([nn.LayerNorm([hidden_dim,]) for i in range(self.mlp_layers)]) 

            self.relation_dim = hidden_dim
            self.semantic_ln = nn.ModuleList([nn.LayerNorm([hidden_dim,]) for _ in range(num_layers)])
            self.semantic_nonlinear = nn.ModuleList([nn.ReLU() for _ in range(num_layers)])
            self.semantic_q = nn.ModuleList([nn.Linear(hidden_dim, self.relation_dim) for _ in range(num_layers)])
            self.semantic_k = nn.ModuleList([nn.Linear(hidden_dim, self.relation_dim) for _ in range(num_layers)])
            self.semantic_v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

            # Bilinear Pooling
            # self.nheads = nheads
            # self.bilinear1 = nn.Linear(hidden_dim, hidden_dim)
            # self.bilinear2 = nn.Linear(hidden_dim, hidden_dim)
            # self.bilinear1 = _get_clones(self.bilinear1, nheads)
            # self.bilinear2 = _get_clones(self.bilinear2, nheads)
            # self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
        
            # hid_hid_dim = hidden_dim//nheads
            # self.W3 = nn.Linear(hidden_dim, hid_hid_dim)
            # self.W3 = _get_clones(self.W3, nheads)
            # self.W2 = nn.Linear(hidden_dim, hidden_dim)
            # self.W1 = nn.Linear(hidden_dim, hidden_dim)
            # self.nonlinear = nn.ReLU(inplace = True)
            # self.LayerNorm = nn.LayerNorm([hidden_dim,])
        
        
    def forward(self, x, cooccur_prior = None):
        assert len(x.shape) == 2
        if self.attention_type == 'embedded_dot_pro':
            for i in range(self.num_layers):
                x_q = self.semantic_q[i](x)
                x_k = self.semantic_k[i](x)
                x_v = self.semantic_v[i](x)
                # x_att = torch.einsum('ac,bc->ab', x_q, x_k)
                x_att = torch.einsum('ac,bc->ab', x_q, x_k) / math.sqrt(self.relation_dim)
                x_att = F.softmax(x_att, dim = -1)
                if cooccur_prior is not None:
                    x_att = x_att + cooccur_prior
                    print('cooccur prior')

                if i == 0:
                    x = F.relu(torch.matmul(x_att, x_v)) + self.semantic_proj_res(x) # self.verb_calibration_embedding
                else:
                    x = F.relu(torch.matmul(x_att, x_v)) + x
            trans_x = x
            # trans_x = norm_tensor(x)
        
        if self.attention_type == 'multihead_transformer':
            len_x = len(x.shape)
            if len_x == 2:
                x = x.unsqueeze(dim = 0)
            elif len_x == 4:
                x.shape = l, bs, q, hiddim
                x = x.view((l*bs, q, hiddim))
            elif len_x == 3:
                None
            else:
                print("Shape is not compatible")
                assert False

            x_v = self.semantic_v(x)
            multihead_ft = []
            for i in range(self.head_num):
                x_q_i = self.semantic_q[i](x)  # lbs, q, hiddim
                x_k_i = self.semantic_k[i](x) # * self.coef[i].expand_as(x_q_i)  # lbs, q, hiddim

                x_att_i = torch.einsum('abc,adc->abd', x_q_i, x_k_i) / math.sqrt(self.relation_dim)
                x_att_i = F.softmax(x_att_i, dim = -1)
                att_ft_i = torch.bmm(x_att_i, x_v)
                multihead_ft.append(att_ft_i)

            multihead_ft = torch.cat(multihead_ft, dim = -1)
            trans_ft = self.W_t1(F.relu(self.LayerNorm(self.W_t2(multihead_ft)), inplace = True))
            trans_x = trans_ft + self.semantic_proj_res(x)

            if len_x == 2:
                trans_x = trans_x.squeeze(dim = 0)
            elif len_x == 4:
                trans_x = trans_x.view((l, bs, q, hiddim))
            elif len_x == 3:
                None
        
        if self.attention_type == 'MLP':
            for i in range(self.mlp_layers):
                x = self.mlp[i](x)
                if i == self.mlp_layers-1:
                    x = self.mlp_ln(x)
                    x = self.nonlinearity[i](x)
                else:
                    x = self.nonlinearity[i](x)
            
            # trans_x = norm_tensor(x)
            trans_x = x
        
        if self.attention_type == 'MLP_GNN':
            for i in range(self.mlp_layers):
                x = self.mlp[i](x)
                x = self.mlp_ln[i](x)
                x = self.nonlinearity[i](x)
            
            for i in range(self.num_layers):
                x_q = self.semantic_q[i](x)
                x_k = self.semantic_k[i](x)
                x_v = self.semantic_v[i](x)
                x_att = torch.einsum('ac,bc->ab', x_q, x_k) / math.sqrt(self.relation_dim)
                x_att = F.softmax(x_att, dim = -1)
                x = self.semantic_nonlinear[i](self.semantic_ln[i](torch.matmul(x_att, x_v))) + x
            
            trans_x = x

        return trans_x


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, relation = 'bilinear', dropout = 0.1):
        super().__init__()
        self.relation = relation
        self.hidden_dim = hidden_dim
        if self.relation == 'bilinear':
            self.nheads = nheads
            self.bilinear1 = nn.Linear(hidden_dim, hidden_dim)
            self.bilinear2 = nn.Linear(hidden_dim, hidden_dim)
            self.bilinear1 = _get_clones(self.bilinear1, nheads)
            self.bilinear2 = _get_clones(self.bilinear2, nheads)
            self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
        
            hid_hid_dim = hidden_dim//nheads
            self.W3 = nn.Linear(hidden_dim, hid_hid_dim)
            self.W3 = _get_clones(self.W3, nheads)
            self.W2 = nn.Linear(hidden_dim, hidden_dim)
            self.W1 = nn.Linear(hidden_dim, hidden_dim)
            self.nonlinear = nn.ReLU(inplace = True)
            self.LayerNorm = nn.LayerNorm([hidden_dim,])
        
        if self.relation == 'embedded_dot_pro':
            self.nheads = nheads
            self.hidden_dim = hidden_dim
            self.hid_hid_dim = hidden_dim//nheads
            self.relation_q = nn.Linear(self.hidden_dim, self.hid_hid_dim)
            self.relation_k = nn.Linear(self.hidden_dim, self.hid_hid_dim)
            self.relation_q = _get_clones(self.relation_q, nheads)
            self.relation_k = _get_clones(self.relation_k, nheads)
            # self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
        
            self.W3 = nn.Linear(self.hidden_dim, self.hid_hid_dim)
            self.W3 = _get_clones(self.W3, nheads)
            self.bottleneck_dim = int(self.hidden_dim//0.5)
            self.W2 = nn.Linear(self.hidden_dim, self.bottleneck_dim)
            self.W1 = nn.Linear(self.bottleneck_dim, self.hidden_dim)
            self.nonlinear = nn.ReLU(inplace = True)
            self.LayerNorm = nn.LayerNorm([self.bottleneck_dim,])

        if self.relation == 'VanillaTrans':
            self.nheads = nheads
            self.hidden_dim = hidden_dim
            self.hid_hid_dim = hidden_dim//nheads
            self.relation_q = nn.Linear(self.hidden_dim, self.hid_hid_dim)
            self.relation_k = nn.Linear(self.hidden_dim, self.hid_hid_dim)
            self.relation_q = _get_clones(self.relation_q, nheads)
            self.relation_k = _get_clones(self.relation_k, nheads)
            # self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
        
            self.W3 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.W3 = _get_clones(self.W3, nheads)
            self.bottleneck_dim = int(self.hidden_dim)
            self.W2 = nn.Linear(self.hidden_dim, self.bottleneck_dim)
            self.W1 = nn.Linear(self.bottleneck_dim, self.hidden_dim)
            self.nonlinear = nn.ReLU(inplace = True)
            self.norm2 = nn.LayerNorm(self.hidden_dim)
            self.norm1 = nn.LayerNorm(self.hidden_dim)
            self.dropout3 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x: shape [6,2,100,256]
        '''
        # cal multi-head attention
        if self.relation == 'bilinear':
            x_trans = []
            for i in range(self.nheads):
                x_b1 = self.bilinear1[i](x) # [6,2,100,256]
                x_b2 = self.bilinear2[i](x)
                # x_b1 = torch.sigmoid(self.bilinear1[i](x)) # [6,2,100,256]
                # x_b2 = torch.sigmoid(self.bilinear2[i](x))

                x_b1 = x_b1 * self.coef[i]
                x_att = torch.einsum('abcd,abed->abce', x_b1, x_b2)
                x_att = torch.softmax(x_att, dim = -1)
                x_emb = self.W3[i](x)
                x_i = torch.einsum('abce,abef->abcf', x_att, x_emb)
                x_trans.append(x_i)  # [6,2,100,256/nheads]
            x_trans = torch.cat(x_trans, dim = -1)
            x_trans = self.W1(self.nonlinear(self.LayerNorm(self.W2(x_trans))))
            x_trans = x + x_trans
        
        if self.relation == 'embedded_dot_pro':
            x_trans = []
            for i in range(self.nheads):
                x_r1 = self.relation_q[i](x) # [6,2,100,256]
                x_r2 = self.relation_k[i](x)
                x_att = torch.einsum('abcd,abed->abce', x_r1, x_r2) / math.sqrt(self.hid_hid_dim)
                x_att = torch.softmax(x_att, dim = -1)
                x_emb = self.W3[i](x)
                x_i = torch.einsum('abce,abef->abcf', x_att, x_emb)
                x_trans.append(x_i)  # [6,2,100,256/nheads]
            x_trans = torch.cat(x_trans, dim = -1)
            x_trans = self.W1(self.nonlinear(self.LayerNorm(self.W2(x_trans))))
            x_trans = x + x_trans
        
        if self.relation == 'VanillaTrans':
            x_n = self.norm2(x)
            x_trans = []
            for i in range(self.nheads):
                x_r1 = self.relation_q[i](x_n) # [6,2,100,256]
                x_r2 = self.relation_k[i](x_n)
                x_att = torch.einsum('abcd,abed->abce', x_r1, x_r2) / math.sqrt(self.hid_hid_dim)
                x_att = torch.softmax(x_att, dim = -1)
                x_emb = self.W3[i](x_n)
                x_i = torch.einsum('abce,abef->abcf', x_att, x_emb)
                x_trans.append(x_i)  # [6,2,100,256/nheads]
            x_trans = torch.stack(x_trans, dim = -1).sum(dim = -1)
            x_trans = x + self.dropout3(x_trans)
            x_trans2 = self.norm1(x_trans)
            x_trans2 = self.W1(self.dropout2(self.nonlinear((self.W2(x_trans2)))))
            x_trans = x_trans + self.dropout1(x_trans2)
            
        return x_trans


class InterTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, nheads):
        super().__init__()
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.bilinear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear1 = _get_clones(self.bilinear1, nheads)
        self.bilinear2 = _get_clones(self.bilinear2, nheads)
        self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
    
        hid_hid_dim = hidden_dim//nheads
        self.W3 = nn.Linear(hidden_dim, hid_hid_dim)
        self.W3 = _get_clones(self.W3, nheads)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlinear = nn.ReLU(inplace = True)
        self.LayerNorm = nn.LayerNorm([hidden_dim,])
    
    def forward(self, x, y):
        '''
        Gather y features to x.
        x: [6,2,100,256]
        y: [6,2,100,256]
        '''
        x_trans = []
        for i in range(self.nheads):
            x_b1 = self.bilinear1[i](x) # [6,2,100,256]
            y_b2 = self.bilinear2[i](y)
            x_b1 = x_b1 * self.coef[i]
            x_att = torch.einsum('abcd,abed->abce', x_b1, y_b2)
            x_att = torch.softmax(x_att, dim = -1)
            y_emb = self.W3[i](y)
            x_i = torch.einsum('abce,abef->abcf', x_att, y_emb)
            x_trans.append(x_i)  # [6,2,100,256/nheads]
        x_trans = torch.cat(x_trans, dim = -1)
        x_trans = self.W1(self.nonlinear(self.LayerNorm(self.W2(x_trans))))
        x_trans = x + x_trans
        return x_trans


class InterLambdaLayer(nn.Module):
    def __init__(self, hidden_dim, nheads):
        super().__init__()
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.bilinear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear1 = _get_clones(self.bilinear1, nheads)
        self.bilinear2 = _get_clones(self.bilinear2, nheads)
        self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
    
        hid_hid_dim = hidden_dim//nheads
        self.W3 = nn.Linear(hidden_dim, hid_hid_dim)
        self.W3 = _get_clones(self.W3, nheads)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlinear = nn.ReLU(inplace = True)
        self.LayerNorm = nn.LayerNorm([hidden_dim,])
    
    def forward(self, x, y):
        '''
        Gather y features to x.
        x: [6,2,100,256]
        y: [6,2,100,256]
        '''
        x_trans = []
        for i in range(self.nheads):
            x_b1 = self.bilinear1[i](x) # [6,2,100,256]
            y_b2 = self.bilinear2[i](y)
            x_b1 = x_b1 * self.coef[i]
            x_att = torch.einsum('abcd,abed->abce', x_b1, y_b2)
            x_att = torch.softmax(x_att, dim = -1)
            y_emb = self.W3[i](y)
            x_i = torch.einsum('abce,abef->abcf', x_att, y_emb)
            x_trans.append(x_i)  # [6,2,100,256/nheads]
        x_trans = torch.cat(x_trans, dim = -1)
        x_trans = self.W1(self.nonlinear(self.LayerNorm(self.W2(x_trans))))
        x_trans = x + x_trans
        return x_trans



class MHCrossAttLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, relation = 'GClike', dropout = 0.1):
        super().__init__()
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        hid_hid_dim = hidden_dim//nheads
        self.bottleneck_dim = int(self.hidden_dim)
        self.relation = relation

        if self.relation == 'GClike':
            self.vision_W3 = nn.Linear(hidden_dim, hid_hid_dim)
            self.vision_sq = nn.Linear(hidden_dim, hid_hid_dim)
            self.vision_ex = nn.Linear(hid_hid_dim, hid_hid_dim)
            self.vision_W3 = _get_clones(self.vision_W3, nheads)
            self.vision_sq = _get_clones(self.vision_sq, nheads)
            self.vision_ex = _get_clones(self.vision_ex, nheads)
            self.vision_W2 = nn.Linear(hidden_dim, self.bottleneck_dim)
            self.vision_W1 = nn.Linear(self.bottleneck_dim, hidden_dim)
            self.vision_LayerNorm = nn.LayerNorm([self.bottleneck_dim,])

            self.semantic_W3 = nn.Linear(hidden_dim, hid_hid_dim)
            self.semantic_sq = nn.Linear(hidden_dim, hid_hid_dim)
            self.semantic_ex = nn.Linear(hid_hid_dim, hid_hid_dim)
            self.semantic_W3 = _get_clones(self.semantic_W3, nheads)
            self.semantic_sq = _get_clones(self.semantic_sq, nheads)
            self.semantic_ex = _get_clones(self.semantic_ex, nheads)
            self.semantic_W2 = nn.Linear(hidden_dim, self.bottleneck_dim)
            self.semantic_W1 = nn.Linear(self.bottleneck_dim, hidden_dim)
            self.semantic_LayerNorm = nn.LayerNorm([self.bottleneck_dim,])
        
        if self.relation == 'VanillaTrans':
            self.vision_W3 = nn.Linear(hidden_dim, hidden_dim)
            self.vision_sq = nn.Linear(hidden_dim, hid_hid_dim)
            self.vision_ex = nn.Linear(hid_hid_dim, hidden_dim)
            self.vision_W3 = _get_clones(self.vision_W3, nheads)
            self.vision_sq = _get_clones(self.vision_sq, nheads)
            self.vision_ex = _get_clones(self.vision_ex, nheads)
            self.vision_W2 = nn.Linear(hidden_dim, self.bottleneck_dim)
            self.vision_W1 = nn.Linear(self.bottleneck_dim, hidden_dim)
            self.vision_LayerNorm2 = nn.LayerNorm(self.hidden_dim)
            self.vision_LayerNorm1 = nn.LayerNorm(self.hidden_dim)
            self.vision_dropout3 = nn.Dropout(dropout)
            self.vision_dropout2 = nn.Dropout(dropout)
            self.vision_dropout1 = nn.Dropout(dropout)

            self.semantic_W3 = nn.Linear(hidden_dim, hidden_dim)
            self.semantic_sq = nn.Linear(hidden_dim, hid_hid_dim)
            self.semantic_ex = nn.Linear(hid_hid_dim, hidden_dim)
            self.semantic_W3 = _get_clones(self.semantic_W3, nheads)
            self.semantic_sq = _get_clones(self.semantic_sq, nheads)
            self.semantic_ex = _get_clones(self.semantic_ex, nheads)
            self.semantic_W2 = nn.Linear(hidden_dim, self.bottleneck_dim)
            self.semantic_W1 = nn.Linear(self.bottleneck_dim, hidden_dim)
            self.semantic_LayerNorm2 = nn.LayerNorm(self.hidden_dim)
            self.semantic_LayerNorm1 = nn.LayerNorm(self.hidden_dim)
            self.semantic_dropout3 = nn.Dropout(dropout)
            self.semantic_dropout2 = nn.Dropout(dropout)
            self.semantic_dropout1 = nn.Dropout(dropout)

    
    def forward(self, vx, sx):
        if self.relation == 'GClike':
            vx_enhance = []
            for i in range(self.nheads):
                vx_att = torch.sigmoid(self.vision_ex[i](torch.relu(self.vision_sq[i](sx))))
                vx_emb = vx_att * self.vision_W3[i](vx) # Self Aggregation (Initial)
                # vx_emb = vx_att * self.vision_W3[i](sx) # Cross Aggregation
                vx_enhance.append(vx_emb)
            vx_enhance = torch.cat(vx_enhance, dim = -1)
            vx_enhance = vx + self.vision_W1(torch.relu(self.vision_LayerNorm(self.vision_W2(vx_enhance))))

            sx_enhance = []
            for i in range(self.nheads):
                sx_att = torch.sigmoid(self.semantic_ex[i](torch.relu(self.semantic_sq[i](vx))))
                sx_emb = sx_att * self.semantic_W3[i](sx) # Self Aggregation (Initial)
                # sx_emb = sx_att * self.semantic_W3[i](vx) # Cross Aggregation
                sx_enhance.append(sx_emb)
            sx_enhance = torch.cat(sx_enhance, dim = -1)
            sx_enhance = sx + self.semantic_W1(torch.relu(self.semantic_LayerNorm(self.semantic_W2(sx_enhance))))
        

        if self.relation == 'VanillaTrans':
            vx_n = self.vision_LayerNorm2(vx)
            sx_n = self.semantic_LayerNorm2(sx)

            vx_enhance = []
            for i in range(self.nheads):
                vx_att = torch.sigmoid(self.vision_ex[i](torch.relu(self.vision_sq[i](sx_n))))
                vx_emb = vx_att * self.vision_W3[i](vx_n) # Self Aggregation (Initial)
                # vx_emb = vx_att * self.vision_W3[i](sx) # Cross Aggregation
                vx_enhance.append(vx_emb)
            vx_enhance = torch.stack(vx_enhance, dim = -1).sum(dim = -1)
            vx = vx + self.vision_dropout3(vx_enhance)
            vx2 = self.vision_LayerNorm1(vx)
            # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            # src = src + self.dropout2(src2)
            vx2 = self.vision_W1(self.vision_dropout2(torch.relu(self.vision_W2(vx2))))
            vx = vx + self.vision_dropout1(vx2)

            sx_enhance = []
            for i in range(self.nheads):
                sx_att = torch.sigmoid(self.semantic_ex[i](torch.relu(self.semantic_sq[i](vx_n))))
                sx_emb = sx_att * self.semantic_W3[i](sx_n) # Self Aggregation (Initial)
                # sx_emb = sx_att * self.semantic_W3[i](vx) # Cross Aggregation
                sx_enhance.append(sx_emb)
            sx_enhance = torch.stack(sx_enhance, dim = -1).sum(dim = -1)
            sx = sx + self.semantic_dropout3(sx_enhance)
            sx2 = self.semantic_LayerNorm1(sx)
            sx2 = self.semantic_W1(self.semantic_dropout2(torch.relu((self.semantic_W2(sx2)))))
            sx = sx + self.semantic_dropout1(sx)

        return vx_enhance, sx_enhance


class MHSelfAttLayer(nn.Module):
    def __init__(self, hidden_dim, nheads):
        super().__init__()
        self.nheads = nheads
        hid_hid_dim = hidden_dim//nheads
        self.vision_W3 = nn.Linear(hidden_dim, hid_hid_dim)
        self.vision_sq = nn.Linear(hidden_dim, hid_hid_dim)
        self.vision_ex = nn.Linear(hid_hid_dim, hid_hid_dim)
        self.vision_W3 = _get_clones(self.vision_W3, nheads)
        self.vision_sq = _get_clones(self.vision_sq, nheads)
        self.vision_ex = _get_clones(self.vision_ex, nheads)
        self.vision_W2 = nn.Linear(hidden_dim, hidden_dim)
        self.vision_W1 = nn.Linear(hidden_dim, hidden_dim)
        self.vision_LayerNorm = nn.LayerNorm([hidden_dim,])

        self.semantic_W3 = nn.Linear(hidden_dim, hid_hid_dim)
        self.semantic_sq = nn.Linear(hidden_dim, hid_hid_dim)
        self.semantic_ex = nn.Linear(hid_hid_dim, hid_hid_dim)
        self.semantic_W3 = _get_clones(self.semantic_W3, nheads)
        self.semantic_sq = _get_clones(self.semantic_sq, nheads)
        self.semantic_ex = _get_clones(self.semantic_ex, nheads)
        self.semantic_W2 = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_W1 = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_LayerNorm = nn.LayerNorm([hidden_dim,])
    
    def forward(self, vx, sx):
        vx_enhance = []
        for i in range(self.nheads):
            vx_att = torch.sigmoid(self.vision_ex[i](torch.relu(self.vision_sq[i](vx))))
            vx_emb = vx_att * self.vision_W3[i](vx)
            vx_enhance.append(vx_emb)
        vx_enhance = torch.cat(vx_enhance, dim = -1)
        vx_enhance = vx + self.vision_W1(torch.relu(self.vision_LayerNorm(self.vision_W2(vx_enhance))))

        sx_enhance = []
        for i in range(self.nheads):
            sx_att = torch.sigmoid(self.semantic_ex[i](torch.relu(self.semantic_sq[i](sx))))
            sx_emb = sx_att * self.semantic_W3[i](sx)
            sx_enhance.append(sx_emb)
        sx_enhance = torch.cat(sx_enhance, dim = -1)
        sx_enhance = sx + self.semantic_W1(torch.relu(self.semantic_LayerNorm(self.semantic_W2(sx_enhance))))
        
        return vx_enhance, sx_enhance


class VanillaCrossAttLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.vision_W1 = nn.Linear(hidden_dim, hidden_dim)
        self.vision_res = nn.Linear(hidden_dim, hidden_dim)
        self.vision_W2 = nn.Linear(hidden_dim, hidden_dim)

        self.semantic_W1 = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_res = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_W2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, vx, sx):
        '''
        vx: vision features [6,2,100,256]
        sx: semantic features [6,2,100,256]
        '''
        # Inter
        res_vx = self.vision_res(vx)
        att_vx = res_vx + res_vx * torch.sigmoid(self.vision_W1(sx))
        # att_vx = res_vx + res_vx * torch.sigmoid(self.vision_W2(torch.relu(self.vision_W1(sx))))
        res_sx = self.semantic_res(sx)
        att_sx = res_sx + res_sx * torch.sigmoid(self.semantic_W1(vx))
        # att_sx = res_sx + res_sx * torch.sigmoid(self.semantic_W2(torch.relu(self.semantic_W1(vx))))

        return att_vx, att_sx



class CrossModalCalibration(nn.Module):
    def __init__(self, hidden_dim, nlayers = 1):
        super().__init__()
        # Inter
        # self.vision_W1 = nn.Linear(hidden_dim, hidden_dim)
        # self.vision_res = nn.Linear(hidden_dim, hidden_dim)
        # self.vision_W2 = nn.Linear(hidden_dim, hidden_dim)
        # self.semantic_W1 = nn.Linear(hidden_dim, hidden_dim)
        # self.semantic_res = nn.Linear(hidden_dim, hidden_dim)
        # self.semantic_W2 = nn.Linear(hidden_dim, hidden_dim)
        # Intra
        # self.vision_intra_trans = TransformerLayer(hidden_dim, nheads = 2)
        # self.semantic_intra_trans = TransformerLayer(hidden_dim, nheads = 2)

        self.nlayers = nlayers
        # Inter-Modal Calibration (InterC)
        # self.CrossAtt = VanillaCrossAttLayer(hidden_dim)
        # self.CrossAtt = MHCrossAttLayer(hidden_dim, nheads = 4, relation = 'VanillaTrans')
        self.CrossAtt = MHCrossAttLayer(hidden_dim, nheads = 2)
        self.CrossAtt = _get_clones(self.CrossAtt, nlayers)
        # self.SelfAtt = MHSelfAttLayer(hidden_dim, nheads=2)
        # self.SelfAtt = _get_clones(self.SelfAtt, nlayers)

        # Inter-Transformer
        # self.vision_inter_trans = InterTransformerLayer(hidden_dim, nheads=2)
        # self.semantic_inter_trans = InterTransformerLayer(hidden_dim, nheads=2)
        # self.vision_inter_trans = _get_clones(self.vision_inter_trans, nlayers)
        # self.semantic_inter_trans = _get_clones(self.semantic_inter_trans, nlayers)

        # Intra-Modal Enhance Calibration (IntraEC)
        # self.vision_intra_trans = TransformerLayer(hidden_dim, nheads = 2, relation = 'embedded_dot_pro')
        # self.semantic_intra_trans = TransformerLayer(hidden_dim, nheads = 2, relation = 'embedded_dot_pro')
        # self.vision_intra_trans = TransformerLayer(hidden_dim, nheads = 4, relation = 'VanillaTrans')
        # self.semantic_intra_trans = TransformerLayer(hidden_dim, nheads = 4, relation = 'VanillaTrans')
        self.vision_intra_trans = TransformerLayer(hidden_dim, nheads = 2)
        self.semantic_intra_trans = TransformerLayer(hidden_dim, nheads = 2)
        self.vision_intra_trans = _get_clones(self.vision_intra_trans, nlayers)
        self.semantic_intra_trans = _get_clones(self.semantic_intra_trans, nlayers)

    def forward(self, vx, sx):
        '''
        vx: vision features [6,2,100,256]
        sx: semantic features [6,2,100,256]
        '''

        for l in range(self.nlayers):
            # MH2CrossAttLayer_intraTrans2_nlayers1, Highest
            # Inter
            att_vx, att_sx = self.CrossAtt[l](vx, sx)
            # Intra
            vx = self.vision_intra_trans[l](att_vx)
            sx = self.semantic_intra_trans[l](att_sx)
            # vx, sx = att_vx, att_sx
            # vx = self.vision_intra_trans[l](vx)
            # sx = self.semantic_intra_trans[l](sx)

            # # SelfAtt
            # # Inter
            # att_vx, att_sx = self.SelfAtt[l](vx, sx)
            # att_vx, att_sx = self.CrossAtt[l](att_vx, att_sx)
            # # Intra
            # vx = self.vision_intra_trans[l](att_vx)
            # sx = self.semantic_intra_trans[l](att_sx)

            # # Inter-Intra
            # # Inter
            # att_vx = self.vision_inter_trans[l](vx, sx)
            # att_sx = self.semantic_inter_trans[l](sx, vx)
            # # Intra
            # vx = self.vision_intra_trans[l](att_vx)
            # sx = self.semantic_intra_trans[l](att_sx)

            # # CrossEnhance-Inter-Intra
            # # Inter
            # att_vx, att_sx = self.CrossAtt[l](vx, sx)
            # att_vx = self.vision_inter_trans[l](att_vx, att_sx)
            # att_sx = self.semantic_inter_trans[l](att_sx, att_vx)
            # # Intra
            # vx = self.vision_intra_trans[l](att_vx)
            # sx = self.semantic_intra_trans[l](att_sx)

            # Inter-Intra-CrossEnhance
            # # Inter
            # att_vx = self.vision_inter_trans[l](vx, sx)
            # att_sx = self.semantic_inter_trans[l](sx, vx)
            # # Intra
            # att_vx = self.vision_intra_trans[l](att_vx)
            # att_sx = self.semantic_intra_trans[l](att_sx)
            # # Cross-Enhance
            # vx, sx = self.CrossAtt[l](att_vx, att_sx)

        return vx, sx


class CrossModalityGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, attention_type='multihead_transformer'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_type = attention_type

        if attention_type == 'embedded_dot_pro':
            self.semantic_q = [nn.Linear(input_dim, hidden_dim),]
            self.semantic_k = [nn.Linear(input_dim, hidden_dim),]
            self.semantic_v = [nn.Linear(input_dim, hidden_dim),]
            # self.semantic_proj_res = nn.Linear(input_dim, hidden_dim)
            for _ in range(num_layers-1):
                self.semantic_q.append(nn.Linear(hidden_dim, hidden_dim))
                self.semantic_k.append(nn.Linear(hidden_dim, hidden_dim))
                self.semantic_v.append(nn.Linear(hidden_dim, hidden_dim))
            self.semantic_q = nn.ModuleList(self.semantic_q)
            self.semantic_k = nn.ModuleList(self.semantic_k)
            self.semantic_v = nn.ModuleList(self.semantic_v)
        elif attention_type == 'multihead_transformer':
            assert self.num_layers == 1
            self.head_num = 4
            self.semantic_q = nn.Linear(input_dim, hidden_dim)
            self.semantic_k = nn.Linear(input_dim, hidden_dim)
            self.semantic_q = _get_clones(self.semantic_q, self.head_num)
            self.semantic_k = _get_clones(self.semantic_k, self.head_num)
            self.semantic_v = nn.Linear(input_dim, hidden_dim)
            self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(self.head_num)])

            self.LayerNorm = nn.LayerNorm([hidden_dim,])
            self.W_t1 = nn.Linear(hidden_dim*self.head_num, hidden_dim)
            self.W_t2 = nn.Linear(hidden_dim, hidden_dim)
        

        self.fusion_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_2 = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_gate = nn.Linear(hidden_dim, hidden_dim)
            
    
    def forward(self, x, y, cooccur_prior = None):
        '''
        x : vision features [6, 2, 100, 256]
        y : language features [117, 256]
        '''
        if self.attention_type == 'embedded_dot_pro':
            assert self.num_layers == 1
            for i in range(self.num_layers):
                x_q = self.semantic_q[i](x)
                y_k = self.semantic_k[i](y)
                y_v = self.semantic_v[i](y)
                # x_att = torch.einsum('ac,bc->ab', x_q, x_k)
                x_att = torch.einsum('abce,de->abcd', x_q, y_k) / math.sqrt(self.hidden_dim) # [6, 2, 100, 117]
                x_att = F.softmax(x_att, dim = -1)
                if cooccur_prior is not None:
                    x_att = x_att + cooccur_prior
                    # print('cooccur')
                semantic_agg = torch.einsum('abcd,de->abce', x_att, y_v)
            return semantic_agg
            
                
                # if i == 0:
                #     x = F.relu(torch.matmul(x_att, x_v)) + self.semantic_proj_res(x) # self.verb_calibration_embedding
                # else:
                #     x = F.relu(torch.matmul(x_att, x_v)) + x
        
        if self.attention_type == 'multihead_transformer':
            assert len(x.shape) == 4
            l, bs, q, hiddim = x.shape
            x = x.reshape((l*bs, q, hiddim))

            assert len(y.shape) == 2
            y = y.unsqueeze(dim = 0)

            y_v = self.semantic_v(y).expand(l*bs, -1, -1)
            multihead_ft = []
            for i in range(self.head_num):
                x_q = self.semantic_q[i](x)  # lbs, q, hiddim
                y_k = self.semantic_k[i](y).expand(l*bs, -1, -1)  # lbs, q, hiddim
                y_k = y_k * self.coef[i].expand_as(y_k)

                x_att = torch.einsum('abc,adc->abd', x_q, y_k)
                x_att = F.softmax(x_att, dim = -1)
                att_ft = torch.bmm(x_att, y_v)
                multihead_ft.append(att_ft)

            multihead_ft = torch.cat(multihead_ft, dim = -1)
            semantic_aug = self.W_t2(F.relu(self.LayerNorm(self.W_t1(multihead_ft)), inplace = True))
            semantic_aug = semantic_aug.view((l, bs, q, hiddim))

            modality_fus = count_fusion(self.fusion_1(semantic_aug), self.fusion_2(x.view((l, bs, q, hiddim))))
            # semantic_gate2 = torch.sigmoid(self.semantic_gate2(x))
            # modality_fus = count_fusion(self.fusion_1(semantic_gate2 * trans_ft), self.fusion_2(x))
            return modality_fus




# inf_time_list = [[],[],[],[],[]]
class OCN(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, dataset = 'hico', aux_loss = False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_verb_classes = num_verb_classes
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.transformer.decoder.obj_class_embed = self.obj_class_embed
        # self.transformer.decoder.verb_class_embed = self.verb_class_embed
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # Initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.obj_class_embed.bias.data = torch.ones(num_obj_classes+1) * bias_value
        self.verb_class_embed.bias.data = torch.ones(num_verb_classes) * bias_value
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)
        # torch.nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        # torch.nn.init.constant_(self.reference_points.bias.data, 0.)

        
        # obj_verb_co with smoothing v2 (81 is uniform)
        if dataset == 'hico':
            # Laplacian Smoothing (Also dubbed Additive Smoothing)
            obj_verb_co = np.load('datasets/priors/obj_verb_cooccurrence.npz')['cond_prob_co_matrices']
            obj_verb_co = torch.cat((torch.tensor(obj_verb_co).float(), torch.zeros((1, num_verb_classes))), dim = 0)
            obj_verb_co = obj_verb_co + 0.1/obj_verb_co.shape[1]
            # obj_verb_co = torch.ones(obj_verb_co.shape)  # beta = infinity
            # obj_verb_co = torch.cat((torch.tensor(obj_verb_co).float(), torch.ones((1, num_verb_classes))), dim = 0)
                # obj_verb_co = obj_verb_co / np.expand_dims(obj_verb_co.sum(axis=1), axis = 1)
            obj_verb_co = obj_verb_co / obj_verb_co.sum(dim=1).unsqueeze(dim = 1)
            self.register_buffer('obj_verb_co', obj_verb_co)
            print('obj_verb_co has nan ? ' + str(np.isnan(obj_verb_co).sum()))

            # Jelinek-Mercer Method
            # obj_verb_co = np.load('datasets/priors/obj_verb_cooccurrence.npz')['cond_prob_co_matrices']
            # obj_verb_co = torch.cat((torch.tensor(obj_verb_co).float(), torch.zeros((1, num_verb_classes))), dim = 0)
            # obj_verb_co = obj_verb_co * (1 - 0.7) + 0.7 / obj_verb_co.shape[1]
            # obj_verb_co = obj_verb_co / obj_verb_co.sum(dim=1).unsqueeze(dim = 1)
            # self.register_buffer('obj_verb_co', obj_verb_co)
            # print('obj_verb_co has nan ? ' + str(np.isnan(obj_verb_co).sum()))

        elif dataset == 'vcoco':
            obj_verb_co = np.load('datasets/priors/obj_verb_cooccurrence_vcoco.npz')['joint_prob_co_matrices']
            print('obj_verb_co has nan ? ' + str(np.isnan(obj_verb_co).sum()))
            obj_verb_co[np.isnan(obj_verb_co)] = 0.1/obj_verb_co.shape[1]  # Eliminate nan entries in the matrix
            obj_verb_co = torch.cat((torch.tensor(obj_verb_co).float(), torch.zeros((1, num_verb_classes))), dim = 0)
            obj_verb_co = obj_verb_co + 0.1/obj_verb_co.shape[1] 
            obj_verb_co = obj_verb_co / obj_verb_co.sum(dim=1).unsqueeze(dim = 1)
            self.register_buffer('obj_verb_co', obj_verb_co)
            print('obj_verb_co has nan ? ' + str(np.isnan(obj_verb_co).sum()))



        # verb_verb_co with smoothing
        if dataset == 'hico':
            verb_verb_co = np.load('datasets/priors/verb_verb_cooccurrence.npz')['cond_prob_co_matrices']  # 实际上用的是Joint Probability
            verb_verb_co = verb_verb_co / np.expand_dims(verb_verb_co.sum(axis=1), axis = 1)
            verb_verb_co[np.isnan(verb_verb_co)] = 0  # add to prevent nan
            self.register_buffer('verb_verb_co', torch.tensor(verb_verb_co).float())
            print('verb_verb_co has nan ? ' + str(np.isnan(verb_verb_co).sum()))
            print('verb_verb_co sum: ' + str(verb_verb_co.sum()))

        elif dataset == 'vcoco':
            verb_verb_co = np.load('datasets/priors/verb_verb_cooccurrence_vcoco.npz')['cond_prob_co_matrices']  # 实际上用的是Joint Probability
            verb_verb_co = verb_verb_co / np.expand_dims(verb_verb_co.sum(axis=1), axis = 1)
            verb_verb_co[np.isnan(verb_verb_co)] = 0  # add to prevent nan
            self.register_buffer('verb_verb_co', torch.tensor(verb_verb_co).float())
            print('verb_verb_co sum: ' + str(verb_verb_co.sum()))


        # verb word embedding
        if dataset == 'hico':
            # Prerained Model embedding
            verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/hico_verb_glove-wiki-gigaword-300.npz')['embedding_list']) # [:,None]
            # verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/hico_verb_glove-wiki-gigaword-50.npz')['embedding_list']) # [:,None]
            # verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/hico_verb_fasttext-wiki-news-subwords-300.npz')['embedding_list'])# [:,None]
            # verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/hico_verb_word2vec-google-news-300.npz')['embedding_list'])# [:,None]
            verb_word_embedding = norm_tensor(verb_word_embedding)
            self.register_buffer('verb_word_embedding', verb_word_embedding)

            # # one_hot verb embedding
            # verb_word_embedding = torch.eye(num_verb_classes)
            # self.register_buffer('verb_word_embedding', verb_word_embedding)
        elif dataset == 'vcoco':
            verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/vcoco_verb_glove-wiki-gigaword-300.npz')['embedding_list'])# [:,None]
            # verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/vcoco_verb_fasttext-wiki-news-subwords-300.npz')['embedding_list'])# [:,None]
            # verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/vcoco_verb_word2vec-google-news-300.npz')['embedding_list'])# [:,None]
            verb_word_embedding = norm_tensor(verb_word_embedding)
            self.register_buffer('verb_word_embedding', verb_word_embedding)

        # Semantic Reasoning
        self.semantic_graph = SemanticGraph(300, 256, 1, attention_type='embedded_dot_pro')
        # self.semantic_graph = SemanticGraph(117, 256, 1, attention_type='MLP_GNN')
        # self.semantic_graph = SemanticGraph(300, 256, 1, attention_type='multihead_transformer', head_num = 2)
        # self.semantic_obj_graph = SemanticGraph(300, 256, 1, attention_type='embedded_dot_pro')
        # self.semantic_graph = SemanticGraph(117, 256, 1, attention_type='MLP')
        

        # Cross modality operation
        # self.cross_modality_graph = CrossModalityGraph(hidden_dim, hidden_dim, 1, attention_type='multihead_transformer')
        # self.cross_modality_graph = CrossModalityGraph(hidden_dim, hidden_dim, 1, attention_type='embedded_dot_pro')
        # self.semantic_gate1 = nn.Linear(hidden_dim, num_verb_classes)
        self.semantic_gate2 = nn.Linear(hidden_dim, hidden_dim)
        self.hs_gate = nn.Linear(hidden_dim, hidden_dim)
        # self.semantic_gate2_1 = nn.Linear(hidden_dim, hidden_dim//16)
        # self.semantic_gate2_2 = nn.Linear(hidden_dim//16, hidden_dim)
        self.cross_modal_calibration = CrossModalCalibration(hidden_dim, nlayers = 1)
        self.fusion_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, samples: NestedTensor, **kwargs):
        inf_time = time.time()
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # inf_time_list[0].append(time.time()-inf_time)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None
        # self.transformer return decoder info and memory
        # hs: tensor [6, 2, 100, 256]  6 is the #decoder_layers

        # New semantic implementation
        # semantic = self.semantic_graph(self.verb_word_embedding, self.verb_verb_co)
        semantic = self.semantic_graph(self.verb_word_embedding)

        # inf_time_list[1].append(time.time()-inf_time)
        # Save semantic embedding
        # np.savez_compressed('GloVe_semantic_embeddings.npz', 
        #                      semantic = np.array(semantic.cpu()))
        
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # self.transformer return decoder info and memory
        # hs: tensor [6, 2, 100, 256]  6 is the #decoder_layers

        outputs_obj_class = self.obj_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
        # inf_time_list[3].append(time.time()-inf_time)

        
        # cross_enhance
        # Attention aggregation
        # semantic_aug = self.cross_modality_graph(hs, semantic)
        # Statistical Prior Aggregation
        outputs_obj_81 = outputs_obj_class.argmax(dim =-1).unsqueeze(-1).expand(-1,-1,-1,self.num_verb_classes) # [6,2,100]
        obj_verb_co = self.obj_verb_co.expand(outputs_obj_81.shape[:-2]+(-1,-1))
        outputs_obj_co = torch.gather(obj_verb_co, dim =2, index = outputs_obj_81) # [6, 2, 100, 117]
        semantic_aug = torch.einsum('abcd,de->abce', outputs_obj_co, semantic)
        cross_hs, cross_semantic_aug = self.cross_modal_calibration(hs, semantic_aug)
        hs_aug = count_fusion(self.fusion_1(cross_hs), self.fusion_2(cross_semantic_aug))


        # Verb Model
            # vanilla
        outputs_verb_class = self.verb_class_embed(hs_aug)


        # Original
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1], 'semantic':semantic, 'verb_verb_co':self.verb_verb_co,}# 'joint_verb_verb_co':self.joint_verb_verb_co,} # 'semantic_low':semantic_low}
        if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)
        return out

    @torch.jit.unused   # Original
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]



class DETRHOI(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        # self.verb_class_embed = MLP(hidden_dim, hidden_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, **kwargs):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print(len(pos))
        # print(pos[-1].shape)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # self.transformer return decoder info and memory
        # hs: tensor [6, 2, 100, 256]  6 is the #decoder_layers

        outputs_obj_class = self.obj_class_embed(hs)
        outputs_verb_class = self.verb_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class SetCriterionHOI(nn.Module):
    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, verb_loss_type):
        super().__init__()

        assert verb_loss_type in ['weighted_bce', 'focal', 'focal_bce', 'asymmetric_bce', 'CB_focal_bce','bce']
        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.verb_loss_type = verb_loss_type

        # For CB focal
        samples = np.load('datasets/priors/hico_verb_samples.npz')['matrices']
        samples = torch.tensor(samples).float()
        self.register_buffer('samples', samples)
        self.img_num_hico = 37536
        self.img_num_vcoco = 5400
        self.query_num = 100
        self.register_buffer('bce_weight', self.BCE_weight())
    
    def BCE_weight(self,):
        total_num = self.img_num_hico * self.query_num
        pos_verb_samples = self.samples
        neg_verb_samples = total_num - pos_verb_samples
        pos_verb_w = torch.ones(self.samples.shape)
        neg_verb_w = torch.sqrt(pos_verb_samples) / torch.sqrt(neg_verb_samples)
        return torch.stack((pos_verb_w, neg_verb_w), dim = 1)

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        # idx: a tuple (batch_idx, src_idx)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # target_classes: init with a tensor of size src_logits.shape[:2]
        #                 and filled with self.num_obj_classes (no object class)
        target_classes[idx] = target_classes_o
        # fill the target_classes with the gt object classes

        # print("src_logits " + str(src_logits.shape)) # [2, 100, 81]
        # print("target_classes " + str(target_classes.shape)) # [2, 100]
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits'] # [2, 100, 81]
        # print('pred_logits' + str(pred_logits.shape))
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        # tgt_lengths: number of predicted objects 
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # card_pred: number of true objects 
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        # l1_loss that takes the mean element-wise absolute value difference.
        
        losses = {'obj_cardinality_error': card_err}
        # print('tgt_lengths:'+str(tgt_lengths.shape)) # [2]
        # print('card_pred:'+str(card_pred.shape)) # [2]
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits'] # [2, 100, 117]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        # [num_of_verbs, 117]
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        # target_classes = torch.ones_like(src_logits)*57
        # target_classes = torch.full(src_logits, self.num_verb_classes,
        #                             dtype=torch.int64, device=src_logits.device)

        if self.verb_loss_type == 'bce':
            loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        elif self.verb_loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._neg_loss(src_logits, target_classes)
        elif self.verb_loss_type == 'focal_bce':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._focal_bce(src_logits, target_classes)
        elif self.verb_loss_type == 'asymmetric_bce':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._asymmetric_bce(src_logits, target_classes)
        elif self.verb_loss_type == 'CB_focal_bce':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._CB_focal_bce(src_logits, target_classes)
        elif self.verb_loss_type == 'weighted_bce':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._weighted_bce(src_logits, target_classes)
        
        if 'pri_pred_verb_logits' in outputs:
            pri_src_logits = outputs['pri_pred_verb_logits']
            if self.verb_loss_type == 'bce':
                loss_verb_ce += F.binary_cross_entropy_with_logits(pri_src_logits, target_classes)
            elif self.verb_loss_type == 'focal':
                pri_src_logits = pri_src_logits.sigmoid()
                loss_verb_ce += self._neg_loss(pri_src_logits, target_classes)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses
    
    def loss_gt_verb_recon(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits'] # [2, 100, 117]
        semantic = outputs['semantic']
        hs = outputs['hs']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        # [num_of_verbs, 117]
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        
        if self.verb_loss_type == 'bce':
            cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        elif self.verb_loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            # self.check_0_1(verb_gt_recon, 'src_logits')
            cls_loss = self._neg_loss(src_logits, target_classes)
        
        # Loss for All queries
        loss_recon = torch.tensor(0., device = target_classes.device)
        semantic_norm = norm_tensor(semantic)
        hs_norm = norm_tensor(hs)
        cos_sim = torch.einsum('abd,cd->abc', hs_norm, semantic_norm)
        pos_loss = 1 - cos_sim
        neg_loss = torch.clamp(cos_sim - 0.1, min = 0)
        recon_loss = (pos_loss * target_classes + neg_loss * (1 - target_classes)).sum() / target_classes.sum()

        loss = cls_loss + recon_loss

        return {'loss_verb_gt_recon': loss}


    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_sub_boxes = outputs['pred_sub_boxes'][idx] # shape like [5, 4] [6, 4]...
        # print(src_sub_boxes)
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # print(target_sub_boxes)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        
        return losses
    
    def loss_kl_divergence(self, outputs, targets, indices, num_interactions):
        if 'verb_kl_divergence' in outputs:
            kl_param = outputs['verb_kl_divergence']
            bs, num_queries, latentdim2 = kl_param.shape # 2, 100, 256*2
            verb_mu, verb_log_var = kl_param[:,:,:latentdim2//2], kl_param[:,:,latentdim2//2:]
            verb_var = torch.exp(verb_log_var)
            loss = -0.5 * (1 + verb_log_var - verb_mu*verb_mu - verb_var)
            loss = torch.mean(loss)
            
            # Loss for Target queries
            # kl_param = outputs['verb_kl_divergence']
            # bs, num_queries, latentdim2 = kl_param.shape # 2, 100, 256*2
            # verb_mu, verb_log_var = kl_param[:,:,:latentdim2//2], kl_param[:,:,latentdim2//2:]
            # idx = self._get_src_permutation_idx(indices)
            # src_verb_mu = verb_mu[idx]
            # src_verb_log_var = verb_log_var[idx]
            # src_verb_var = torch.exp(src_verb_log_var)
            # loss = -0.5 * (1 + src_verb_log_var - src_verb_mu*src_verb_mu - src_verb_var)
            # loss = loss.sum() / num_interactions
        else:
            assert False

        return {'loss_kl_divergence': loss}

    def cal_entropy_loss(self, log_var, latentdim, bound):
        cons_term = latentdim/2.*(math.log(2*math.pi) + 1.)
        var_term = 0.5*torch.sum(log_var, dim = 1)
        avg_entropy = torch.mean(cons_term + var_term) 
        loss = torch.max(torch.Tensor((0, bound - avg_entropy)).to(avg_entropy.device))
        return loss

    def loss_entropy_bound(self, outputs, targets, indices, num_interactions):
        if 'verb_log_var' in outputs:
            log_var = outputs['verb_log_var']
            b, nq, latentdim = log_var.shape
            latentdim = latentdim//2
            verb_log_var, obj_class_log_var = log_var[...,:latentdim], log_var[...,latentdim:]
            loss = self.cal_entropy_loss(verb_log_var, latentdim, bound = 256) +\
                   self.cal_entropy_loss(obj_class_log_var, latentdim, bound = 256)

            # # assert 'verb_log_var' in outputs
            # verb_log_var = outputs['verb_log_var']
            # b, nq, latentdim = verb_log_var.shape
            # # Loss for all queries
            # verb_log_var = outputs['verb_log_var']
            # cons_term = latentdim/2.*(math.log(2*math.pi) + 1.)
            # var_term = 0.5*torch.sum(verb_log_var, dim = 1)
            # avg_entropy = torch.mean(cons_term + var_term) 
            # loss = torch.max(torch.Tensor((0, 256 - avg_entropy)).to(avg_entropy.device))

            # Loss for matched queries
            # idx = self._get_src_permutation_idx(indices)
            # src_verb_log_var = outputs['verb_log_var'][idx] 
            # src_verb_std shape: [all machted queries in all batches, latent_dim]
            # cons_term = latentdim/2.*(math.log(2*math.pi) + 1.)
            # # print('consterm ' + str(cons_term))
            # var_term = 0.5*torch.sum(src_verb_log_var, dim = 1)
            # # print('varterm ' + str(var_term.shape))
            # avg_entropy = torch.mean(cons_term + var_term) 
            # # print('avg_entropy ' + str(avg_entropy))
            # loss = torch.max(torch.Tensor((0, 32 - avg_entropy)).to(avg_entropy.device))
            
        elif 'masked_context_log_var' in outputs:
            masked_memory_log_var = outputs['masked_context_log_var']
            _, latentdim = masked_memory_log_var.shape

            # Entropy bound
            cons_term = latentdim/2.*(math.log(2*math.pi) + 1.)
            var_term = 0.5*torch.sum(masked_memory_log_var, dim = 1) # [all pixels with false masks in all batches,]
            pixel_avg_entropy = torch.mean(cons_term + var_term)
            # print(pixel_avg_entropy)
            
            # image_avg_entropy = torch.mean(torch.sum(cons_term + var_term, dim = 0),
            #                                dim = 0)
            # pixel_avg_entropy = torch.mean(torch.sum(cons_term + var_term, dim = 0),
            #                                dim = 0)
            loss = torch.max(torch.Tensor((0, 256 - pixel_avg_entropy))).to(pixel_avg_entropy.device)

        else:
            assert False

        return {'loss_entropy_bound': loss}

        # assert 'verb_std' in outputs
        # verb_std = outputs['verb_std']
        # # print(sigma2.shape)
        # b, nq, latentdim = verb_std.shape
        # verb_var = verb_std * verb_std
        
        # # Entropy bound 
        # idx = self._get_src_permutation_idx(indices)
        # src_verb_std = outputs['verb_std'][idx] 
        # # src_verb_std shape: [all machted queries in all batches, latent_dim]
        
        # # cons_term = torch.Tensor((latentdim/2.*(math.log(2*math.pi) + 1.))).to(verb_std.device)
        # cons_term = latentdim/2.*(math.log(2*math.pi) + 1.)
        # # print('consterm ' + str(cons_term))
        # var_term = torch.sum(verb_std, dim = 1)
        # # print('varterm ' + str(var_term))
        # avg_entropy = torch.mean(cons_term + var_term) 
        # # print('avg_entro ' + str(avg_entropy))
        # # loss = torch.max(torch.Tensor((0, 128 - avg_entropy),
        # #                               device = avg_entropy.device))
        # loss = torch.max(torch.Tensor((0, 256 - avg_entropy)).to(avg_entropy.device))

        # # KL_divergence
        # # loss = -0.5 * (1 + )

        # return {'loss_entropy_bound': loss}
    
    def loss_verb_hm(self, outputs, targets, indices, num_interactions):
        pred_verb_hm, mask = outputs['verb_hm']
        neg_loss = 0.
        # mask shape [bs,c,h,w]
        for ind, t in enumerate(targets):
            gt_verb_hm = t['verb_hm']
            valid_1 = torch.sum(~mask[ind][:,:,0])
            valid_2 = torch.sum(~mask[ind][:,0,:])
            # interpolate input [bs,c,h,w]
            gt_verb_hm = F.interpolate(gt_verb_hm.unsqueeze(0), size = (valid_1, valid_2)).squeeze(0)

            # print(gt_verb_hm.shape)
            # print(mask.shape)
            # print(valid_w, valid_h)
            # print(pred_verb_hm[ind].shape)
            # print(gt_verb_hm[:,:valid_w,:valid_h].shape)
            neg_loss += self._neg_loss(pred_verb_hm[ind][:,:valid_1,:valid_2], gt_verb_hm)

        # print(pred_verb_hm.shape)
        # print(gt_verb_hm.shape)
        return {'loss_verb_hm': neg_loss}
    
    def loss_verb_threshold(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        assert 'pred_verb_thr' in outputs
        src_logits = outputs['pred_verb_logits'] # [2, 100, 117]
        thr = outputs['pred_verb_thr'] # [2, 100, 117]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        # [num_of_verbs, 117]
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o  # [2, 100, 117]
        # target_classes = torch.ones_like(src_logits)*57
        # target_classes = torch.full(src_logits, self.num_verb_classes,
        #                             dtype=torch.int64, device=src_logits.device)

        sigma = torch.sigmoid(src_logits - thr)
        # Vanilla loss
        # pos_verb = torch.log(sigma)
        # neg_verb = torch.log(1-sigma)
        # loss_verb_thr = pos_verb * target_classes + neg_verb * (1 - target_classes)
        # loss_verb_thr = (-loss_verb_thr).sum(dim = -1).mean()
        # Binary Focal loss
        loss_verb_thr = self._neg_loss(sigma, target_classes, eps = 1e-6)

        return {'loss_verb_threshold': loss_verb_thr}


    def loss_semantic_similar(self, outputs, targets, indices, num_interactions):
        temperature = 0.05
        if 'semantic' in outputs and 'verb_verb_co' in outputs:
            # semantic = outputs['semantic_low'] # 117, 256
            semantic = outputs['semantic'] # 117, 256
            # verb_verb_co = outputs['verb_verb_co'] # 117, 117
            # verb_verb_co = outputs['joint_verb_verb_co'] # 117, 117
            # Symmetric cond prob
            verb_verb_co = outputs['verb_verb_co'] 
            verb_verb_co = verb_verb_co + verb_verb_co.T
            verb_verb_co = verb_verb_co / verb_verb_co.sum()

            norm_semantic = norm_tensor(semantic)
            # norm_semantic = semantic
            semantic_sim = torch.einsum('ab,cb->ac', norm_semantic, norm_semantic)  # 117, 117
            eye_mask = ~(torch.eye(verb_verb_co.shape[0], device = verb_verb_co.device) == 1)
            semantic_sim = semantic_sim[eye_mask]
            

            # normalize: make semantic_sim sum to 1
            # semantic_sim = semantic_sim

            # MSE loss
            # semantic_mse = F.mse_loss(semantic_sim, verb_verb_co, reduce = False)
            #     # w/ Relaxation
            # # relax_flag = verb_verb_co > 0.01   # 0.01:308; 0.02:97; 0.05:36; 0.1:13; 
            # # loss_sim = torch.sum(semantic_mse[relax_flag])
            #     # w/o Relaxation
            # loss_sim = torch.sum(semantic_mse)

            # KL loss
            # # verb_verb_co = verb_verb_co + verb_verb_co.T
            # semantic_sim = F.log_softmax(semantic_sim, dim = -1)
            # loss_sim = F.kl_div(semantic_sim, verb_verb_co, reduction = 'sum')
            # sum normalization KL
            # semantic_sim = torch.log(semantic_sim / semantic_sim.sum(dim = -1).unsqueeze(dim = -1))
            # loss_sim = F.kl_div(semantic_sim, verb_verb_co, reduction = 'sum')
            # KL loss with relaxation
            # semantic_sim = F.log_softmax(semantic_sim, dim = -1)
            # loss_sim = F.kl_div(semantic_sim, verb_verb_co, reduce = False)
            # loss_sim = (loss_sim[verb_verb_co>0.3]).sum()
            
            # KL loss for joint probability distribution
            # semantic_sim = F.log_softmax(semantic_sim.flatten())
            # loss_sim = F.kl_div(semantic_sim, verb_verb_co.flatten(), reduction = 'sum')
            # KL loss for joint probability distribution with eye mask
            semantic_sim = F.log_softmax(semantic_sim / temperature)
            loss_sim = F.kl_div(semantic_sim, verb_verb_co[eye_mask], reduction = 'sum')

            # semantic_sim_soft = F.softmax(semantic_sim / temperature)
            # semantic_sim_logsoft = F.log_softmax(semantic_sim / temperature)
            # loss_sim = (semantic_sim_soft * (semantic_sim_logsoft - (verb_verb_co[eye_mask]).log())).sum()

            #################### AAAI 2020 implementatioin ##################
            # semantic = outputs['semantic'] # 117, 256
            # verb_verb_co = outputs['verb_verb_co'] 
            # verb_verb_co = verb_verb_co + verb_verb_co.T
            # norm_semantic = norm_tensor(semantic)
            # semantic_sim = torch.einsum('ab,cb->ac', norm_semantic, norm_semantic) # # 117, 117
            # # MSE loss
            # semantic_mse = F.mse_loss(semantic_sim, verb_verb_co, reduce = False)
            #     # w/ Relaxation
            # relax_flag = verb_verb_co > 0.1   # 0.01:308; 0.02:97; 0.05:36; 0.1:13; 
            # loss_sim = torch.sum(semantic_mse[relax_flag])
            #     # w/o Relaxation
            # # loss_sim = torch.sum(semantic_mse)

        else:
            loss_sim = torch.tensor([0.], device = outputs['pred_obj_logits'].device).sum()
        

        # Semantic prototype (S_au)
        # src_logits = outputs['pred_verb_logits'] # [2, 100, 117]
        # semantic = outputs['semantic'] # 117, 256
        # norm_semantic = norm_tensor(semantic)  # 我猜这是prototype2
        # # norm_semantic = semantic
        # hs = outputs['hs']
        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.zeros_like(src_logits)
        # target_classes[idx] = target_classes_o
        # proto_logits = torch.einsum('ac,bqc->bqa', norm_semantic, hs)
        # loss_sim = self._neg_loss(proto_logits, target_classes)

        return {'loss_semantic_similar': loss_sim}
    
    def _weighted_bce(self, pred, gt, eps = 1e-6):
        # pos_inds = gt.eq(1).float()
        # num_pos  = pos_inds.float().sum()
        # print(pred.shape)
        # print(gt.shape)
        # loss = F.binary_cross_entropy_with_logits(pred, gt, weight = self.bce_weight.T) / num_pos
        # return loss

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = self.bce_weight[:,1]

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * pos_inds
        neg_loss = torch.log(1 - pred) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    
    def _CB_focal_bce(self, pred, gt, eps = 1e-6, gamma = 2, alpha = 0.5, vol = 2, beta = 0.9999):
        beta_weight = (1-beta) / (1 - torch.pow(beta, self.samples)) 
        beta_weight = beta_weight.unsqueeze(dim = 0).unsqueeze(dim = 0)

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * alpha * vol * pos_inds * beta_weight
        neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * (1 - alpha) * vol * neg_inds * beta_weight

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss



    def _asymmetric_bce(self, pred, gt, eps = 1e-6, gamma_pos = 0, gamma_neg = 3, m = 0.01, vol = 1):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred_p = torch.clamp(pred, min = eps, max = 1.)
        pos_loss = torch.log(pred_p) * torch.pow(1 - pred_p, gamma_pos) * vol * pos_inds
        pred_m = torch.clamp(pred - m, min = 0, max = 1. - eps)
        neg_loss = torch.log(1 - pred_m) * torch.pow(pred_m, gamma_neg) * neg_weights * vol * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    
    def _focal_bce(self, pred, gt, eps = 1e-6, gamma = 2, alpha = 0.5, vol = 4):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * alpha * vol * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * (1 - alpha) * vol * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss



    def _neg_loss(self, pred, gt, eps = 1e-6):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        
        # indices: list of tensor tuples
        # like [(tensor([ 5, 42, 51, 61]), tensor([2, 3, 0, 1])), (tensor([20]), tensor([0]))]
        
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            # 'verb_labels': self.loss_gt_verb_recon,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'entropy_bound':self.loss_entropy_bound,
            'kl_divergence':self.loss_kl_divergence,
            'verb_hm':self.loss_verb_hm,
            'semantic_similar':self.loss_semantic_similar,
            'verb_threshold':self.loss_verb_threshold,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # print(outputs_without_aux['pred_verb_logits'].shape)
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'verb_hm':
                        continue
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # matching every two decodings 
        # num_interactions = sum(len(t['obj_labels']) for t in targets)
        # num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_interactions)
        # num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()
        # # Compute all the requested losses
        # losses = {}
        # # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # if 'aux_outputs' in outputs:
        #     for i, aux_outputs in enumerate(outputs['aux_outputs']):
        #         if i in [0,2,4]:
        #             indices = self.matcher(aux_outputs, targets)
        #         for loss in self.losses:
        #             kwargs = {}
        #             if loss == 'obj_labels':
        #                 # Logging is enabled only for the last layer
        #                 kwargs = {'log': False}
        #             l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
        #             l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
        #             losses.update(l_dict)
        # for loss in self.losses:
        #     losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        return losses


class PostProcessHOI(nn.Module):
    def __init__(self, subject_category_id, sigmoid = True):
        super().__init__()
        self.subject_category_id = subject_category_id
        self.sigmoid = sigmoid
        print('Postpeocess sigmoid = '+str(self.sigmoid))

        obj_verb_co = np.load('datasets/priors/obj_verb_cooccurrence.npz')['cond_prob_co_matrices']
        obj_verb_co = torch.tensor(obj_verb_co).float()
        obj_verb_co = obj_verb_co + 0.1/obj_verb_co.shape[1]
        obj_verb_co = obj_verb_co / obj_verb_co.sum(dim=1).unsqueeze(dim = 1)
        self.register_buffer('obj_verb_co', obj_verb_co)
        print(np.isnan(obj_verb_co).sum())

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']
        # shape [bs, 100, 81]
        # shape [bs, 100, 117]
        # shape [bs, 100, 4]
        # shape [bs, 100, 4]

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2  # h, w

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)  
        # obj_prob[..., :-1] ([bs,100,80]) deletes the final class for no objects 
        # [bs, 100] [bs, 100] =  torch.max() returns values and indices

        if self.sigmoid:
            verb_scores = out_verb_logits.sigmoid()
        else:
            verb_scores = out_verb_logits

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]  # scale 0~1. to 0~w and 0~h

        results = []
        for os, ol, vs, sb, ob, op in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes, obj_prob[..., :-1]):
            sl = torch.full_like(ol, self.subject_category_id)
            # sl(Subject label) denotes the person label, set subject_category_id by default
            # sl shape [100]

            l = torch.cat((sl, ol))
            # l shape [200]
            b = torch.cat((sb, ob))
            # b shape [200, 4]
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1) # multiply the object score, a general score for an classified object
            # [100, 117] * [100, 1] = [100, 117]
            # Alternation
            # vs = vs * torch.matmul(op, self.obj_verb_co.to(op.device))

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)