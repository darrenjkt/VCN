import torch
import torch.nn as nn
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2

import torch
import torch.nn as nn
import numpy as np
from extensions.chamfer_dist import ChamferDistanceL2
    
def mlp_conv(in_channels, layer_dims, bn=None, bn_params=None):
    layers = []
    for i, out_channel in enumerate(layer_dims[:-1]):
        layers += [nn.Conv1d(in_channels, out_channel, kernel_size=1),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU(inplace=True)]
        in_channels = out_channel

    layers += [nn.Conv1d(in_channels, layer_dims[-1], kernel_size=1)]
    mlp_block = nn.Sequential(*layers)
    
    return nn.Sequential(*layers)

class PCNencoder(nn.Module):
    def __init__(self):
        super(PCNencoder, self).__init__()
        self.mlp_conv1 = nn.Sequential(
            nn.Conv1d(3,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,1)
        )
        self.mlp_conv2 = nn.Sequential(
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,1024,1)
        )
        
    def point_maxpool(self, inputs, npts, keepdims):
        # Max pool to get a 256 feature vec for the whole object 
        # [(1,256,1),(1,256,1),...] of len=batchsize
        outputs = [torch.max(f, dim=2, keepdim=keepdims)[0] for f in torch.split(inputs, npts, dim=2)]
        return torch.cat(outputs, dim=0)
    
    def point_unpool(self, inputs, npts):
        # Assign same 256 features to all points in the original object pcd
        # [(1,256,N1),(1,256,N2),...] of len=batchsize
        outputs = [torch.tile(f, [1, 1, npts[i]]) for i,f in enumerate(inputs)]

        return torch.cat(outputs, dim=2)

    def forward(self, x, npts_per_id):
        # Pytorch is (B,C,N) format

        # 1 3 N
        mlp_feat = self.mlp_conv1(x) # 1 256 N
        symmetric_feat = self.point_maxpool(mlp_feat, npts_per_id, keepdims=True)  # B 256 1
        symmetric_feat = self.point_unpool(symmetric_feat, npts_per_id)  # 1 256 N
        
        # Concatenate global (symmetric) and point (mlp) features
        features = torch.cat([mlp_feat, symmetric_feat], dim=1) # 1 512 N

        # Process the combined features
        # 1 global 1024-feature vector per object (i.e. 1024 channels)
        combined_feat = self.mlp_conv2(features)  # 1 1024 N
        combined_feat = self.point_maxpool(combined_feat, npts_per_id, keepdims=False)  # B 1024 1
        
        return combined_feat

class PCNdecoder(nn.Module):
    def __init__(self):
        super(PCNdecoder, self).__init__()
        self.grid_scale = 0.05
        self.grid_size = 4
        self.num_coarse = 1024
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.mlp = nn.Sequential(
            nn.Linear(self.num_coarse,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,3*self.num_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024+3+2,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,3,1)
        )
    def forward(self, x):

        # Coarse completion
        coarse = self.mlp(x) # (B,num_coarse*3)
        coarse = coarse.reshape([-1, 3, self.num_coarse]) # B 3 1024
        
        # --- Fine completion ---
        # 2D grid u x u for each coarse point
        u = torch.linspace(start=-self.grid_scale, end=self.grid_scale, steps=self.grid_size).cuda() # create 4x4 grid
        grid = torch.meshgrid(u,u) # 4x4 grid
        grid = torch.unsqueeze(torch.reshape(torch.stack(grid, dim=2), (2,-1)),0) 
        folding_seed = torch.tile(grid, [x.shape[0], 1, self.num_coarse]) # B 2 N
        
        point_feat = torch.tile(torch.unsqueeze(coarse, 2), [1, 1, self.grid_size **2, 1])
        point_feat = point_feat.reshape([-1, 3, self.num_fine]) # B 3 N
        
        global_feat = torch.tile(x.unsqueeze(2), [1, 1, self.num_fine]) # B 1024 N
        feat = torch.cat([folding_seed, point_feat, global_feat], dim=1) # B 1024+3+2 N
        center = torch.tile(torch.unsqueeze(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = center.reshape([-1, 3, self.num_fine]) # B 3 N
        
        fine = self.final_conv(feat) + center # B 3 N

        # For some reason, contiguous improved my model losses
        return (coarse.transpose(1,2).contiguous(), fine.transpose(1,2).contiguous())

@MODELS.register_module()
class PCNSSC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = PCNencoder()
        self.decoder = PCNdecoder()
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine
            
    def forward(self, x, npts):
        x = x.permute(0,2,1)
        encoded_feats = self.encoder(x, npts)
        coarse, fine = self.decoder(encoded_feats)
        return coarse, fine