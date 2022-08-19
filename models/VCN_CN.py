# This computes coarse loss with fps sampling, no fine loss or pred
import torch
import torch.nn as nn
from .build import MODELS
from utils import misc
from extensions.chamfer_dist import ChamferDistanceL2
from utils.transform import rot_from_heading, rotate_points_along_z
from utils.losses import geodesic_distance
from utils.bbox_utils import get_dims, get_bbox_from_keypoints
from utils.sampling import get_partial_mesh_batch

def fc_layers(layer_dims, last_as_linear=True):
    
    in_channels = layer_dims[0]
    layers = []
    for out_channel in layer_dims[1:]:
        if out_channel == layer_dims[-1] and last_as_linear:
            layers.append(nn.Linear(in_channels,out_channel))
            break
            
        layers += [nn.Linear(in_channels,out_channel),
                    nn.ReLU(inplace=True)]
        in_channels = out_channel            
        
    return nn.Sequential(*layers)

class FeatureEncoder(nn.Module):
    def __init__(self, dims):
        super(FeatureEncoder, self).__init__()
        # 3, 64, 128, 256, 256, 512
        self.mlp_conv1 = nn.Sequential(
            nn.Conv1d(dims[0],dims[1],1),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(dims[1],dims[2],1)
        )
        self.mlp_conv2 = nn.Sequential(
            nn.Conv1d(dims[3],dims[4],1),
            nn.BatchNorm1d(dims[4]),
            nn.ReLU(inplace=True),
            nn.Conv1d(dims[4],dims[5],1)
        )
    def forward(self, x, n, keepdims=False):
        # Pytorch is (B,C,N) format

        feature = self.mlp_conv1(x)  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.mlp_conv2(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=keepdims)[0]
        
        return feature_global

    
@MODELS.register_module()
class VCN_CN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sel_k = 30 # select nearest 30 points to each input point
        self.number_coarse = 1024

        self.encoder = FeatureEncoder([3, 128, 256, 512, 512, self.number_coarse])
        self.shape_fc = fc_layers([1024, 1024, 1024, 3*self.number_coarse], last_as_linear=True) # canonical shape
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_coarse = ChamferDistanceL2()
        self.loss_partial = ChamferDistanceL2()
        self.loss_translation = nn.SmoothL1Loss(reduction='none')
        self.loss_dims = nn.SmoothL1Loss(reduction='none')    

    def get_loss(self, ret_dict, in_dict):     
        gt_boxes = in_dict['gt_boxes']

        loss_dict = {}        
        # Coarse loss - downsample complete with fps
        if in_dict['training']:
            ds_complete = misc.fps(in_dict['complete'], ret_dict['coarse'].shape[1])
            loss_dict['coarse'] = self.loss_coarse(ret_dict['coarse'], ds_complete)            

            pred_surface = get_partial_mesh_batch( in_dict['input'], ret_dict['coarse'], k=self.sel_k)
            gt_surface = get_partial_mesh_batch( in_dict['input'], ds_complete, k=self.sel_k)
            loss_dict['partial'] = self.loss_partial(pred_surface, gt_surface)                        

        return loss_dict

    def forward(self, in_dict):
        ret = {}

        bs , n , _ = in_dict['input'].shape
        pc_cn = vc_to_cn(in_dict['input'], in_dict['gt_boxes']) # B N 3
        pc = normalize_scale(pc_cn, in_dict['gt_boxes'])

        # encoder
        feature_global = self.encoder(pc.permute(0,2,1), n)  # B 1024
        coarse = self.shape_fc(feature_global).reshape(-1,self.number_coarse,3) # B coarse_pts 3            
        
        # Bring points back to sensor view
        coarse_rescaled = restore_scale(coarse.contiguous(), in_dict['gt_boxes'])
        ret['coarse'] = cn_to_vc(coarse_rescaled, in_dict['gt_boxes'])
        
        return ret