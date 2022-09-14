import torch
import torch.nn as nn
from .build import MODELS
import numpy as np
from utils import misc
import torch.nn.functional as F
from extensions.chamfer_dist import ChamferDistanceL2
from utils.losses import geodesic_distance
from utils.bbox_utils import get_bbox_from_keypoints
from utils.transform import rot_from_heading
from utils.transform import vc_to_cn, cn_to_vc, normalize_scale, restore_scale
from utils.sampling import get_partial_mesh_batch

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
    
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix
    
@MODELS.register_module()    
class VCN_VC(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Less patches/number_fine increases the runtime of this model            
        self.sel_k = 30
        self.number_fine = config.num_pred
        self.encoder_channel = config.encoder_channel        
        self.n_patches = 16 # must be powers of 4 if we want integer division of 16384
        grid_size = int(np.sqrt(self.number_fine // self.n_patches))
        self.pose_encoder = nn.Sequential(
            nn.Conv1d(3,64,1),            
            nn.LeakyReLU(),
            nn.Conv1d(64,128,1),
            nn.LeakyReLU(),
            nn.Conv1d(128,1024,1),
            nn.AdaptiveMaxPool1d(output_size=1)
        )
        self.pose_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 9)            
        )
        
        self.first_conv = nn.Sequential(
            nn.Conv1d(3,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,self.encoder_channel,1)
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.encoder_channel) for i in range(0,self.n_patches)])
        a = torch.linspace(-0.5, 0.5, steps=grid_size, dtype=torch.float).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
        b = torch.linspace(-0.5, 0.5, steps=grid_size, dtype=torch.float).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2).cuda() # 1 2 S
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_coarse = ChamferDistanceL2()
        self.loss_partial = ChamferDistanceL2()
        self.loss_translation = nn.SmoothL1Loss(reduction='none')
        self.loss_dims = nn.SmoothL1Loss(reduction='none')  

    def get_loss(self, ret_dict, in_dict):     
        gt_boxes = in_dict['gt_boxes']

        loss_dict = {}        
        coarse_vc = torch.matmul(ret_dict['coarse_cn'], ret_dict['reg_rot']) + ret_dict['reg_centre'].unsqueeze(1)
        pred_box = get_bbox_from_keypoints(coarse_vc, gt_boxes) # B 7        
        loss_dict['dims'] = self.loss_dims(gt_boxes[:,3:6].cuda(), pred_box[:,3:6]).mean()

        # Translation loss
        gt_centres = gt_boxes[:,:3]
        loss_dict['translation'] = self.loss_translation(gt_centres.cuda(), ret_dict['reg_centre']).mean()
        
        # Rotation loss
        gt_headings = gt_boxes[:,-1]
        gt_rmats = rot_from_heading(gt_headings)
        theta = geodesic_distance(ret_dict['reg_rot'], gt_rmats)
        loss_dict['rotation'] = theta.mean()
        
        # gt_cn = vc_to_cn(in_dict['complete'], in_dict['gt_boxes']) # B N 3
        # loss_dict['coarse'] = self.loss_coarse(ret_dict['coarse_cn'], gt_cn)        
        # loss_dict['partial'] = self.loss_partial(ret_dict['coarse_cn'], gt_cn)  

        # Coarse loss - downsample complete with fps
        if in_dict['training']:
            gt_cn = vc_to_cn(in_dict['complete'], in_dict['gt_boxes']) # B N 3
            ds_complete = misc.fps(gt_cn, ret_dict['coarse_cn'].shape[1])
            loss_dict['coarse'] = self.loss_coarse(ret_dict['coarse_cn'], ds_complete)               

            in_cn = vc_to_cn(in_dict['input'], in_dict['gt_boxes'])
            pred_surface = get_partial_mesh_batch( in_cn, ret_dict['coarse_cn'], k=self.sel_k)
            gt_surface = get_partial_mesh_batch( in_cn, ds_complete, k=self.sel_k)
            loss_dict['partial'] = self.loss_partial(pred_surface, gt_surface)       
        return loss_dict

    def forward(self, in_dict):
        ret = {}
        
        pc = in_dict['input']
        bs , n , _ = pc.shape
        
    
        # Centre pointcloud on mean of the points
        pts_mean = pc.mean(dim=1).unsqueeze(1)
        pts_meancentered = pc - pts_mean
        
        # Regress the relative pose
        pose_feat = self.pose_encoder(pts_meancentered.permute(0,2,1)).view(bs, -1)  # B 256 n
        rel_pose = self.pose_fc(pose_feat)        
        trans = rel_pose[:,:3].unsqueeze(1) # B 1 3
        centre = pts_mean + trans
        rot6d = rel_pose[:,3:9]
        rot_mat = compute_rotation_matrix_from_ortho6d(rot6d)     
        ret['reg_rot'] = rot_mat
        
        pc_cn = torch.matmul(pc - centre, rot_mat.permute(0,2,1))     

        # encoder
        feature = self.first_conv(pc_cn.transpose(2,1))  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.second_conv(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0] # B 1024
        
        patch_list = []
        for i in range(0,self.n_patches):             
            seed = self.folding_seed.tile(bs, 1,1)            
            feat = feature_global.expand(feature_global.size(0), feature_global.size(1), seed.size(2)).contiguous()
            y = torch.cat([seed, feat], dim=1).contiguous() # B 1024+2 grid_size**2
            patch_list.append(self.decoder[i](y))            
            
            coarse = torch.cat(patch_list, 2).transpose(1,2).contiguous()
        ret['coarse_cn'] = coarse
        coarse_vc = torch.matmul(coarse, rot_mat) + centre
        ret['coarse'] = coarse_vc.contiguous()
        ret['reg_centre'] = centre.squeeze(1)
                
        return ret
