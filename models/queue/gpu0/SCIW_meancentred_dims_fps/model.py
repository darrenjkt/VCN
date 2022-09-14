import torch
import torch.nn as nn
from .build import MODELS
from utils import misc
from extensions.chamfer_dist import ChamferDistanceL2
from utils.transform import rot_from_heading
from utils.bbox_utils import get_dims, get_bbox_from_keypoints
from utils.transform import cn_to_vc_rt, vc_to_cn_rt

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

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)    
    return theta

def conv_layers(layer_dims, last_as_conv=False):
    
    in_channels = layer_dims[0]
    conv_layers = []
    for out_channel in layer_dims[1:]:
        if out_channel == layer_dims[-1] and last_as_conv:
            conv_layers.append(nn.Conv1d(in_channels, out_channel, kernel_size=1))
            break
            
        conv_layers += [nn.Conv1d(in_channels, out_channel, kernel_size=1),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU()]
        in_channels = out_channel
    return nn.Sequential(*conv_layers)

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

def compute_ensemble_loss(gt_boxes, pred, t_weight=10, r_weight=1):
    """
    gt_boxes size (B 7)
    pred size (N_branch B 9)
    """
    gt_trans = gt_boxes[:,:3] # B 3
    gt_rotm = rot_from_heading(gt_boxes[:,-1]) # B 3 3 
    n_branches, batch_size, _ = pred.shape
    
    # Here we calculate a single loss value for each branch, then select the min
    pred_trans = pred[:,:,:3]
    branch_trans_loss = torch.square(pred_trans-gt_trans).mean(dim=2).sum(dim=1) # N_branch
    
    pred_rotm = [compute_rotation_matrix_from_ortho6d(p[:,3:9]) for p in pred]
    branch_rotm_loss = torch.stack([compute_geodesic_distance_from_two_matrices(prot, gt_rotm) for prot in pred_rotm]).sum(dim=1) # N_branch
    
    # Loss for each branch, shape (N_branch) 
    branch_loss = t_weight*branch_trans_loss + r_weight*branch_rotm_loss
    min_loss_idx = torch.argmin(branch_loss, dim=0)
    min_loss_mask = torch.zeros(branch_loss.shape).cuda() # one hot vector
    min_loss_mask[min_loss_idx] = 1
    branch_mask = torch.tile(min_loss_mask.view(n_branches,1,1), (1,batch_size,3))
    
    # Calculate loss only for best branch
    trans_l2 = nn.MSELoss(reduction='none')
    trans_loss = trans_l2(pred_trans, gt_trans.tile([n_branches,1,1])) * branch_mask
    rotm_loss = (branch_rotm_loss * min_loss_mask)
    ens_loss = t_weight*trans_loss.mean() + r_weight*rotm_loss.mean()
    
    return ens_loss, min_loss_idx  
    
@MODELS.register_module()    
class VCN_VC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.number_fine = config.num_pred
        self.encoder_channel = config.encoder_channel
        num_pose_candidates = 4
        
        grid_size = 4 # set default
        self.grid_size = grid_size
        assert self.number_fine % grid_size**2 == 0
        self.number_coarse = self.number_fine // (grid_size ** 2 )
        
  # Branch 0
        self.encoder_0 = FeatureEncoder([3, 128, 256, 512, 512, self.number_coarse])        
        self.pose_ensemble0 = nn.ModuleList([nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 9)            
        ) for i in range(num_pose_candidates)])        
        self.distilled_pose0 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 9)           
        )
        self.shape_decoder_0 = fc_layers([1024, 1024, 1024, 3*self.number_coarse], last_as_linear=True) # canonical shape

        # Branch 1
        self.encoder_1 = FeatureEncoder([3, 128, 256, 512, 512, self.number_coarse])    
        self.pose_ensemble1 = nn.ModuleList([nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 9)            
        ) for i in range(num_pose_candidates)])        
        self.distilled_pose1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 9)           
        )
        
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_coarse = ChamferDistanceL2()
        self.loss_coarse1 = ChamferDistanceL2()
        self.loss_translation = nn.MSELoss(reduction='none')
        self.loss_dims = nn.SmoothL1Loss(reduction='none')    

    def get_loss(self, ret_dict, in_dict):     
        if in_dict['training']: 
            gt_boxes_0 = in_dict['gt_boxes_0']
            gt_boxes_1 = in_dict['gt_boxes_1']
            complete_0 = in_dict['complete_0']
            complete_1 = in_dict['complete_1']            
        else:
            complete_0 = in_dict['complete']
            gt_boxes_0 = in_dict['gt_boxes']
        
        loss_dict = {}                
        pred_box = get_bbox_from_keypoints(ret_dict['coarse'], gt_boxes_0) # B 7        
        loss_dict['dims'] = self.loss_dims(gt_boxes_0[:,3:6].cuda(), pred_box[:,3:6]).mean()

        # ensemble loss        
        loss_dict['teacher_loss_0'], loss_dict['student_loss_0'] = self.get_student_teacher_loss(gt_boxes_0, ret_dict, branch_id=0)        
        
        # cd loss
        ds_complete_0 = misc.fps(complete_0, ret_dict['coarse'].shape[1])
        loss_dict['coarse'] = self.loss_coarse(ret_dict['coarse'], ds_complete_0)
        if in_dict['training']:
            ds_complete_1 = misc.fps(complete_1, ret_dict['coarse_1'].shape[1])
            loss_dict['coarse_1'] = self.loss_coarse1(ret_dict['coarse_1'], ds_complete_1)
            loss_dict['teacher_loss_1'], loss_dict['student_loss_1'] = self.get_student_teacher_loss(gt_boxes_1, ret_dict, branch_id=1)        
        
        return loss_dict
    
    def get_student_teacher_loss(self, gt_boxes, ret_dict, branch_id, t_weight=10, r_weight=1):
        pose_candidates = ret_dict[f'pose_candidates_{branch_id}']
        
        teacher_loss, best_branch_idx = compute_ensemble_loss(gt_boxes, pose_candidates)
        teacher_branch_output = pose_candidates[best_branch_idx,:,:] # B 9
        teacher_trans = teacher_branch_output[:,:3]
        teacher_rotm = compute_rotation_matrix_from_ortho6d(teacher_branch_output[:,3:9])                
        
        l_student_trans = self.loss_translation(ret_dict[f'reg_centre_{branch_id}'], teacher_trans).mean()
        l_student_rot = compute_geodesic_distance_from_two_matrices(ret_dict[f'reg_rot_{branch_id}'], teacher_rotm).mean()
        student_loss = t_weight*l_student_trans + r_weight*l_student_rot   
        
        return teacher_loss, student_loss
    
    def pose_ensemble(self, pc_mean, encoder_feat, branch_id):
        if branch_id == 0:
            pose_ensemble = self.pose_ensemble0
            distilled_pose_fc = self.distilled_pose0
        elif branch_id == 1:
            pose_ensemble = self.pose_ensemble1
            distilled_pose_fc = self.distilled_pose1

        pose_candidates = torch.stack([pe(encoder_feat) for pe in pose_ensemble], dim=0) # B 9 for each
        pose_candidates[:,:,:3] += pc_mean.unsqueeze(0) # N_branch B 3        
        distilled_pose = distilled_pose_fc(encoder_feat) # B 9
        trans = pc_mean.unsqueeze(1) + distilled_pose[:,:3].unsqueeze(1)
        rot6d = distilled_pose[:,3:9]
        rot_mat = compute_rotation_matrix_from_ortho6d(rot6d)
        return rot_mat, trans.squeeze(1), pose_candidates

    def forward(self, in_dict):        
        
        ret = {}        
        if in_dict['training']:
            pc_0 = in_dict['input_0'] # B 1024 3
            pc_1 = in_dict['input_1'] # B 1024 3
        else:
            pc_0 = in_dict['input']

        bs , n , _ = pc_0.shape 
        

        # Branch 0
        pc0_mean = pc_0.mean(dim=1)  
        z0 = self.encoder_0((pc_0 - pc0_mean.unsqueeze(1)).permute(0,2,1), n)
        shape = self.shape_decoder_0(z0).reshape(-1,self.number_coarse,3)
        ret['reg_rot_0'], ret['reg_centre_0'], ret['pose_candidates_0'] = self.pose_ensemble(pc0_mean, z0, branch_id=0)        
        
        pc0_vc = cn_to_vc_rt(shape, ret['reg_rot_0'], ret['reg_centre_0'])
        ret['coarse'] = pc0_vc.contiguous()

        if in_dict['training']:
            pc1_mean = pc_1.mean(dim=1)
            z1 = self.encoder_1((pc_1 - pc1_mean.unsqueeze(1)).permute(0,2,1), n)
            ret['reg_rot_1'], ret['reg_centre_1'], ret['pose_candidates_1'] = self.pose_ensemble(pc1_mean, z1, branch_id=1)      
            pc1_vc = cn_to_vc_rt(shape, ret['reg_rot_1'], ret['reg_centre_1'])  
            ret['coarse_1'] = pc1_vc.contiguous()        

        # # Siamese branch 0
        # pc0_mean = pc_0.mean(dim=1)       
        # feature_global_0 = self.encoder_0((pc_0 - pc0_mean.unsqueeze(1) ).permute(0,2,1), n)  # B 1024
        # ret['reg_rot_0'], ret['reg_centre_0'], ret['pose_candidates_0'] = self.pose_ensemble(pc0_mean, feature_global_0, branch_id=0)        
        
        # shape = self.shape_fc(feature_global_0).reshape(-1,self.number_coarse,3) # B coarse_pts 3
        # pc0_cn = cn_to_vc_rt(shape, ret['reg_rot_0'], ret['reg_centre_0'])
        # ret['coarse'] = pc0_cn.contiguous()
        
        # # Siamese branch 1                
        # if in_dict['training']:
        #     pc1_mean = pc_1.mean(dim=1)
        #     feature_global_1 = self.encoder_1((pc_1 - pc1_mean.unsqueeze(1)).permute(0,2,1), n)  # B 1024
        #     ret['reg_rot_1'], ret['reg_centre_1'], ret['pose_candidates_1'] = self.pose_ensemble(pc1_mean, feature_global_1, branch_id=1)                    
        #     pc1_cn = cn_to_vc_rt(shape, ret['reg_rot_1'], ret['reg_centre_1'])
        #     ret['coarse_1'] = pc1_cn.contiguous()        


        return ret