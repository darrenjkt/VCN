import numpy as np
import torch
import torch.nn as nn
from utils.vis_utils import *
from utils import misc

def get_partial_mesh(partial_pc, complete_pc, k=30, surface_pts=1024):
    """
    For each point in the partial pc, we find the k nearest points in 
    the complete pc to keep. We take a set of all the points so there
    are no duplicates. 

    partial_pc (torch N 3): Input sparse point cloud 
    complete_pc (torch N 3): Dense complete surface/ coarse prediction
    k (int): Number of nearest points to sample

    Returns pc (N 3)

    """
    surface_idx = []
    partial_pc = torch.unique(partial_pc, dim=0)
    for i in range(len(partial_pc)):
        dist = torch.norm(complete_pc - partial_pc[i], dim=1, p=None)
        knn = dist.topk(k, largest=False)
        surface_idx.extend(knn[1].cpu().numpy())
        
    surface_idx = torch.tensor(list(set(surface_idx))).cuda()
    sel_complete = torch.index_select(complete_pc, dim=0, index=surface_idx)
    repeat_factor = int(np.ceil(surface_pts/sel_complete.shape[0]))
    sel_sampled = sel_complete.repeat(repeat_factor, 1)[:surface_pts,:]
    return sel_sampled

def get_partial_mesh_batch(batch_partial, batch_complete, k=30, surface_pts=1024):
    """
    Runs "get_partial_mesh" function but in batches.

    partial_pc (B N 3): Input sparse point cloud
    complete_pc (B N 3): Dense complete surface/ coarse prediction
    k (int): Number of nearest points to sample

    Returns pc (B N 3)
    """
    surfaces = [get_partial_mesh(bp, bc, k=k, surface_pts=surface_pts).unsqueeze(0) for bp, bc in zip(batch_partial, batch_complete)]
    return torch.cat(surfaces, dim=0)


def get_largest_cluster(pc, eps=0.4, min_points=1, istensor=False, total_pts=1024):
    """
    partial_pc (N 3): Input sparse point cloud 
    Returns pc (N 3)

    """
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
        istensor = True

    sel_pc = convert_to_o3dpcd(pc)
    labels = np.array(sel_pc.cluster_dbscan(eps=eps, min_points=min_points))
    y = np.bincount(labels[labels >= 0])
    value = np.argmax(y)
    most_points = np.argwhere(labels == value)
    f_pcd = sel_pc.select_by_index(most_points)
    ret_pc = np.asarray(f_pcd.points)
    ret_pc_sampled = np.tile(ret_pc, (int(np.ceil(total_pts/ret_pc.shape[0])), 1))[:total_pts,:]

    if istensor:
        return torch.from_numpy(ret_pc_sampled).float()
    else:
        return ret_pc_sampled

def get_largest_cluster_batch(pc, eps=0.4, min_points=1, total_pts=1024):
    """
    partial_pc (B N 3): Input sparse point cloud (Tensor)
    Returns pc (B N 3)

    """   
    clusters = [get_largest_cluster(p, eps=eps, min_points=min_points, total_pts=total_pts).unsqueeze(0) for p in pc]
    return torch.cat(clusters, dim=0)

def get_npatches(pc, n_patches, area_scaling=2):
    assert len(pc.shape) == 2, f"pc shape should be 2, was given len(pc.shape)={len(pc.shape)}"
    keypoints = misc.fps(pc.unsqueeze(0), n_patches).squeeze(0) 
    n_total = pc.shape[0]
    spp = int(n_total//n_patches) # samples per patch
    k =  spp * area_scaling
    surface_idx = []
    for i in range(len(keypoints)):
        dist = torch.norm(pc - keypoints[i], dim=1, p=None)
        knn = dist.topk(k, largest=False)
        surface_idx.append(knn[1].unsqueeze(0))

    complete_patches = []
    for sidx in surface_idx:
        sel_complete = torch.index_select(pc, dim=0, index=sidx.squeeze())    
        sel_sampled = misc.fps(sel_complete.unsqueeze(0), spp).squeeze(0)
        complete_patches.append(sel_sampled.unsqueeze(0))

    return torch.cat(complete_patches)

def get_npatches_batch(pc, n_patches):
    """
    pc (B N 3): Input complete cloud
    n_patches (int): How many patches

    Return (B n_patches N//n_patches 3)
    """
    npatch_list = [get_npatches(pc[i], n_patches=n_patches).unsqueeze(0) for i in range(len(pc))]
    return torch.cat(npatch_list, dim=0)
    