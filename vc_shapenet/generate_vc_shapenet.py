"""
This code generates the surface view-centric dataset. We utilize the 
positions of the bounding boxes given by the Waymo dataset to place our
virtual vehicles in the scene. We perform raycasting on the scene to obtain virtual vehicles with 
realistic occlusions from real-world scenarios. 

Each vehicle is scaled to the size of the waymo bounding box for the 
class "vehicle". For Waymo, vehicle includes construction vehicles, trucks
and buses. To avoid our virtual car being scaled to obscure dimensions,
we filter out the construction vehicles, trucks and buses using their 
aspect ratio. 

"""

from dataset_functions import *

if __name__ == "__main__":
    
    print('Loading Waymo infos')
    infos_path = '/PoinTr/data/waymo/infos_openpcdetv0.3.0/waymo_infos_train.pkl'
    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)

    data_dir = '/PoinTr/data/shapenet/ShapeNetCore.v2/02958343'
    with open(data_dir + '/ignore_models.txt','r') as f:
        ignore = [lines.split('\n')[0] for lines in f.readlines()]
        
    # Get model ids (there's a lot of impure car pointclouds in shapenet. Also we ignore trucks/buses)
    model_glob = glob.glob(f'{data_dir}/*/')
    models = set([model.split('/')[-2] for model in model_glob])
    ignore_combined = set(ignore)
    models.difference_update(ignore_combined)
    models = list(models)
    
    print(f'Filtered model ids has {len(models)} models')
    frames = get_frames(infos)
    
    dataset_name = 'vc-surface'    
    generate_dataset(data_dir, frames, models[:int(len(models)*0.9)], 
                     dataset_split='train', dataset_name=dataset_name,
                     min_pts=30, max_pts=50000, 
                     nviews=20, npoints_complete=16384)
    
    generate_dataset(data_dir, frames, models[int(len(models)*0.9):], 
                     dataset_split='val', dataset_name=dataset_name,
                     min_pts=30, max_pts=50000, 
                     nviews=20, npoints_complete=16384)
