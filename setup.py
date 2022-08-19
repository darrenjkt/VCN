from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# pip install -e .
# using setup.py develop somehow fails at "open3d==0.14.1" saying it can't find module

setup(name='partialsc',
      version='0.1',
      description='Complete the surface of incomplete lidar point clouds',      
      author='Darren Tsai',
      author_email='d.tsai@acfr.usyd.edu.au',
      license='MIT',
      packages=find_packages(exclude=['scripts','experiments']),
      cmdclass={'build_ext': BuildExtension},
      install_requires=[  
        'argparse',
        'easydict',
        'h5py',
        'matplotlib',
        'numpy',    
        'opencv-python',
        'pyyaml',
        'scipy',
        'tensorboardX',
        'timm==0.4.5 ',
        'tqdm>=4.51.0',
        'open3d==0.14.1', 
        'transforms3d'    
      ],
      ext_modules=[
          CUDAExtension(
            name='chamfer', 
            sources=[
              'extensions/chamfer_dist/chamfer_cuda.cpp',
              'extensions/chamfer_dist/chamfer.cu',
          ]),
          CUDAExtension(
            name='iou3d_nms_cuda', 
            sources=[
              'extensions/iou3d_nms/src/iou3d_cpu.cpp',
              'extensions/iou3d_nms/src/iou3d_nms_api.cpp',
              'extensions/iou3d_nms/src/iou3d_nms.cpp',
              'extensions/iou3d_nms/src/iou3d_nms_kernel.cu',
          ]),
      ])