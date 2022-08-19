#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python3 setup.py install --user

# NOTE: For GRNet 

# Cubic Feature Sampling
cd $HOME/extensions/cubic_feature_sampling
python3 setup.py install --user

# Gridding & Gridding Reverse
cd $HOME/extensions/gridding
python3 setup.py install --user

# Gridding Loss
cd $HOME/extensions/gridding_loss
python3 setup.py install --user

