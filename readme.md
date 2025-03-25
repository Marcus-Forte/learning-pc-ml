
# Resources

- `https://arxiv.org/abs/1612.00593`
- `https://github.com/charlesq34/pointnet`
- `https://github.com/yanx27/Pointnet_Pointnet2_pytorch`

# Datasets

- `http://buildingparser.stanford.edu/dataset.html` 
- `https://modelnet.cs.princeton.edu/`

# Usage

- Download dataset `modelnet40_ply_hdf5_2048.zip`


- Run: `python src/app.py --pretrained_c best_model.t7 --pretrained_s best_model.t7 --pointcloud caixa.xyz`


# Labels

SemSeg: `classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']`

Class: ?