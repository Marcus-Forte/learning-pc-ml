
# Resources

- `https://arxiv.org/abs/1612.00593`
- `https://github.com/charlesq34/pointnet`
- `https://github.com/yanx27/Pointnet_Pointnet2_pytorch`

# Datasets

- `http://buildingparser.stanford.edu/dataset.html` 
- `https://modelnet.cs.princeton.edu/`

# Usage

- Download dataset `modelnet40_ply_hdf5_2048.zip`


- Run: `python src/app.py --pretrained_c pretrained/best_model_cls.t7`


# API

Generate proto files:
- `python -m grpc_tools.protoc -I./grpc --python_out=src/gen/ --grpc_python_out=src/gen/ --pyi_out=src/gen grpc/ai.proto`

# Labels

SemSeg: `classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']`
