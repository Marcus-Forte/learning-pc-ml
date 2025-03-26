import argparse
import os
import sys
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from concurrent import futures

from featModel.FeatModel import FeatModel

sys.path.append('.')

from gen import ai_pb2
from gen import ai_pb2_grpc

import grpc

class AIServicesServicerImpl(ai_pb2_grpc.AIServicesServicer):
    def __init__(self, model, testset):
        self.model = model
        self.testset = testset

    def classifyAI(self, request, context):
        filename = request.filepath
        print(f'Classify AI called: processing {filename}')
        
        label = predict_c(torch.device('cuda:0'), self.model, self.testset, filename)
        response = ai_pb2.ClassifyResponse(label=label)
        
        return response

def load_xyz(filename):
    points = [[]]
    with open(f"{filename}",'r') as f:
        while True:
            line = f.readline()
            if line == "":
                break
            xyz = line.replace('\n', '').split(' ')
            xyz = xyz[0:3]
            point = []
            for el in xyz:
                point.append(float(el))
            
            points[0].append(point)
    return torch.tensor(points)

def save_xyz(points, filename):
    with open(f"{filename}.xyz",'w') as f:
        for point in points[0]:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
    sys.path.append(os.path.join(BASE_DIR, os.pardir))
    os.chdir(os.path.join(BASE_DIR, os.pardir))
    
from learning3d.models import PointNet
from learning3d.models import Classifier, Segmentation
from learning3d.data_utils import ClassificationData, ModelNet40Data

def predict_c(device, model, testset, pointcloud_file):
    model.eval()
    points = load_xyz(pointcloud_file)
    points = points.to(device)
    # print(points)
    output = model(points)
    print(output)
    label = torch.argmax(output[0]).item()
    print(f"label: {label} := {testset.get_shape(label)}")
    return str(testset.get_shape(label))

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--dataset_path', type=str, default='ModelNet40',
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')

    # settings for PointNet
    parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
                        help='train pointnet (default: tune)')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--emb_dims', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for on training
    parser.add_argument('--pretrained_c', default='learning3d/pretrained/exp_classifier/models/best_model.t7', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--pretrained_s', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')    
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')
    
    args = parser.parse_args()
    return args

def main():
    args = options()
    args.dataset_path = "./" #os.path.join(os.getcwd(), os.pardir, os.pardir, 'ModelNet40', 'ModelNet40')
    
    data = ModelNet40Data(train=False, root_dir=args.dataset_path)
    testset = ClassificationData(data)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # Create PointConv Model.
    # PointConv = create_pointconv(classifier=False, pretrained=None)
    # ptconv = PointConv(emb_dims=args.emb_dims, classifier=True, pretrained=None)

    ptnet = PointNet(emb_dims=args.emb_dims, use_bn=True, global_feat=True)
    segptnet = FeatModel(1024, use_bn=False, global_feat=False)

    model = Classifier(feature_model=ptnet) # select model
    segmodel = Segmentation(feature_model=segptnet) # select model

    if args.pretrained_c:
        assert os.path.isfile(args.pretrained_c)
        model.load_state_dict(torch.load(args.pretrained_c, map_location='cpu'))
        model.to(args.device)

    if args.pretrained_s:
        segmodel.load_state_dict(torch.load(args.pretrained_s, map_location='cpu')) 
        segmodel.to(args.device)

    # predict_c(args.device, model, testset, args.pointcloud)
    print('Models loaded successfully.')
    # run server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ai_pb2_grpc.add_AIServicesServicer_to_server(AIServicesServicerImpl(model, testset), server)
    server.add_insecure_port('[::]:50051')
    server.start()

    print('gRPC server started on port 50051')
    server.wait_for_termination()
    
if __name__ == '__main__':
    main()