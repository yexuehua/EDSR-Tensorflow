import data
import argparse
from model import EDSR
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="./data/General-100")
parser.add_argument("--imgsize",default=96,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=16,type=int)
parser.add_argument("--savedir",default='saved_models')
parser.add_argument("--iterations",default=500,type=int)
args = parser.parse_args()
data.load_dataset(r'D:\python\EDSR-Tensorflow-master\dataset\med_dataset',args.imgsize)
if args.imgsize % args.scale != 0:
    print(f"Image size {args.imgsize} is not evenly divisible by scale {arg.scale}")
    exit()
down_size = args.imgsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,args.scale)
network.set_data_fn(data.get_batch,(args.batchsize,args.imgsize,down_size),data.get_test_set,(args.imgsize,down_size))

network.train(args.iterations,args.savedir)
