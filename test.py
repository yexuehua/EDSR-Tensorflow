from model import EDSR
import scipy.misc
import numpy as np
import argparse
import data
import os
import pydicom
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")
parser.add_argument("--image")
args = parser.parse_args()
if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)
down_size = args.imgsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume(args.savedir)
# if args.image:
# 	x = scipy.misc.imread(args.image)
# else:
# 	print("No image argument given")

#x = scipy.misc.imread("lungCT.bmp")
#print(x.shape)
x = pydicom.read_file('16.DCM')
inputs = x.pixel_array
inputs = cv2.cvtColor(inputs,cv2.COLOR_GRAY2BGR)
outputs = network.predict(inputs)
#print(outputs.shape)
# outputs = np.resize(inputs,[3,256*2,224*2])
# outputs = outputs[0]
# outputs.astype(int)
#x.PixelData = outputs.tobytes()
#x.Rows,x.Columns = outputs.shape
#print(x.pixel_array.shape)
#x.save_as('out_16.DCM')
if args.image:
	scipy.misc.imsave('in16.bmp',inputs)
	scipy.misc.imsave('out16.bmp',outputs)

scipy.misc.imsave('in16.bmp',inputs)
scipy.misc.imsave('out16.bmp',outputs)