# hed batch processing script; modified from https://github.com/s9xie/hed/blob/master/examples/hed/hed-tutorial.ipynb
# step 1: download the hed repo: https://github.com/s9xie/hed
# step 2: download the models and protoxt, and put them under {caffe_root}/examples/hed/
# step 3: put this script under {caffe_root}/examples/hed/
# step 4: run the following script: 
#       python batch_hed.py --images_dir=/data/to/path/photos/ --hed_mat_dir=/data/to/path/hed_mat_files/
# the code sometimes crashes after computation is done. error looks like "check failed: ... driver shutting down". you can just kill the job. 
# for large images, it will produce gpu memory issue. therefore, you better resize the images before running this script. 
# step 5: run the matlab post-processing script "postprocesshed.m" 
import numpy as np
import scipy.misc
from PIL import Image
import scipy.io
import os
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='batch proccesing: photos->edges')
    parser.add_argument('--caffe_root', dest='caffe_root', help='caffe root', default='../../', type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', help='caffemodel', default='./hed_pretrained_bsds.caffemodel', type=str)
    parser.add_argument('--prototxt', dest='prototxt', help='caffe prototxt file', default='./deploy.prototxt', type=str)
    parser.add_argument('--images_dir', dest='images_dir', help='directory to store input photos', type=str)
    parser.add_argument('--hed_mat_dir', dest='hed_mat_dir', help='directory to store output hed edges in mat file',  type=str)
    parser.add_argument('--border', dest='border', help='padding border', type=int, default=128)
    parser.add_argument('--gpu_id', dest='gpu_id', help='gpu id', type=int, default=1)
    args = parser.parse_args()
    return args

args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))
# make sure that caffe is on the python path:  
caffe_root = args.caffe_root   # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import scipy.io as sio

if not os.path.exists(args.hed_mat_dir):
    print('create output directory %s' % args.hed_mat_dir)
    os.makedirs(args.hed_mat_dir)

# imglist = os.listdir(args.images_dir)
# nimgs = len(imglist)
# print('#images = %d' % nimgs)

full_image_paths = []
class_list = os.listdir(args.images_dir)
for i in range(len(class_list)):
    videos = os.listdir(os.path.join(args.images_dir, class_list[i]))
    for j in range(len(videos)):
        mat_path = os.path.join(args.hed_mat_dir, class_list[i], videos[j])
        os.makedirs(mat_path, exist_ok=True)
        frames = os.listdir(os.path.join(args.images_dir, class_list[i], videos[j]))
        for k in range(len(frames)):
            name, _ = os.path.splitext(frames[k])
            if '.jpg' in frames[k] and not os.path.exists(os.path.join(mat_path, name + '.mat')):
                full_image_paths.append(os.path.join(args.images_dir, class_list[i], videos[j], frames[k]))

caffe.set_mode_gpu()
caffe.set_device(args.gpu_id)
# load net
net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
# pad border
border = args.border

for i in range(len(full_image_paths)):
    im = Image.open(os.path.join(args.images_dir, full_image_paths[i]))

    in_ = np.array(im, dtype=np.float32)
    try:
        in_ = np.pad(in_, ((border, border), (border, border), (0, 0)), 'reflect')
    except ValueError:
        print('check {} please'.format(full_image_paths[i]))
        with open('batch_hed.log', mode='a') as f:
            f.write(full_image_paths[i])
        continue

    in_ = in_[:,:,0:3]
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2, 0, 1))
    # remove the following two lines if testing with cpu

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    fuse = net.blobs['sigmoid-fuse'].data[0][0, :, :]
    # get rid of the border
    correction_value = 36
    fuse = fuse[(border + correction_value):-(border - correction_value), (border + correction_value):-(border - correction_value)]
    # save hed file to the disk
    full_frame_name, _ = os.path.splitext(full_image_paths[i])
    full_video_name, frame_name = os.path.split(full_frame_name)
    full_class_name, video_name = os.path.split(full_video_name)
    _, class_name = os.path.split(full_class_name)
    save_name = os.path.join(args.hed_mat_dir, class_name, video_name, frame_name + '.mat')
    sio.savemat(save_name, {'predict':fuse})
    if i % 1000 == 0:
        print('[{}/{}]'.format(i, len(full_image_paths)))

print('END!!')

