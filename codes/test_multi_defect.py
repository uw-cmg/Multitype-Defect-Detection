from utils import imageUtils
from utils import postProcessing
from utils import MultiDefectDetectionDataset
from utils import evaluation
from utils import visualization
import numpy as np
import os
from chainercv.links import FasterRCNNVGG16
from chainercv.visualizations import vis_bbox
from chainercv.utils import write_image
from utils.evaluation import evaluate_set_by_iou_kinds
import chainer
import math
from chainercv import utils
import matplotlib.pyplot as plt
import time


#load Data
root = '../data/3Types/Data3TypesYminXminYmaxXmax'
#print(os.listdir(root))
dataset_test = MultiDefectDetectionDataset(data_dir=root, split='validation2')

bbox_label_names = ('111', 'dot','100')

# DataSet Statistics
print('total number of training images: ', len(dataset))
print('total number of test images: ', len(dataset_test))
print('type of defects: ', bbox_label_names)

# predict figures using new methods
use_gpu = False#True

proposal_params = {'min_size': 8,'nms_thresh': 0.5}
model = FasterRCNNVGG16(n_fg_class=3, pretrained_model='../BestWeight.npz', ratios=[ 0.5, 1, 1.5, 2, 2.5, 4,8,16],anchor_scales=[1, 4, 8, 16], min_size=1024, max_size=1024,proposal_creator_params=proposal_params)

if use_gpu:
    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()

bbox_label_names = ('111loop', 'dot', '100loop')

thetaList = [0.001,0.005,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
IoUList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


for iou_i in IoUList:
    os.mkdir(str(iou_i))
    Plist = []
    Rlist = []
    F1list = []
    for theta_i in thetaList:
        print("Processing theta %s"%theta_i)
        correct, cls_error, confMatrix, gtNumDefects, predNumDefects = evaluate_set_by_iou_kinds(model, dataset_test,bbox_label_names, threshold=theta_i, threshold_IoU = iou_i)
        statTXTname = str(iou_i) + "/stat_" + str(theta_i) + ".txt"
        with open(statTXTname, 'w') as statfile:
            statfile.write("correct prediction : ")
            statfile.write( str(correct))
            statfile.write("\n")

            statfile.write("classification error : ")
            statfile.write( str(cls_error))
            statfile.write("\n")

            statfile.write("confusion matrix : \n")
            statfile.write(str(confMatrix))
            statfile.write("\n")

            statfile.write("ground truth data : ")
            statfile.write(str(gtNumDefects))
            statfile.write("\n")

            statfile.write("prediction data : ")
            statfile.write(str(predNumDefects))
            statfile.write("\n")

            precision = 1.0 * correct / (np.sum(predNumDefects))
            recall = 1.0 * correct / (np.sum(gtNumDefects))
            F1 = 2.0 * recall * precision / (recall + precision)

            statfile.write("============ Performance ==============\n")

            statfile.write("P : %f \n" % precision)
            statfile.write("R : %f \n" % recall)
            statfile.write("F1 : %f \n" % F1)

            statfile.write("============ Performance ==============")

            Plist.append(precision)
            Rlist.append(recall)
            F1list.append(F1)


    fignow = plt.figure()
    plt.plot( thetaList, Plist, marker='o', markerfacecolor='red', markersize=12, color='skyblue', linewidth=4, label='precision')
    plt.plot( thetaList, Rlist, marker='o', markerfacecolor='green', markersize=12, color='skyblue', linewidth=4,label='recall')
    plt.plot( thetaList, F1list, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,label='F1')
    plt.title("Performance with different threshold")
    plt.xlabel('threshold value')
    fignow.savefig(str(iou_i)+"/Summary.png", dpi = 300)


    # Plot PR curve
    # Plotting Results
    fignow = plt.figure()
    plt.plot( Rlist, Plist, marker='o', markerfacecolor='red', markersize=12, color='skyblue', linewidth=4, label='precision')
    plt.title("PR Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    fignow.savefig(str(iou_i)+"/PR.png", dpi = 300)

print("Done")
