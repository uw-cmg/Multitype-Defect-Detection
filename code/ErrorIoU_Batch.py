# print(os.getcwd())
# # import os
# # os.getcwd()
# os.chdir("..")
# os.chdir("..")
# print(os.getcwd())

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
import chainer
import math
from chainercv import utils
import matplotlib.pyplot as plt
import time


#load Data
root = './data/3Types/Data3TypesYminXminYmaxXmax9'
dataset = MultiDefectDetectionDataset(data_dir=root, split='train')
#dataset_test = MultiDefectDetectionDataset(data_dir=root, split='test')
dataset_test = MultiDefectDetectionDataset(data_dir=root, split='validation2')
dataset_valid = MultiDefectDetectionDataset(data_dir=root, split='validation2')
bbox_label_names = ('111', 'dot','100')

# DataSet Statistics
print('total number of training images: ', len(dataset))
print('total number of test images: ', len(dataset_test))
print('type of defects: ', bbox_label_names)

# predict figures using new methods
use_gpu = False

################### 0.80 F1 #########################
# # WithDA Model
# proposal_params = {'min_size': 4}
# model = FasterRCNNVGG16(n_fg_class=3,
#                         pretrained_model='./modelResults/snapshot_model_510000_20181001_DataSet6WithDA.npz',
#                         ratios=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], anchor_scales=[1, 4, 8, 16], min_size=1024,
#                         max_size=1024, proposal_creator_params=proposal_params)
#####################################################################

#Higher NMS
proposal_params = {'min_size': 4,'nms_thresh':0.8}
model = FasterRCNNVGG16(n_fg_class=3, pretrained_model='./modelResults/snapshot_model_510000_higherNMS.npz', ratios=[ 0.5, 1, 1.5, 2, 2.5, 4,8,16],anchor_scales=[1, 4, 8, 16], min_size=1024, max_size=1024,proposal_creator_params=proposal_params)

#####################################################################
# proposal_params = {'min_size': 8,'nms_thresh':0.4}
# model = FasterRCNNVGG16(n_fg_class=3,
#                         pretrained_model='./modelResults/snapshot_model_510000_20181001_DataSet6WithDA.npz',
#                         ratios=[ 0.5, 1, 1.5, 2, 2.5, 4,8,16], anchor_scales=[1, 4, 8, 16], min_size=1024,
#                         max_size=1024, proposal_creator_params=proposal_params)
#####################################################################

# Lower NMS
# 8min size
# proposal_params = {'min_size': 8,'nms_thresh':0.4}
# 0p2 NMS

# proposal_params = {'min_size': 4,'nms_thresh':0.2}
# model = FasterRCNNVGG16(n_fg_class=3, pretrained_model='../../modelResults/snapshot_model_366000_Shuffule_20190122.npz',
#                         ratios=[ 0.5, 1, 1.5, 2, 2.5, 4, 8,16],
#                         anchor_scales=[1, 4, 8, 16],
#                         min_size=1024, max_size=1024,
#                         proposal_creator_params=proposal_params)

if use_gpu:
    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()

bbox_label_names = ('111loop', 'dot', '100loop')

from utils.evaluation import evaluate_set_by_centroid

# Using Validation Set
#dataset_test_validation = MultiDefectDetectionDataset(data_dir=root, split='validatio

# print("Average recall ", sum(recalls)/len(recalls))
# print("Average precision ", sum(precisions)/len(precisions))
# r = sum(recalls)/len(recalls)
# p = sum(precisions)/len(precisions)
# print("Average F1", 2 * r * p / (r + p))

from utils.evaluation import evaluate_set_by_iou_kinds

thetaList = [0.001,0.005,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
IoUList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


for iou_i in IoUList:
    os.mkdir(str(iou_i))
    Plist = []
    Rlist = []
    F1list = []
    for theta_i in thetaList:
        print("Processing theta %s"%theta_i)
        correct, cls_error,loc_error,confMatrix,area_loc_error_list,gtNumDefects = evaluate_set_by_iou_kinds(model, dataset_test,bbox_label_names, threshold=theta_i, threshold_IoU = iou_i)
        statTXTname = str(iou_i) + "/stat_" + str(theta_i) + ".txt"
        with open(statTXTname, 'w') as statfile:
            statfile.write("correct prediction : ")
            statfile.write( str(correct))
            statfile.write("\n")

            statfile.write("location error : ")
            statfile.write( str(loc_error))
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

            precision = 1.0 * correct / (loc_error + cls_error + correct)
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
    # import matplotlib.pyplot as plt
    fignow = plt.figure()
    plt.plot( Rlist, Plist, marker='o', markerfacecolor='red', markersize=12, color='skyblue', linewidth=4, label='precision')
    #plt.plot( thetaList, Rlist, marker='o', markerfacecolor='green', markersize=12, color='skyblue', linewidth=4,label='recall')
    #plt.plot( thetaList, F1list, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,label='F1')
    plt.title("PR Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    fignow.savefig(str(iou_i)+"/PR.png", dpi = 300)

# print(correct)
# print(loc_error)
# print(cls_error)
# print(confMatrix)
# print(gtNumDefects)

#plt.hist(area_loc_area_list,bins=500)
#plt.savefig("newList.png")



# precision = 1.0 * correct / (loc_error + cls_error + correct)
# recall = 1.0 * correct/(np.sum(gtNumDefects))
# F1 = 2.0 * recall * precision / (recall + precision)

# print("============ Performance ==============")
# print("P : %f" % precision)
# print("R : %f" % recall)
# print("F1 : %f" % F1)
# print("============ Performance ==============")

# import csv
# csvFileName = "area.csv" + time.strftime("%Y%m%d_%H%M%S")
# with open(csvFileName, 'w') as myfile:
#     for i, (label, area) in enumerate(area_loc_error_list):
#         myfile.write("%s,%s \n" %(label, area))
#
# # Plot Histogram of location error
# import pandas as pd
# areaDF = pd.DataFrame(columns=['classlabel','area'])
#
# for i, (classlabel, area) in enumerate(area_loc_error_list):
#     areaDF.loc[i] = [classlabel, area]
#
# import matplotlib.pyplot as plt
#
# histgramFileName = "Hist_" + time.strftime("%Y%m%d_%H%M%S")
# fig = plt.figure(figsize=(15,6))
# fig, ax = plt.subplots(1,2)
# areaDF.hist(bins=50, ax=ax)
# fig.savefig(histgramFileName)

print("Done")