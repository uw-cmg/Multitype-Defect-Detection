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
import shutil

#load Data
root = '../data/3Types/Data3TypesYminXminYmaxXmax5'
dataset = MultiDefectDetectionDataset(data_dir=root, split='train')
dataset_test = MultiDefectDetectionDataset(data_dir=root, split='test')
dataset_valid = MultiDefectDetectionDataset(data_dir=root, split='validation')
bbox_label_names = ('111', 'dot','100')

# DataSet Statistics
print('total number of training images: ', len(dataset))
print('total number of test images: ', len(dataset_test))
print('type of defects: ', bbox_label_names)

# predict figures using new methods
use_gpu = False
proposal_params = {'min_size': 8}
model = FasterRCNNVGG16(n_fg_class=3,
                        pretrained_model='../modelResults/snapshot_model_510000_20181001_DataSet6WithDA.npz',
                        ratios=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], anchor_scales=[1, 4, 8, 16], min_size=1024,
                        max_size=1024, proposal_creator_params=proposal_params)

if use_gpu:
    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()

bbox_label_names = ('111loop', 'dot', '100loop')

from utils.evaluation import evaluate_set_by_centroid


# plot both original image and gt image
imgInx = 0
img, gt_bbs, gt_lbs = dataset_valid[imgInx]
dataset_valid.copy_example_image(imgInx)
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(1, 1, 1)

img = img.transpose((1, 2, 0))[:, :, 0]
ax1.imshow(img, cmap='gray')
figName = "original_" + time.strftime("%Y%m%d_%H%M%S") + ".jpg"
fig.savefig(figName)

from utils.outputUtil import output_gt_bbox
gt_bboxes = gt_bbs.tolist()
gt_labels = gt_lbs.tolist()

output_gt_bbox(gt_bboxes,gt_labels,"gt")

# re-read data
img, gt_bbs, gt_lbs = dataset_valid[imgInx]
model.score_thresh = 0.25
pred_bbs, pred_lbs, pred_scores = model.predict([img])

pred_bboxes = pred_bbs[0].tolist()
pred_labels = pred_lbs[0]

output_gt_bbox(pred_bboxes,pred_labels,"pred")


print("Done")

# print("Average recall ", sum(recalls)/len(recalls))
# print("Average precision ", sum(precisions)/len(precisions))
# r = sum(recalls)/len(recalls)
# p = sum(precisions)/len(precisions)
# print("Average F1", 2 * r * p / (r + p))



# correct, cls_error,loc_error,confMatrix,area_loc_error_list,gtNumDefects = evaluate_set_by_iou_kinds(model,dataset_test_validation,bbox_label_names, threshold=0.25, threshold_IoU = 0.9)
#
#
#
# precision = 1.0 * correct / (loc_error + cls_error + correct)
# recall = 1.0 * correct/(np.sum(gtNumDefects))
# F1 = 2.0 * recall * precision / (recall + precision)
#
# print("============ Performance ==============")
# print("P : %f" % precision)
# print("R : %f" % recall)
# print("F1 : %f" % F1)
# print("============ Performance ==============")
#
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
