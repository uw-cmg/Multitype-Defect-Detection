#!/usr/bin/env python
"""

This is the checker program that checks whether two bounding boxes are matched.

It will output the mismatched ones and gives more detailed information later.


"""
import numpy as np
from skimage import io
#import matplotlib
#matplotlib.use('GTK3Cairo')
#import matplotlib.pyplot as plt
# import time
# import pandas as pd

#imgPath = "c.jpg"
gtPath = "gt_c.txt"
predPath = "pred_c.txt"
bbox_label_names = ('111', 'dot','100')


def bbox_iou(a, b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        a: (list of 4 numbers) [x1,y1,x2,y2]
        b: (list of 4 numbers) [x1,y1,x2,y2]
    Returns:
        iou: the value of the IoU of two bboxes

    """
    # (float) Small value to prevent division by zero
    epsilon = 1e-5
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

def bbox_area(bbox):
    """Calculate the area of a bounding boxes.
    Args:
        bbox: (array of 4 numbers) [x1,y1,x2,y2]
    Returns:
        area: the area of a bbox
    """
    area = (bbox[3] - bbox[1])*(bbox[2] - bbox[0])
    return area

def evaluate_twoBBox_by_iou_kinds(gtBBoxArr, predBBoxArr, bbox_label_names=('0'), threshold_IoU=0.5):
    """Calculate the matching status between two BBox lists contains
        different kinds of labels

    Args:
        gtBBoxArr:  numpy.ndarray of (R,5) of ground truth
        predBBoxArr:  numpy.ndarray of (R',5) of prediction
        bbox_label_names: tuples contains lists of class names which should be aggreed with
                          their order of labels in bbox txt
        threshold_IoU: the threshold value used two decide whether two bboxes overlapped

    Returns:
        correct: the number of correct predictions
        cls_error: the number of classification errors
        loc_error: the number of location errors
        confMatrix: the confusion matrix of classification errors
        area_loc_error_list: the pixel area of location error bbox
        gtNumDefects: the number of ground truth bbox for each type defects
        cls_error_list: the list of classification error list
                        each element is a pair of (pred_record, pred_lineNumber, gt_record,gt_lineNumber)
                        pred_record contains 6 items [label,x1,y1,x2,y2]
                        gt_record contains 6 items [label,x1,y1,x2,y2,lineNumber]
        loc_error_list: if it is a list of (pred_record, pred_lineNumber)
                        each pred_record contains 6 items [label,x1,y1,x2,y2]

    """

    correct = 0
    cls_error = 0
    loc_error = 0
    gtNumDefects = np.zeros(shape=(1, len(bbox_label_names)))
    confMatrix = np.zeros(shape=(len(bbox_label_names), len(bbox_label_names)))
    area_loc_error_list = list()
    cls_error_list = list()
    loc_error_list = list()

    # get ground truth class numbers
    for k in range(0, len(gtBBoxArr)):
        gtNumDefects[0, int(gtBBoxArr[k][0])] += 1

    # for each predicted bbox
    for i in range(0, len(predBBoxArr)):
        tmpIoU_list = []
        for j in range(0, len(gtBBoxArr)):
            tmpIoU_list.append(bbox_iou(predBBoxArr[i][1:], gtBBoxArr[j][1:]))
        # After go through all the gt bbox
        # check whether any IoU > threshold_IoU
        tmpIoU_list_loc_true = [x for x in tmpIoU_list if x >= threshold_IoU]
        if len(tmpIoU_list_loc_true) >= 1:
            # find the maximum index of this item this follows the NMS convention
            maxpos = tmpIoU_list.index(max(tmpIoU_list))
            # if label matched
            confMatrix[int(predBBoxArr[i][0])][int(gtBBoxArr[maxpos][0])] += 1
            if gtBBoxArr[maxpos][0] == predBBoxArr[i][0]:
                correct += 1
            else:
                cls_error += 1
                tmp = list()
                tmp.append(predBBoxArr[i].tolist())
                tmp.append(i+1)
                tmp.append(gtBBoxArr[maxpos].tolist())
                tmp.append(maxpos+1)
                cls_error_list.append(tmp)

        else:  # no IoU > threshold_IoU found
            loc_error += 1
            tmp = list()
            tmp.append(predBBoxArr[i].tolist())
            tmp.append(i + 1)
            loc_error_list.append(tmp)
            #area_loc_error_list.append((predBBoxArr[i][0], bbox_area(predBBoxArr[i][1:])))
    return correct, cls_error, loc_error, confMatrix, gtNumDefects,cls_error_list,loc_error_list

if __name__ == '__main__':
    print("Test")
    predArr = np.loadtxt(predPath, delimiter=',')
    gtArr = np.loadtxt(gtPath,delimiter=',')
    #img = io.imread(imgPath)

    correct, cls_error, loc_error, confMatrix, gtNumDefects,cls_error_list,loc_error_list = evaluate_twoBBox_by_iou_kinds(gtArr, predArr, bbox_label_names, threshold_IoU=0.5)

    print(correct)
    print(loc_error)
    print(cls_error)
    print(confMatrix)
    # print(area_loc_error_list)
    print(gtNumDefects)

    precision = 1.0 * correct / (loc_error + cls_error + correct)
    recall = 1.0 * correct / (np.sum(gtNumDefects))
    F1 = 2.0 * recall * precision / (recall + precision)

    print("============ Performance ==============")
    print("P : %f" % precision)
    print("R : %f" % recall)
    print("F1 : %f" % F1)
    print("============ Performance ==============")

    print(cls_error_list)
    print(loc_error_list)

    # csvFileName = "area.csv" + time.strftime("%Y%m%d_%H%M%S")
    # with open(csvFileName, 'w') as myfile:
    #     for i, (label, area) in enumerate(area_loc_error_list):
    #         myfile.write("%s,%s \n" % (label, area))

    # Plot Histogram of location error

    # areaDF = pd.DataFrame(columns=['classlabel', 'area'])
    #
    # for i, (classlabel, area) in enumerate(area_loc_error_list):
    #     areaDF.loc[i] = [classlabel, area]
    # #
    # histgramFileName = "Hist_" + time.strftime("%Y%m%d_%H%M%S")

    # fig = plt.figure(figsize=(15, 6))
    # fig, ax = plt.subplots(1, 2)
    # areaDF.hist(bins=50, ax=ax)
    # fig.savefig(histgramFileName)

    print("Done")




