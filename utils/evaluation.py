import numpy as np
from .imageUtils import get_bbox_sz
from .postProcessing import img_ellipse_fitting
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 14})


def compute_score_by_centroid(pred_bbox, gt_bbox, tot=(20, 20)):
    pred_c = bbox2centroid(pred_bbox)
    gt_c = bbox2centroid(gt_bbox)
    diffs = abs(pred_c[:, None] - gt_c)
    x1, x2 = np.nonzero((diffs < tot).all(2))
    precision = np.unique(x1).shape[0]/pred_bbox.shape[0]
    recall = np.unique(x2).shape[0]/gt_bbox.shape[0]
    return recall, precision


def bbox2centroid(bboxes):
    return np.column_stack(((bboxes[:, 0] + bboxes[:, 2])/2, (bboxes[:, 1] + bboxes[:, 3])/2))


def evaluate_set_by_centroid(model, dataset, threshold=0.5):
    model.score_thresh = threshold
    recall_list = []
    precision_list = []
    for instance in dataset:
        img, gt_bbox, _ = instance
        pred_bbox, _, _ = model.predict([img])
        recall, precision = compute_score_by_centroid(pred_bbox[0], gt_bbox)
        recall_list.append(recall)
        precision_list.append(precision)
    return recall_list, precision_list

def bbox_iou(a, b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        a: (list of 4 numbers) [y1,x1,y2,x2]
        b: (list of 4 numbers) [y1,x1,y2,x2]
    Returns:
        iou: the value of the IoU of two bboxes

    """
    # (float) Small value to prevent division by zero
    epsilon = 1e-5
    # COORDINATES OF THE INTERSECTION BOX
    y1 = max(a[0], b[0])
    x1 = max(a[1], b[1])
    y2 = min(a[2], b[2])
    x2 = min(a[3], b[3])

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

def evaluate_set_by_iou_kinds(model, dataset,bbox_label_names = ('0'), threshold=0.5,threshold_IoU = 0.5):
    """Calculate the performance of the model by IoU metrics and different kinds of labels
    
    Args:
        model: the Faster R-CNN model
        dastaset: testing dataset
        threshold: the threshold value used by Faster R-CNN
    Returns:
        

    """
    model.score_thresh = threshold
    correct = 0
    cls_error = 0
    loc_error = 0
    gtNumDefects = np.zeros(shape=(1, len(bbox_label_names)))
    confMatrix= np.zeros(shape=(len(bbox_label_names), len(bbox_label_names)))
    cls_error_size_list = list()
    for instance in dataset:
        img, gt_bboxes, gt_labels = instance
        pred_bboxes, pred_labels, pred_scores = model.predict([img])
        #print(pred_scores)
        pred_bboxes = pred_bboxes[0].tolist()
        gt_bboxes = gt_bboxes.tolist()
        #pred_labels = pred_labels.tolist()
        gt_labels = gt_labels.tolist()
        # for each predicted bbox
        for i in range(0,len(pred_bboxes)):
            tmpIoU_list = []
            # go through all the gt bbox
            for j in range(0,len(gt_bboxes)):
                tmpIoU_list.append(bbox_iou(pred_bboxes[i],gt_bboxes[j]))
            # After go through all the gt bbox
            # check whether any IoU > threshold_IoU
            tmpIoU_list_loc_true = [ x for x in tmpIoU_list if x >= threshold_IoU]
            if len(tmpIoU_list_loc_true) >= 1:
                # find the maximum index of this item this follows the NMS convention
                maxpos = tmpIoU_list.index(max(tmpIoU_list))
                # if label matched
                confMatrix[pred_labels[0][i]][gt_labels[maxpos]] += 1
                if gt_labels[maxpos] == pred_labels[0][i]:
                    correct += 1
                else:
                    cls_error += 1
            else: # no IoU > threshold_IoU found
                loc_error += 1
    return correct, cls_error,loc_error,confMatrix


def analyze_and_fitting(model, dataset, threshold=0.5, use_gpu=True):
    model.score_thresh = threshold
    for instance in dataset:
        img, gt_bbox, _ = instance
        pred_bbox, _, _ = model.predict([img])
        img_ellipse_fitting(img, pred_bbox[0])


def evaluate_set_by_defect_size(model, dataset, threshold=0.5, num_bins=20, size_range=(20, 120), use_gpu=True):
    min_sz = size_range[0]
    max_sz = size_range[1]
    num_bins = num_bins
    splits = [0] + list(np.linspace(min_sz, max_sz, num=num_bins)) + [np.inf]
    num_splits = len(splits) - 1
    tot_p = [0] * num_splits
    tot_g = [0] * num_splits
    tp_p = [0] * num_splits
    tp_g = [0] * num_splits

    model.score_thresh = threshold

    for instance in dataset:
        img, gt_bbox, _ = instance
        pred_bbox, _, _ = model.predict([img])
        # TODO: Currently We use bounding box to represent the size for speed considerations.
        # Later we will use actual size. (i.e. rewrite get_bbox_sz method)
        gt_sz = get_bbox_sz(gt_bbox)
        pred_sz = get_bbox_sz(pred_bbox[0])
        recall_index, precision_index = compute_score_detail_by_centroid(pred_bbox[0], gt_bbox)
        gt_tp_sz = gt_sz[recall_index]
        pred_tp_sz = pred_sz[precision_index]

        for i in range(num_splits):
            tp_g[i] += np.sum(np.all([gt_tp_sz < splits[i+1], gt_tp_sz >= splits[i]], axis=0))
            tp_p[i] += np.sum(np.all([pred_tp_sz < splits[i+1], pred_tp_sz >= splits[i]], axis=0))
            tot_g[i] += np.sum(np.all([gt_sz < splits[i+1], gt_sz >= splits[i]], axis=0))
            tot_p[i] += np.sum(np.all([pred_sz < splits[i+1], pred_sz >= splits[i]], axis=0))

    return (tp_g, tot_g), (tp_p, tot_p)


def compute_score_detail_by_centroid(pred_bbox, gt_bbox, tot=(20, 20)):
    pred_c = bbox2centroid(pred_bbox)
    gt_c = bbox2centroid(gt_bbox)
    diffs = abs(pred_c[:, None] - gt_c)
    x1, x2 = np.nonzero((diffs < tot).all(2))
    return np.unique(x2), np.unique(x1)


def pr_plot_by_size(values, label='score', num_bins=20, size_range=(20, 120)):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    min_sz = size_range[0]
    max_sz = size_range[1]
    num_bins = num_bins
    step = (max_sz-min_sz)/(num_bins-1)
    true_splits = [min_sz-step] + list(np.linspace(min_sz, max_sz, num=num_bins))
    columns = np.array(true_splits) + step / 2
    width = 0.7*step
    ax1.bar(columns, values[1], width, color='b', label='total number')
    ax1.bar(columns, values[0], width, color='r', label='correct prediction')
    ax1.set_xlabel('size of loops')
    ax1.set_ylabel('number of loops')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xticks(np.linspace(min_sz, max_sz, num=num_bins), rotation=30)

    p_values = cal_pr_values(values)
    ax2 = ax1.twinx()
    ax2.plot(columns, p_values, 'go--')
    ax2.set_ylabel(label, color='g')


def size_distribution_comparison(precision_scores, recall_scores, labels=('prediction', 'ground truth'),
                                 num_bins=20, size_range=(20, 120)):
    min_sz = size_range[0]
    max_sz = size_range[1]
    num_bins = num_bins
    step = (max_sz - min_sz) / (num_bins - 1)
    true_splits = [min_sz - step] + list(np.linspace(min_sz, max_sz, num=num_bins))
    columns = np.array(true_splits) + step / 2
    width = 0.3 * step

    fig, ax1 = plt.subplots(figsize=(12, 8))
    p1 = ax1.bar(columns - width / 2, precision_scores[1], width, color='b', label=labels[0]+' counts')
    p2 = ax1.bar(columns + width / 2, recall_scores[1], width, color='g', label=labels[1]+' counts')
    plt.xticks(np.linspace(min_sz, max_sz, num=num_bins), rotation=30)
    ax1.set_xlabel('size of loops')
    ax1.set_ylabel('number of loops')

    ax2 = ax1.twinx()
    p_values = cal_pr_values(precision_scores)
    r_values = cal_pr_values(recall_scores)

    p3 = ax2.plot(columns, p_values, 'ro--', label='precision')
    p4 = ax2.plot(columns, r_values, 'co--', label='recall')
    ax2.set_ylabel('score')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.04, 1), loc="upper left")


def cal_pr_values(pr_list):
    p_values = []
    for i in range(len(pr_list[0])):
        if pr_list[1][i] == 0:
            p_values.append(1)
            continue
        p_values.append(pr_list[0][i]/pr_list[1][i])
    return p_values
