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


def evaluate_set_by_centroid(model, dataset, threshold=0.5, use_gpu=True):
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
