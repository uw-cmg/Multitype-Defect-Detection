import matplotlib.pyplot as plt
import numpy as np


def cropImage(img, bboxes, expand=True):
    """crop images by the given bounding boxes.

    Args:
        img (numpy.ndarray): image in CHW format
        bboxes (numpy.ndarray): bounding boxes in the format specified by chainerCV
        expand (bool): whether to expand the bounding boxes or not

    Returns:
        a batch of cropped image in CHW format
        The image is in CHW format and its color channel is ordered in
        RGB.

    Return type: list

    """

    if expand:
        _, H, W = img.shape
        bboxes = expand_bbox(bboxes, H, W)

    subimages = list()
    for bbox in bboxes:
        bbox = bbox.astype(np.int)
        subimages.append(img[:, bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1])
    return subimages, bboxes


def expand_bbox(bbox, H, W):
    """
    expand the bounding box within the range of height and width of the image
    :param bbox: numpy.ndarray bounding box N by 4
    :param H: int Height of the image
    :param W: int Width of the image
    :return: numpy.ndarray expanded bounding box
    """
    b_height = 0.15*(bbox[:, 2] - bbox[:, 0])
    b_width = 0.15*(bbox[:, 3] - bbox[:, 1])
    b_height[b_height < 7] = 7
    b_width[b_width < 7] = 7
    adjust = np.array((-b_height, -b_width, b_height, b_width)).transpose()
    new_bbox = bbox + adjust
    new_bbox[new_bbox < 0] = 0
    new_bbox[new_bbox[:, 2] >= H, 2] = H - 1
    new_bbox[new_bbox[:, 3] >= W, 3] = W - 1

    return new_bbox


def showImage(img):
    """
    :param img (numpy.ndarray): image in CHW format
    :return: plot the red channel in grayscale color map

    """
    plt.imshow(img.transpose((1, 2, 0))[:, :, 0], cmap='gray')


def get_bbox_sz(bbox):
    """
    return the size of the bounding boxes
    :param bbox: numpy.ndarray bounding box N by 4
    :return: numpy.ndarray bounding box size, (width+length)/2
    """
    return (bbox[:, 2]+bbox[:, 3] - bbox[:, 0] - bbox[:, 1]) / 2

