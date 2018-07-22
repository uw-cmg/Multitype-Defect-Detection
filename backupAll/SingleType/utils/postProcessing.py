from .imageUtils import cropImage
from skimage import exposure, morphology, measure, draw
import numpy as np
import matplotlib.pyplot as plt
import math


def watershed_image(img):
    """
    use watershed flooding algorithm to extract the loop contour
    :param img: type(numpy.ndarray) image in CHW format
    :return: type(numpy.ndarray) image in HW format
    """
    img_gray = img[1,:,:]
    h, w = img_gray.shape
    img1 = exposure.equalize_hist(img_gray)
    # invert the image
    img2 = np.max(img1) - img1
    inner = np.zeros((h, w), np.bool)
    centroid = [round(a) for a in findCentroid(img2)]
    inner[centroid[0], centroid[1]] = 1
    min_size = round((h + w) / 20)
    kernel = morphology.disk(min_size)
    inner = morphology.dilation(inner, kernel)

    out = np.zeros((h,w), np.bool)
    out[0, 0] = 1
    out[h - 1, 0] = 1
    out[0, w - 1] = 1
    out[h - 1, w - 1] = 1
    out = morphology.dilation(out, kernel)
    out[0, :] = 1
    out[h - 1, :] = 1
    out[:, w - 1] = 1
    out[:, 0] = 1

    markers = np.zeros((h, w), np.int)
    markers[inner] = 2
    markers[out] = 1

    labels = morphology.watershed(img2, markers)

    return labels

def findCentroid(img):
    """
    find the centroid position of a image by weighted method
    :param img: (numpy.ndarray) image in HW format
    :return: (tuple) (y,x) coordinates of the centroid
    """
    h, w = img.shape
    # TODO: add weighted method later
    return h/2, w/2


def flood_fitting(img):
    """
    Use watershed flooding algorithm and regional property analysis
    to output the fitted ellipse parameters
    :param img: (numpy.ndarray) image in CHW format
    :return: region property, where property can be accessed through attributes
            example:
            area, bbox, centroid, major_axis_length, minor_axis_length, orientation
    """
    labels = watershed_image(img)
    results = measure.regionprops(labels - 1)
    sorted(results, key=lambda k: k['area'],reverse=True)
    # return the one with largest area
    return results[0]

def show_fitted_ellipse(img):
    """
    Show fitted ellipse on the image
    :param img: img in CHW format
    :return: plot ellipse on top of the image
    """
    region1 = flood_fitting(img)
    rr, cc = draw.ellipse_perimeter(int(region1['centroid'][0]), int(region1['centroid'][1]),
                                    int(region1['minor_axis_length'] / 2),
                                    int(region1['major_axis_length'] / 2), -region1['orientation'], img.shape[1:])
    plt.imshow(img[1,:,:], cmap='gray')
    plt.plot(cc, rr, '.')


def img_ellipse_fitting(img, bboxes):
    subimages, bboxes = cropImage(img, bboxes)
    y_points = np.array([])
    x_points = np.array([])
    for subim, bbox in zip(subimages, bboxes):
        region1 = flood_fitting(subim)
        result = (int(region1['centroid'][0]+bbox[0]), int(region1['centroid'][1]+bbox[1]),
                  int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                  -region1['orientation'])
        rr,cc = draw.ellipse_perimeter(*result)
        y_points = np.concatenate((y_points,rr))
        x_points = np.concatenate((x_points,cc))
    fig = plt.figure(figsize=(10,10))
    plt.imshow(img[0,:,:], cmap='gray')
    plt.scatter(x_points,y_points,s=(1*72./fig.dpi)**2,alpha=0.5)

def img_ellipse_fitting_area(img, bboxes):
    subimages, bboxes = cropImage(img, bboxes)
    ellipse_info_list = list()
    for subim, bbox in zip(subimages, bboxes):
        region1 = flood_fitting(subim)
        result = (int(region1['centroid'][0]+bbox[0]), int(region1['centroid'][1]+bbox[1]),
                  int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                  -region1['orientation'])
        ellipse_info_list.append(result)
    area = list()
    for item in enumerate(ellipse_info_list):
        area.append(math.pi * item[1][1] * item[1][2])
    plt.hist(area, bins=50)
    plt.xlabel('Area of ellipse')
    plt.ylabel('Frequency')
    plt.title(r'$\mathrm{Distribution\  of\  Ellipse\ Area}$')
    fig = plt.figure(figsize=(10, 10))
    plt.show()