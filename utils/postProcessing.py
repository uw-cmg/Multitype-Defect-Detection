from .imageUtils import cropImage
from .FitEllipse import FitEllipseAndParameters
from skimage import exposure, morphology, measure, draw
from skimage.filters import threshold_yen, threshold_minimum, threshold_otsu, gaussian, threshold_adaptive
from skimage.measure import label,find_contours
from skimage.filters import rank
<<<<<<< HEAD
from skimage.filters import median
from scipy import ndimage
from skimage.morphology import disk
from .visualization import vis_image
from skimage import exposure, morphology, measure, draw
from skimage.morphology import remove_small_objects,closing
from matplotlib.patches import Ellipse
# from utils.postProcessing import img_ellipse_fitting, flood_fitting, binary_threshold_fitting
import os
=======
from skimage.morphology import disk
from .visualization import vis_image
from utils.imageUtils import cropImage
from skimage import exposure, morphology, measure, draw
from skimage.morphology import remove_small_objects,closing
# from utils.postProcessing import img_ellipse_fitting, flood_fitting, binary_threshold_fitting
>>>>>>> 11523272183bfa4166cae2d6e8f88181519fa603
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import cv2


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
    #print(centroid)
    inner[ int(centroid[0]), int(centroid[1])] = 1
    min_size = round((h + w) / 18)
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

<<<<<<< HEAD
# def watershed_image_100_openCV(img):
#     """
#     use watershed flooding algorithm to extract the loop contour
#     :param img: type(numpy.ndarray) image in CHW format
#     :return: type(numpy.ndarray) image in HW format
#     """
#     img_gray = img[1,:,:]
#     img = cv2.medianBlur(img, 25)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(gray,80,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C )
#     # noise removal
#     kernel = np.ones((4,4),np.uint8)
#     opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)
#     # sure background area
#     sure_bg = cv2.dilate(opening,kernel,iterations=3)
#     plt.imshow(sure_bg)
#     # Finding sure foreground area
#     dist_transform = cv2.distanceTransform(opening,1,5)
#     ret, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(),255,0)
#     # Finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg,sure_fg)
#     # Marker labelling
#     ret, markers = cv2.connectedComponents(sure_fg)
#     # Add one to all labels so that sure background is not 0, but 1
#     markers = markers+1
#     # Now, mark the region of unknown with zero
#     markers[unknown==255] = 0
#     markers = cv2.watershed(img,markers)
#     img[markers == -1] = [255,0,0]

#     h, w = img_gray.shape
#     img1 = exposure.equalize_hist(img_gray)
#     # invert the image
#     img2 = np.max(img1) - img1
#     img2 = ndimage.median_filter(img2, 5)
#     inner = np.zeros((h, w), np.bool)
#     centroid = [round(a) for a in findCentroid(img2)]
#     inner[ int(centroid[0]), int(centroid[1])] = 1
#     min_size = round((h + w) / 10 )
#     kernel = morphology.disk(min_size)
#     inner = morphology.dilation(inner, kernel)

#     out = np.zeros((h,w), np.bool)
#     out[0, 0] = 1
#     out[h - 1, 0] = 1
#     out[0, w - 1] = 1
#     out[h - 1, w - 1] = 1
#     out = morphology.dilation(out, kernel)
#     out[0, :] = 1
#     out[h - 1, :] = 1
#     #out[:, w - 1] = 1
#     #out[:, 0] = 1

#     markers = np.zeros((h, w), np.int)
#     markers[inner] = 2
#     markers[out] = 1

#     labels = morphology.watershed(img2, markers)

#     return labels

def flood_Fitting_100_openCV(img):
    # Save img and read in to avoid OpenCV format problem
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    img = img.transpose((1, 2, 0))[:, :, 0]
    ax.imshow(img.astype(np.uint8), cmap='gray')
    fig.savefig('tmp.jpg')
    img = cv2.imread('tmp.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 15)
    ret, thresh = cv2.threshold(gray,80,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # noise removal
    kernel = np.ones((4,4),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    #plt.imshow(sure_bg)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,1,5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.6*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    #img[markers == -1] = [255,0,0]
    ### End of OpenCV
    # labels = watershed_image_100(img)
    results = measure.regionprops(markers)
    sorted(results, key=lambda k: k['area'],reverse=True)
    if len(results) >= 2:
        return results[1]
    else:
        return results[0]


=======
>>>>>>> 11523272183bfa4166cae2d6e8f88181519fa603
def watershed_image_100(img):
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
<<<<<<< HEAD
    img2 = median(img2)
    #img2 = ndimage.gaussian_filter(img2,0.2) #median_filter(img2, 50)
    inner = np.zeros((h, w), np.bool)
    centroid = [round(a) for a in findCentroid(img2)]
    inner[ int(centroid[0]), int(centroid[1])] = 1
    min_size = round((h + w) / 10 )
=======
    inner = np.zeros((h, w), np.bool)
    centroid = [round(a) for a in findCentroid(img2)]
    inner[centroid[0], centroid[1]] = 1
    min_size = round((h + w) / 6 )
>>>>>>> 11523272183bfa4166cae2d6e8f88181519fa603
    kernel = morphology.disk(min_size)
    inner = morphology.dilation(inner, kernel)

    out = np.zeros((h,w), np.bool)
    out[0, 0] = 1
    out[h - 1, 0] = 1
    out[0, w - 1] = 1
    out[h - 1, w - 1] = 1
    out = morphology.dilation(out, kernel)
<<<<<<< HEAD
    #out[0, :] = 1
    #out[h - 1, :] = 1
=======
    out[0, :] = 1
    out[h - 1, :] = 1
>>>>>>> 11523272183bfa4166cae2d6e8f88181519fa603
    #out[:, w - 1] = 1
    #out[:, 0] = 1

    markers = np.zeros((h, w), np.int)
    markers[inner] = 2
    markers[out] = 1

    labels = morphology.watershed(img2, markers)

    return labels

def flood_fitting_100(img):
    """
    Use watershed flooding algorithm and regional property analysis
    to output the fitted ellipse parameters
    :param img: (numpy.ndarray) image in CHW format
    :return: region property, where property can be accessed through attributes
            example:
            area, bbox, centroid, major_axis_length, minor_axis_length, orientation
    """
    labels = watershed_image_100(img)
    results = measure.regionprops(labels - 1)
    sorted(results, key=lambda k: k['area'],reverse=True)
    # return the one with largest area
<<<<<<< HEAD
    # if len(results) == 1:
    #     return results[0]
    # else:
    #     return results[1]
=======
>>>>>>> 11523272183bfa4166cae2d6e8f88181519fa603
    return results[0]


def watershed_image_blackdot(img):
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
<<<<<<< HEAD
    inner[ int(centroid[0]), int(centroid[1])] = 1
=======
    inner[centroid[0], centroid[1]] = 1
>>>>>>> 11523272183bfa4166cae2d6e8f88181519fa603
    min_size = round((h + w) / 4 )
    kernel = morphology.disk(min_size)
    inner = morphology.binary_dilation(inner, kernel)

    out = np.zeros((h,w), np.bool)
    out[0, 0] = 1
    out[h - 1, 0] = 1
    out[0, w - 1] = 1
    out[h - 1, w - 1] = 1
    out = morphology.dilation(out, kernel)
    #out[0, :] = 1
    #out[h - 1, :] = 1
    #out[:, w - 1] = 1
    #out[:, 0] = 1

    markers = np.zeros((h, w), np.int)
    markers[inner] = 2
    markers[out] = 1

    labels = morphology.watershed(img2, markers)

    return labels

def flood_fitting_blackdot(img):
    """
    Use watershed flooding algorithm and regional property analysis
    to output the fitted ellipse parameters
    :param img: (numpy.ndarray) image in CHW format
    :return: region property, where property can be accessed through attributes
            example:
            area, bbox, centroid, major_axis_length, minor_axis_length, orientation
    """
    labels = watershed_image_blackdot(img)
    results = measure.regionprops(labels - 1)
    sorted(results, key=lambda k: k['area'],reverse=True)
    # return the one with largest area
    return results[0]

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

def binary_threshold_fitting_100(image):
    """
    Use watershed flooding algorithm and regional property analysis
    to output the fitted ellipse parameters
    :param img: (numpy.ndarray) image in CHW format
    :return: region property, where property can be accessed through attributes
            example:
            area, bbox, centroid, major_axis_length, minor_axis_length, orientation
    """
    # minimum threshold
    img_gray = image[1, :, :]
    h, w = img_gray.shape
<<<<<<< HEAD
    img0 = ndimage.median_filter(img_gray, 5)
    #img0 = median(, disk(5))
    img1 = exposure.equalize_hist(img0)
    block_size = 11
    binary= threshold_adaptive(img1, block_size)

    label_img = label(binary,connectivity=5)
    results = measure.regionprops(label_img)
    sorted(results, key=lambda k: k['area'], reverse=True)
    # return the one with largest area
    print(results[0])
=======
    img1 = exposure.equalize_hist(img_gray)
    # invert the image
    # img2 = np.max(img1) - img1
    #inner = np.zeros((h, w), np.bool)
    #centroid = [round(a) for a in findCentroid(img2)]
    #inner[centroid[0], centroid[1]] = 1

    # h, w = img_gray.shape
    # img1 = exposure.equalize_hist(img_gray)
    # # invert the image
    # img2 = np.max(img1) - img1
    print(img1.shape)

    #thresh = threshold_yen( img1 )
    #thresh = threshold_otsu(img_gray)

    #binary = img_gray > thresh
    block_size = 11
    binary= threshold_adaptive(img1, block_size)

    label_img = label(binary,connectivity=2)
    results = measure.regionprops(label_img)
    #########################
    #
    #########################
    #min_size = round((h + w) / 6)
    #kernel = morphology.disk(min_size)
    #binary = morphology.dilation(binary0, kernel)

    #binary = morphology.convex_hull_image(binary0)
    #selem = disk(10)
    # binary_mean = rank.mean(binary,selem= np.ones((3,3)))
    # sigma = 0.05
    # sigma = 0.005 tilt
    # sigma = 0.0005
    # binary_mean = gaussian(binary, sigma=0.0005, preserve_range=True)
    # binary = morphology.remove_small_holes(binary0, 5000, connectivity=4, in_place=True)

    # # find contours
    # Xlist = list()
    # Ylist = list()
    # for contour in find_contours(binary, 0, fully_connected='high'):
    #     for item in contour:
    #         Xlist.append(item[0])
    #         Ylist.append(item[1])
    # Fit the Ellipse
    sorted(results, key=lambda k: k['area'], reverse=True)
    # return the one with largest area
>>>>>>> 11523272183bfa4166cae2d6e8f88181519fa603
    return results[0]
    #return FitEllipseAndParameters(Xlist,Ylist)

def binary_threshold_fitting_blackdots(image):
    """
    Use watershed flooding algorithm and regional property analysis
    to output the fitted ellipse parameters
    :param img: (numpy.ndarray) image in CHW format
    :return: region property, where property can be accessed through attributes
            example:
            area, bbox, centroid, major_axis_length, minor_axis_length, orientation
    """
    # minimum threshold
    img_gray = image[1, :, :]
    # h, w = img_gray.shape
    # img1 = exposure.equalize_hist(img_gray)
    # # invert the image
    # img2 = np.max(img1) - img1
    print(img_gray.shape)

    #thresh = threshold_otsu( img_gray )
    thresh = threshold_otsu(img_gray)

    binary = img_gray > thresh
    #selem = disk(10)
    # binary_mean = rank.mean(binary,selem= np.ones((3,3)))
    # sigma = 0.05
    # sigma = 0.005 tilt
    # sigma = 0.0005
    # binary_mean = gaussian(binary, sigma=0.0005, preserve_range=True)
    # binary = morphology.remove_small_holes(binary, 5000, connectivity=4, in_place=True)
    # find contours
    Xlist = list()
    Ylist = list()
    for contour in find_contours(closing(binary), 0,fully_connected='high'):
        for item in contour:
            Xlist.append(item[0])
            Ylist.append(item[1])
    # Fit the Ellipse
    return FitEllipseAndParameters(Xlist,Ylist)


# # from utils.postProcessing import img_ellipse_fitting_3kinds
# from utils.imageUtils import cropImage
# from skimage import exposure, morphology, measure, draw
# from utils.postProcessing import img_ellipse_fitting, flood_fitting, flood_fitting_blackdot, \
#     binary_threshold_fitting_100, flood_fitting_100, binary_threshold_fitting_blackdots


def img_ellipse_fitting_3kinds(img, bboxes, labels):
    imageSize = img[0, :, :]
    subimages, bboxes = cropImage(img, bboxes)
    y_points_0 = np.array([])
    x_points_0 = np.array([])
    y_points_1 = np.array([])
    x_points_1 = np.array([])
    y_points_2 = np.array([])
    x_points_2 = np.array([])
    for subim, bbox, label in zip(subimages, bboxes, labels):
        # 111 Loops
        if label == 0:
            region1 = flood_fitting(subim)
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_0 = np.concatenate((y_points_0, rr))
            x_points_0 = np.concatenate((x_points_0, cc))

        # Black Dots
        if label == 1:
            region1 = flood_fitting_blackdot(subim)
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_1 = np.concatenate((y_points_1, rr))
            x_points_1 = np.concatenate((x_points_1, cc))
            #
            # center,phi,axes = binary_threshold_fitting_blackdots(subim)
            # rr, cc = draw.ellipse_perimeter(int(center[0]+bbox[0]),
            #                 int(center[1]+bbox[1]),
            #                 int(axes[1] ) ,
            #                 int(axes[0] ) ,
            #                 phi,
            #                 imageSize.shape)
            # y_points_1 = np.concatenate((y_points_1,rr))
            # x_points_1 = np.concatenate((x_points_1,cc))

        # 100 Loops
        if label == 2:
            region1 = flood_fitting_100(subim)
            result = (int(region1['centroid'][0]+bbox[0]), int(region1['centroid'][1]+bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr,cc = draw.ellipse_perimeter(*result)
            y_points_2 = np.concatenate((y_points_2,rr))
            x_points_2 = np.concatenate((x_points_2,cc))
            #
            # center, phi, axes = binary_threshold_fitting_100(subim)
            # if math.isnan(center[0]) or math.isnan(center[1]):
            #     print("X")
            #     region1 = flood_fitting(subim)
            #     result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
            #               int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
            #               -region1['orientation'])
            # if math.isnan( phi ):
            #     print("Y")
            #     region1 = flood_fitting(subim)
            #     result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
            #               int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
            #               -region1['orientation'])
            # if math.isnan(axes[0]) or math.isnan(axes[1]):
            #     print("Z")
            #     region1 = flood_fitting(subim)
            #     result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
            #               int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
            #               -region1['orientation'])
            # #print(axes)
            # rr, cc = draw.ellipse_perimeter(int(center[0] + bbox[0]),
            #                                 int(center[1] + bbox[1]),
            #                                 int(axes[1]),
            #                                 int(axes[0]),
            #                                 phi,
            #                                 imageSize.shape)
            # y_points_2 = np.concatenate((y_points_2, rr))
            # x_points_2 = np.concatenate((x_points_2, cc))

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img[0, :, :], cmap='gray')
    plt.scatter(x_points_0, y_points_0, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='r')
    plt.scatter(x_points_1, y_points_1, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='b')
    plt.scatter(x_points_2, y_points_2, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='y')

def img_ellipse_fitting_3kinds_stat(img, bboxes, labels, Results):
    imageSize = img[0, :, :]
    subimages, bboxes = cropImage(img, bboxes)
    y_points_0 = np.array([])
    x_points_0 = np.array([])
    y_points_1 = np.array([])
    x_points_1 = np.array([])
    y_points_2 = np.array([])
    x_points_2 = np.array([])
    for subim, bbox, label in zip(subimages, bboxes, labels):
        # 111 Loops
        if label == 0:
            region1 = flood_fitting(subim)
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_0 = np.concatenate((y_points_0, rr))
            x_points_0 = np.concatenate((x_points_0, cc))
            Results[0].append( (int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2)) )

        # Black Dots
        if label == 1:
            region1 = flood_fitting_blackdot(subim)
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_1 = np.concatenate((y_points_1, rr))
            x_points_1 = np.concatenate((x_points_1, cc))
            Results[1].append((int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2)))
            #
            # center,phi,axes = binary_threshold_fitting_blackdots(subim)
            # rr, cc = draw.ellipse_perimeter(int(center[0]+bbox[0]),
            #                 int(center[1]+bbox[1]),
            #                 int(axes[1] ) ,
            #                 int(axes[0] ) ,
            #                 phi,
            #                 imageSize.shape)
            # y_points_1 = np.concatenate((y_points_1,rr))
            # x_points_1 = np.concatenate((x_points_1,cc))

        # 100 Loops
        if label == 2:
            region1 = binary_threshold_fitting_100(subim)
            result = (int(region1['centroid'][0]+ bbox[0] ), int(region1['centroid'][1]+ bbox[1] ),
                      int(0.95*region1['minor_axis_length'] / 2), int(0.95*region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr,cc = draw.ellipse_perimeter(*result)
            #print(rr.shape,cc.shape)
            try :
                y_points_2 = np.concatenate((y_points_2,rr))
                x_points_2 = np.concatenate((x_points_2,cc))
                Results[2].append((int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2)))
            except ValueError:
                region1 = flood_fitting_100(subim)
                result = (int(region1['centroid'][0] +  bbox[0] ), int(region1['centroid'][1] +  bbox[1]),
                          int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                          -region1['orientation'])
                rr, cc = draw.ellipse_perimeter(*result)
                y_points_2 = np.concatenate((y_points_2, rr))
                x_points_2 = np.concatenate((x_points_2, cc))
                start = (int(bbox[0]),int(bbox[1]))
                extent = (int(bbox[2]-bbox[0]),int(bbox[3]-bbox[1]))
                print("Extent",extent)
                rr, cc = draw.rectangle(start, extent=extent,shape=imageSize.size)
                print(rr.shape,cc.shape)
                try:
                    y_points_2 = np.concatenate((y_points_2, rr))
                    x_points_2 = np.concatenate((x_points_2, cc))
                except:
                    print("!!!Fitting Error")

            # center, phi, axes = binary_threshold_fitting_100(subim)
            # if math.isnan(center[0]) or math.isnan(center[1]):
            #     print("X")
            #     continue
            # if math.isnan( phi ):
            #     print("Y")
            #     continue
            # if math.isnan(axes[0]) or math.isnan(axes[1]):
            #     print("Z")
            #     continue
            # #print(axes)
            # # check if all the fitted ellipse is inside the bbox
            # # check longer axis

            # rr, cc = draw.ellipse_perimeter(int(center[0] + bbox[0]),
            #                                 int(center[1] + bbox[1]),
            #                                 int(axes[1]), # Minor
            #                                 int(axes[0]), # Major
            #                                 phi,
            #                                 imageSize.shape)
            # y_points_2 = np.concatenate((y_points_2, rr))
            # x_points_2 = np.concatenate((x_points_2, cc))
            # Results[2].append(( int(axes[1]) ,int(axes[0]) ))

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img[0, :, :], cmap='gray')
    plt.scatter(x_points_0, y_points_0, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='r')
    plt.scatter(x_points_1, y_points_1, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='b')
    plt.scatter(x_points_2, y_points_2, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='y')

<<<<<<< HEAD
def img_ellipse_fitting_3kinds_stat_convert(img, bboxes, labels, convFactor,index, Results):
    imageSize = img[0, :, :]
    subimages, bboxes = cropImage(img, bboxes)
    y_points_0 = np.array([])
    x_points_0 = np.array([])
    y_points_1 = np.array([])
    x_points_1 = np.array([])
    y_points_2 = np.array([])
    x_points_2 = np.array([])
    rectList = list()
    for subim, bbox, label in zip(subimages, bboxes, labels):
        # 111 Loops
        if label == 0:
            region1 = flood_fitting(subim)
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_0 = np.concatenate((y_points_0, rr))
            x_points_0 = np.concatenate((x_points_0, cc))
            Results[0].append( (region1['minor_axis_length'] * convFactor, region1['major_axis_length'] * convFactor) )

        # Black Dots
        if label == 1:
            region1 = flood_fitting_blackdot(subim)
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_1 = np.concatenate((y_points_1, rr))
            x_points_1 = np.concatenate((x_points_1, cc))
            Results[1].append( (region1['minor_axis_length'] * convFactor, region1['major_axis_length'] * convFactor) )
            #
            # center,phi,axes = binary_threshold_fitting_blackdots(subim)
            # rr, cc = draw.ellipse_perimeter(int(center[0]+bbox[0]),
            #                 int(center[1]+bbox[1]),
            #                 int(axes[1] ) ,
            #                 int(axes[0] ) ,
            #                 phi,
            #                 imageSize.shape)
            # y_points_1 = np.concatenate((y_points_1,rr))
            # x_points_1 = np.concatenate((x_points_1,cc))

        # 100 Loops
        if label == 2:
            curr_bbox = ( bbox[2] - bbox[0], bbox[3] - bbox[1] )
            MinMaxRatio = 1.0 * np.max(curr_bbox) / np.min(curr_bbox)
            if MinMaxRatio < 1.4:
                region1 = flood_fitting_100(subim)
                result = (int(region1['centroid'][0] + bbox[0]), 
                          int( region1['centroid'][1] + bbox[1]),
                          int( 0.95 * region1['minor_axis_length'] / 2), 
                          int( 0.95 * region1['major_axis_length'] / 2),
                          -region1['orientation'])
                rr, cc = draw.ellipse_perimeter(*result)
                try :
                    y_points_2 = np.concatenate((y_points_2, rr))
                    x_points_2 = np.concatenate((x_points_2, cc))
                    Results[2].append( (region1['minor_axis_length'] * convFactor, region1['major_axis_length'] * convFactor) )
                except:
                    print("Using Rectangle to Fix 100 in %d"%index)
                    # record down information of bounding boxes
                    rectList.append(
                        patches.Rectangle(
                            (bbox[1], bbox[2]),
                            np.abs(bbox[3] - bbox[1]),
                            np.abs(bbox[2] - bbox[0]),
                            fill=False, 
                            linewidth = 4,
                            edgecolor = "olive"))
                    Results[2].append( ( np.abs(bbox[3] - bbox[1]) * convFactor, np.abs(bbox[2] - bbox[0]) * convFactor) )
            else:
                region1 = binary_threshold_fitting_100(subim)
                result = (int(region1['centroid'][0]+ bbox[0] ), int(region1['centroid'][1]+ bbox[1] ),
                        int( region1['minor_axis_length'] / 2), int( region1['major_axis_length'] / 2),
                        -region1['orientation'])
                rr,cc = draw.ellipse_perimeter(*result)
                #print(rr.shape,cc.shape)
                try :
                    y_points_2 = np.concatenate((y_points_2,rr))
                    x_points_2 = np.concatenate((x_points_2,cc))
                    Results[2].append( (region1['minor_axis_length'] * convFactor, region1['major_axis_length'] * convFactor) )
                except :
                    print("Using Rectangle to Fix 100 in %d"%index)
                    # record down information of bounding boxes
                    rectList.append(
                        patches.Rectangle(
                            (bbox[1], bbox[2]),
                            np.abs(bbox[3] - bbox[1]),
                            np.abs(bbox[2] - bbox[0]),
                            fill=False, 
                            linewidth = 4,
                            edgecolor = "green"))
                    Results[2].append( ( np.abs(bbox[3] - bbox[1]) * convFactor, np.abs(bbox[2] - bbox[0]) * convFactor) )
            # center, phi, axes = binary_threshold_fitting_100(subim)
            # if math.isnan(center[0]) or math.isnan(center[1]):
            #     print("X")
            #     continue
            # if math.isnan( phi ):
            #     print("Y")
            #     continue
            # if math.isnan(axes[0]) or math.isnan(axes[1]):
            #     print("Z")
            #     continue
            # #print(axes)
            # # check if all the fitted ellipse is inside the bbox
            # # check longer axis

            # rr, cc = draw.ellipse_perimeter(int(center[0] + bbox[0]),
            #                                 int(center[1] + bbox[1]),
            #                                 int(axes[1]), # Minor
            #                                 int(axes[0]), # Major
            #                                 phi,
            #                                 imageSize.shape)
            # y_points_2 = np.concatenate((y_points_2, rr))
            # x_points_2 = np.concatenate((x_points_2, cc))
            # Results[2].append(( int(axes[1]) ,int(axes[0]) ))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    plt.imshow(img[0, :, :], cmap='gray')
    plt.scatter(x_points_0, y_points_0, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='r')
    plt.scatter(x_points_1, y_points_1, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='b')
    plt.scatter(x_points_2, y_points_2, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='y')
    # Plot missing green rectangle
    for p in rectList:
        ax.add_patch(p)
    plt.savefig(str(index)+".jpg",dpi=150,bbox_inches='tight')
    plt.clf()
=======
>>>>>>> 11523272183bfa4166cae2d6e8f88181519fa603

def img_ellipse_fitting_3kinds_Fig3(img, bboxes, labels, ax = None):
    imageSize = img[0, :, :]
    subimages, bboxes = cropImage(img, bboxes)
    y_points_0 = np.array([])
    x_points_0 = np.array([])
    y_points_1 = np.array([])
    x_points_1 = np.array([])
    y_points_2 = np.array([])
    x_points_2 = np.array([])
    from matplotlib import pyplot as plot
    plot.axis('off')
    for subim, bbox, label in zip(subimages, bboxes, labels):
        # 111 Loops
        if label == 0:
            region1 = flood_fitting(subim)
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_0 = np.concatenate((y_points_0, rr))
            x_points_0 = np.concatenate((x_points_0, cc))

        # Black Dots
        if label == 1:
            region1 = flood_fitting_blackdot(subim)
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_1 = np.concatenate((y_points_1, rr))
            x_points_1 = np.concatenate((x_points_1, cc))
            # Results[1].append((int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2)))
            #
            # center,phi,axes = binary_threshold_fitting_blackdots(subim)
            # rr, cc = draw.ellipse_perimeter(int(center[0]+bbox[0]),
            #                 int(center[1]+bbox[1]),
            #                 int(axes[1] ) ,
            #                 int(axes[0] ) ,
            #                 phi,
            #                 imageSize.shape)
            # y_points_1 = np.concatenate((y_points_1,rr))
            # x_points_1 = np.concatenate((x_points_1,cc))

        # 100 Loops
        if label == 2:
            # region1 = flood_fitting_100(subim)
            # result = (int(region1['centroid'][0]+bbox[0]), int(region1['centroid'][1]+bbox[1]),
            #           int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
            #           -region1['orientation'])
            # rr,cc = draw.ellipse_perimeter(*result)
            # y_points_2 = np.concatenate((y_points_2,rr))
            # x_points_2 = np.concatenate((x_points_2,cc))

            center, phi, axes = binary_threshold_fitting_100(subim)

            if math.isnan(center[0]) or math.isnan(center[1]) or math.isnan( phi ) or math.isnan(axes[0]) or math.isnan(axes[1]):
                region1 = flood_fitting(subim)
                center[0] = region1['centroid'][0]
                center[1] = region1['centroid'][1]
                axes[1] = int(region1['minor_axis_length'] / 2)
                axes[0] = int(region1['major_axis_length'] / 2)
                phi = -region1['orientation']
            #print(axes)
            rr, cc = draw.ellipse_perimeter(int(center[0] + bbox[0]),
                                            int(center[1] + bbox[1]),
                                            int(axes[1]),
                                            int(axes[0]),
                                            phi,
                                            imageSize.shape)
            y_points_2 = np.concatenate((y_points_2, rr))
            x_points_2 = np.concatenate((x_points_2, cc))
            # Results[2].append(( int(axes[1]) ,int(axes[0]) ))

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)
    ax.scatter(x_points_0, y_points_0, s=1, alpha=0.5, c='r')
    ax.scatter(x_points_1, y_points_1, s=1, alpha=0.5, c='b')
    ax.scatter(x_points_2, y_points_2, s=1, alpha=0.5, c='y')
<<<<<<< HEAD
    return ax

def img_ellipse_fitting_3kinds_stat_convert_debug_OutPut(img, bboxes, labels, convFactor,index, Results):
    # Creating the storing folder
    try:
        os.mkdir("Results_"+str(index))
        print(index)
    except OSError:
        print("Creation of the directory is failed")
    #imageSize = img[0, :, :]
    subimages, bboxes = cropImage(img, bboxes)
    y_points_0 = np.array([])
    x_points_0 = np.array([])
    y_points_1 = np.array([])
    x_points_1 = np.array([])
    y_points_2 = np.array([])
    x_points_2 = np.array([])
    rectList = list()
    subImgID = 0
    for subim, bbox, label in zip(subimages, bboxes, labels):
        plt.figure()
        ax = plt.gca()
        # Saving subImg Original for Proprocessing
        plt.imshow(subim.transpose((1, 2, 0))[:, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig("Results_"+str(index)+"/"+str(subImgID)+".jpg",dpi=300,bbox_inches='tight')
        plt.clf()       
        # 111 Loops
        if label == 0:
            region1 = flood_fitting(subim)
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            # Saving Fitting
            ellipse = Ellipse(xy=(region1['centroid'][0], region1['centroid'][1]), width=region1['minor_axis_length'] / 2.0, height=region1['major_axis_length'] / 2.0, 
                        angle = -180.0*region1['orientation'], edgecolor='r', fc='None', lw=2)
            plt.figure()
            ax = plt.gca()
            # Saving subImg Original for Proprocessing
            plt.imshow(subim.transpose((1, 2, 0))[:, :, 0], cmap='gray')
            plt.axis('off')
            #plt.savefig("Results_"+str(index)+"/"+str(subImgID)+".jpg",dpi=300,bbox_inches='tight')   
            ax.add_patch(ellipse)
            plt.savefig("Results_"+str(index)+"/"+str(subImgID)+"_Fitted_111"+".jpg",dpi=300,bbox_inches='tight')
            plt.clf()
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_0 = np.concatenate((y_points_0, rr))
            x_points_0 = np.concatenate((x_points_0, cc))
            Results[0].append( (region1['minor_axis_length'] * convFactor, region1['major_axis_length'] * convFactor) )
        # Black Dots
        if label == 1:
            region1 = flood_fitting_blackdot(subim)
            # Saving Fitting
            ellipse = Ellipse(xy=(region1['centroid'][0], region1['centroid'][1]), width=region1['minor_axis_length'] / 2.0, height=region1['major_axis_length'] / 2.0, 
                        angle = -180.0*region1['orientation'], edgecolor='r', fc='None', lw=2)
            plt.figure()
            ax = plt.gca()
            plt.imshow(subim.transpose((1, 2, 0))[:, :, 0], cmap='gray')
            plt.axis('off')
            #plt.savefig("Results_"+str(index)+"/"+str(subImgID)+".jpg",dpi=300,bbox_inches='tight')   
            ax.add_patch(ellipse)
            plt.savefig("Results_"+str(index)+"/"+str(subImgID)+"_Fitted_BD"+".jpg",dpi=300,bbox_inches='tight')
            plt.clf()
            # Ending Saving Fitting
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_1 = np.concatenate((y_points_1, rr))
            x_points_1 = np.concatenate((x_points_1, cc))
            Results[1].append( (region1['minor_axis_length'] * convFactor, region1['major_axis_length'] * convFactor) )

        # 100 Loops
        if label == 2:
            curr_bbox = ( bbox[2] - bbox[0], bbox[3] - bbox[1] )
            MinMaxRatio = 1.0 * np.max(curr_bbox) / np.min(curr_bbox)
            if MinMaxRatio < 1.4:
                region1 = flood_fitting_100(subim)
                # Saving Fitting
                ellipse = Ellipse(xy=(region1['centroid'][0], region1['centroid'][1]), width=region1['minor_axis_length'] / 2.0, height=region1['major_axis_length'] / 2.0, 
                            angle = -180.0*region1['orientation'], edgecolor='r', fc='None', lw=2)
                plt.figure()
                ax = plt.gca()
                plt.imshow(subim.transpose((1, 2, 0))[:, :, 0], cmap='gray')
                plt.axis('off')
                #plt.savefig("Results_"+str(index)+"/"+str(subImgID)+".jpg",dpi=300,bbox_inches='tight')   
                ax.add_patch(ellipse)
                plt.savefig("Results_"+str(index)+"/"+str(subImgID)+"_Fitted_100"+".jpg",dpi=300,bbox_inches='tight')
                plt.clf()
                # Ending Saving Fitting
                result = (int(region1['centroid'][0] + bbox[0]), 
                          int( region1['centroid'][1] + bbox[1]),
                          int( 0.95 * region1['minor_axis_length'] / 2), 
                          int( 0.95 * region1['major_axis_length'] / 2),
                          -region1['orientation'])
                rr, cc = draw.ellipse_perimeter(*result)
                try :
                    y_points_2 = np.concatenate((y_points_2, rr))
                    x_points_2 = np.concatenate((x_points_2, cc))
                    Results[2].append( (region1['minor_axis_length'] * convFactor, region1['major_axis_length'] * convFactor) )
                except:
                    print("Using Rectangle to Fix 100 in %d"%index)
                    # record down information of bounding boxes
                    rectList.append(
                        patches.Rectangle(
                            (bbox[1], bbox[2]),
                            np.abs(bbox[3] - bbox[1]),
                            np.abs(bbox[2] - bbox[0]),
                            fill=False, 
                            linewidth = 4,
                            edgecolor = "olive"))
                    Results[2].append( ( np.abs(bbox[3] - bbox[1]) * convFactor, np.abs(bbox[2] - bbox[0]) * convFactor) )
            else:
                region1 = binary_threshold_fitting_100(subim)
                result = (int(region1['centroid'][0]+ bbox[0] ), int(region1['centroid'][1]+ bbox[1] ),
                        int( region1['minor_axis_length'] / 2), int( region1['major_axis_length'] / 2),
                        -region1['orientation'])
                rr,cc = draw.ellipse_perimeter(*result)
                # Saving Fitting
                ellipse = Ellipse(xy=(region1['centroid'][0], region1['centroid'][1]), width=region1['minor_axis_length'] / 2.0, height=region1['major_axis_length'] / 2.0, 
                            angle = -180.0*region1['orientation'], edgecolor='r', fc='None', lw=2)
                plt.figure()
                ax = plt.gca()
                plt.imshow(subim.transpose((1, 2, 0))[:, :, 0], cmap='gray')
                plt.axis('off')
                #plt.savefig("Results_"+str(index)+"/"+str(subImgID)+".jpg",dpi=300,bbox_inches='tight')   
                ax.add_patch(ellipse)
                plt.savefig("Results_"+str(index)+"/"+str(subImgID)+"_Fitted_100"+".jpg",dpi=300,bbox_inches='tight')
                plt.clf()
                # Ending Saving Fitting
                #print(rr.shape,cc.shape)
                try :
                    y_points_2 = np.concatenate((y_points_2,rr))
                    x_points_2 = np.concatenate((x_points_2,cc))
                    Results[2].append( (region1['minor_axis_length'] * convFactor, region1['major_axis_length'] * convFactor) )
                except :
                    print("Using Rectangle to Fix 100 in %d"%index)
                    # record down information of bounding boxes
                    rectList.append(
                        patches.Rectangle(
                            (bbox[1], bbox[2]),
                            np.abs(bbox[3] - bbox[1]),
                            np.abs(bbox[2] - bbox[0]),
                            fill=False, 
                            linewidth = 4,
                            edgecolor = "green"))
                    Results[2].append( ( np.abs(bbox[3] - bbox[1]) * convFactor, np.abs(bbox[2] - bbox[0]) * convFactor) )
        subImgID += 1
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    plt.imshow(img[0, :, :], cmap='gray')
    plt.scatter(x_points_0, y_points_0, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='r')
    plt.scatter(x_points_1, y_points_1, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='b')
    plt.scatter(x_points_2, y_points_2, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='y')
    # Plot missing green rectangle
    for p in rectList:
        ax.add_patch(p)
    plt.savefig(str(index)+".jpg",dpi=150,bbox_inches='tight')
    plt.clf()

def img_ellipse_fitting_3kinds_stat_convert_debug_OutPut_OpenCV(img, bboxes, labels, convFactor,index, Results):
    # Creating the storing folder
    try:
        os.mkdir("Results_"+str(index))
        print(index)
    except OSError:
        print("Creation of the directory is failed")
    #imageSize = img[0, :, :]
    subimages, bboxes = cropImage(img, bboxes)
    y_points_0 = np.array([])
    x_points_0 = np.array([])
    y_points_1 = np.array([])
    x_points_1 = np.array([])
    y_points_2 = np.array([])
    x_points_2 = np.array([])
    rectList = list()
    subImgID = 0
    for subim, bbox, label in zip(subimages, bboxes, labels):
        plt.figure()
        ax = plt.gca()
        # Saving subImg Original for Proprocessing
        plt.imshow(subim.transpose((1, 2, 0))[:, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig("Results_"+str(index)+"/"+str(subImgID)+".jpg",dpi=300,bbox_inches='tight')
        plt.clf()       
        # 111 Loops
        if label == 0:
            region1 = flood_fitting(subim)
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            # Saving Fitting
            ellipse = Ellipse(xy=(region1['centroid'][0], region1['centroid'][1]), width=region1['minor_axis_length'] / 2.0, height=region1['major_axis_length'] / 2.0, 
                        angle = 180.0*region1['orientation'], edgecolor='r', fc='None', lw=2)
            plt.figure()
            ax = plt.gca()
            # Saving subImg Original for Proprocessing
            plt.imshow(subim.transpose((1, 2, 0))[:, :, 0], cmap='gray')
            plt.axis('off')
            #plt.savefig("Results_"+str(index)+"/"+str(subImgID)+".jpg",dpi=300,bbox_inches='tight')   
            ax.add_patch(ellipse)
            plt.savefig("Results_"+str(index)+"/"+str(subImgID)+"_Fitted_111"+".jpg",dpi=300,bbox_inches='tight')
            plt.clf()
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_0 = np.concatenate((y_points_0, rr))
            x_points_0 = np.concatenate((x_points_0, cc))
            Results[0].append( (region1['minor_axis_length'] * convFactor, region1['major_axis_length'] * convFactor) )
        # Black Dots
        if label == 1:
            region1 = flood_fitting_blackdot(subim)
            # Saving Fitting
            ellipse = Ellipse(xy=(region1['centroid'][0], region1['centroid'][1]), width=region1['minor_axis_length'] / 2.0, height=region1['major_axis_length'] / 2.0, 
                        angle = 180.0*region1['orientation'], edgecolor='r', fc='None', lw=2)
            plt.figure()
            ax = plt.gca()
            plt.imshow(subim.transpose((1, 2, 0))[:, :, 0], cmap='gray')
            plt.axis('off')
            #plt.savefig("Results_"+str(index)+"/"+str(subImgID)+".jpg",dpi=300,bbox_inches='tight')   
            ax.add_patch(ellipse)
            plt.savefig("Results_"+str(index)+"/"+str(subImgID)+"_Fitted_BD"+".jpg",dpi=300,bbox_inches='tight')
            plt.clf()
            # Ending Saving Fitting
            result = (int(region1['centroid'][0] + bbox[0]), int(region1['centroid'][1] + bbox[1]),
                      int(region1['minor_axis_length'] / 2), int(region1['major_axis_length'] / 2),
                      -region1['orientation'])
            rr, cc = draw.ellipse_perimeter(*result)
            y_points_1 = np.concatenate((y_points_1, rr))
            x_points_1 = np.concatenate((x_points_1, cc))
            Results[1].append( (region1['minor_axis_length'] * convFactor, region1['major_axis_length'] * convFactor) )

        # 100 Loops
        if label == 2:
            region1 = binary_threshold_fitting_100(subim)#flood_fitting_100(subim)#flood_Fitting_100_openCV(subim)
            # Saving Fitting
            ellipse = Ellipse(xy=(region1['centroid'][0], region1['centroid'][1]), width=region1['minor_axis_length'] / 2.0, height=region1['major_axis_length'] / 2.0, 
                        angle = 180.0*region1['orientation'], edgecolor='r', fc='None', lw=2)
            plt.figure()
            ax = plt.gca()
            plt.imshow(subim.transpose((1, 2, 0))[:, :, 0], cmap='gray')
            plt.axis('off')
            #plt.savefig("Results_"+str(index)+"/"+str(subImgID)+".jpg",dpi=300,bbox_inches='tight')   
            ax.add_patch(ellipse)
            plt.savefig("Results_"+str(index)+"/"+str(subImgID)+"_Fitted_100"+".jpg",dpi=300,bbox_inches='tight')
            plt.clf()
            # Ending Saving Fitting
            result = (int(region1['centroid'][0] + bbox[0]), 
                        int( region1['centroid'][1] + bbox[1]),
                        int( region1['minor_axis_length'] / 2), 
                        int( region1['major_axis_length'] / 2),
                        region1['orientation'])
            rr, cc = draw.ellipse_perimeter(*result)
            print(len(y_points_2))
            print(len(rr))
            print(len(cc))
            y_points_2 = np.concatenate((y_points_2,rr))
            x_points_2 = np.concatenate((x_points_2,cc))
            Results[2].append( (region1['minor_axis_length'] * convFactor, region1['major_axis_length'] * convFactor) )
        subImgID += 1
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    plt.imshow(img[0, :, :], cmap='gray')
    plt.scatter(x_points_0, y_points_0, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='r')
    plt.scatter(x_points_1, y_points_1, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='b')
    plt.scatter(x_points_2, y_points_2, s=(1 * 72. / fig.dpi) ** 2, alpha=0.5, c='y')
    # Plot missing green rectangle
    for p in rectList:
        ax.add_patch(p)
    plt.savefig(str(index)+".jpg",dpi=150,bbox_inches='tight')
    plt.clf()
=======
    return ax
>>>>>>> 11523272183bfa4166cae2d6e8f88181519fa603
