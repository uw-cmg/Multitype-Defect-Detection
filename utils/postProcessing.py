from .imageUtils import cropImage
from .FitEllipse import FitEllipseAndParameters
from skimage import exposure, morphology, measure, draw
from skimage.filters import threshold_yen, threshold_minimum, threshold_otsu, gaussian, threshold_adaptive
from skimage.measure import label,find_contours
from skimage.filters import rank
from skimage.morphology import disk
from .visualization import vis_image
from utils.imageUtils import cropImage
from skimage import exposure, morphology, measure, draw
from skimage.morphology import remove_small_objects,closing
# from utils.postProcessing import img_ellipse_fitting, flood_fitting, binary_threshold_fitting
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
    inner = np.zeros((h, w), np.bool)
    centroid = [round(a) for a in findCentroid(img2)]
    inner[centroid[0], centroid[1]] = 1
    min_size = round((h + w) / 6 )
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
    inner[centroid[0], centroid[1]] = 1
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
    return ax