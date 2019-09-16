import numpy as np
import os
from skimage import exposure, filters
import chainer
from chainercv import utils
from chainercv import transforms
import warnings

#root = '/home/wei/Data/Loop_detection/'
root = '/srv/home/dwu93/data/loop111'
root2 = './data/Data3TypesYminXminYmaxXmax2'

class DefectDetectionDataset(chainer.dataset.DatasetMixin):
    """Base class for defect defection dataset
    """

    def __init__(self, data_dir='auto', split='', img_size=1024, resize=False):
        if data_dir == 'auto':
            data_dir = root
        self.data_dir = data_dir
        self.img_size = img_size
        self.resize = resize
        images_file = os.path.join(self.data_dir, '{}images.txt'.format(split))
        self.images = [
            line.strip() for line in open(images_file)]

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and its label.
            The image is in CHW format and its color channel is ordered in
            RGB.
            a bounding box is appended to the returned value.
        """
        img = utils.read_image(
            os.path.join(self.data_dir, 'images', self.images[i]),
            color=True)
        # Add processing to the other two channels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img[1, :, :] = exposure.rescale_intensity(exposure.equalize_adapthist(
                exposure.rescale_intensity(img[1, :, :])), out_range=(0, 255))
            img[2, :, :] = exposure.rescale_intensity(filters.gaussian(
                exposure.rescale_intensity(img[2, :, :])), out_range=(0, 255))

        # bbs should be a matrix (m by 4). m is the number of bounding
        # boxes in the image
        # labels should be an integer array (m by 1). m is the same as the bbs

        bbs_file = os.path.join(self.data_dir, 'bounding_boxes', self.images[i][0:-4]+'.txt')
        
        bbs = np.stack([line.strip().split() for line in open(bbs_file)]).astype(np.float32)
        label = np.stack([0]*bbs.shape[0]).astype(np.int32)

        _, H, W = img.shape
        if self.resize and (H != self.img_size or W != self.img_size):
            img = transforms.resize(img, (self.img_size, self.img_size))
            bbs = transforms.resize_bbox(bbs, (H, W), (self.img_size, self.img_size))

        return img, bbs, label

class MultiDefectDetectionDataset(chainer.dataset.DatasetMixin):
    """Base class for multi defect defection dataset
    """

    def __init__(self, data_dir='auto', split='', img_size=1024, resize=False):
        if data_dir == 'auto':
            data_dir = root2
        self.data_dir = data_dir
        self.img_size = img_size
        self.resize = resize
        images_file = os.path.join(self.data_dir, '{}images.txt'.format(split))
        self.images = [
            line.strip() for line in open(images_file)]

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and its label.
            The image is in CHW format and its color channel is ordered in
            RGB.
            a bounding box is appended to the returned value.
        """
        #print("The image file name is %s"%self.images[i][0:-4])
        img = utils.read_image(
            os.path.join(self.data_dir, 'images', self.images[i]),
            color=True)
        # Add processing to the other two channels
        with warnings.catch_warnings():
            # print("read in by expanding")
            warnings.simplefilter("ignore")
            img[1, :, :] = exposure.rescale_intensity(exposure.equalize_adapthist(
                exposure.rescale_intensity(img[1, :, :])), out_range=(0, 255))
            img[2, :, :] = exposure.rescale_intensity(filters.gaussian(
                exposure.rescale_intensity(img[2, :, :])), out_range=(0, 255))

        # bbs should be a matrix (m by 4). m is the number of bounding
        # boxes in the image
        # labels should be an integer array (m by 1). m is the same as the bbs

        bbs_file = os.path.join(self.data_dir, 'bounding_boxes', self.images[i][0:-4]+'.txt')
        
        label_bbs = np.loadtxt(bbs_file, dtype=np.float32)
        label = label_bbs[:,0].astype(np.int32)
        bbs = label_bbs[:,1:5]

        _, H, W = img.shape
        if self.resize and (H != self.img_size or W != self.img_size):
            img = transforms.resize(img, (self.img_size, self.img_size))
            bbs = transforms.resize_bbox(bbs, (H, W), (self.img_size, self.img_size))

        return img, bbs, label

    def get_example_by_name(self, strImageName):
        """Returns the image of a certain name.

        Args:
            strImageName (string): The string that represents the image name.

        Returns:
            tuple of an image and its label.
            The image is in CHW format and its color channel is ordered in
            RGB.
            a bounding box is appended to the returned value.
        """
        fullstrImageName = strImageName + ".jpg"
        img = utils.read_image(os.path.join(self.data_dir, 'images',fullstrImageName ),color=True)
        # Add processing to the other two channels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img[1, :, :] = exposure.rescale_intensity(exposure.equalize_adapthist(
                exposure.rescale_intensity(img[1, :, :])), out_range=(0, 255))
            img[2, :, :] = exposure.rescale_intensity(filters.gaussian(
                exposure.rescale_intensity(img[2, :, :])), out_range=(0, 255))

        # bbs should be a matrix (m by 4). m is the number of bounding
        # boxes in the image
        # labels should be an integer array (m by 1). m is the same as the bbs

        bbs_file = os.path.join(self.data_dir, 'bounding_boxes', strImageName + '.txt')

        label_bbs = np.loadtxt(bbs_file, dtype=np.float32)
        label = label_bbs[:, 0].astype(np.int32)
        bbs = label_bbs[:, 1:5]

        _, H, W = img.shape
        if self.resize and (H != self.img_size or W != self.img_size):
            img = transforms.resize(img, (self.img_size, self.img_size))
            bbs = transforms.resize_bbox(bbs, (H, W), (self.img_size, self.img_size))

        return img, bbs, label

    def copy_example_image(self, i):
        """Copy the i-th example JPG image.

        Args:
            i (int): The index of the example.

        Returns:
            None
        """
        #print("The image file name is %s"%self.images[i][0:-4])
        from shutil import copy
        copy(os.path.join(self.data_dir, 'images', self.images[i]),'.')

    def get_image_name(self, i):
        """Return the i-th example JPG image name.

        Args:
            i (int): The index of the example.

        Returns:
            None
        """
        #print("The image file name is %s"%self.images[i][0:-4])
        return self.images[i][0:-4]