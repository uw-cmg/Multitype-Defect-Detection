from chainercv import transforms
import random
import numpy as np
from chainercv import utils
import six


def rotate_bbox(bbox, size, k):
    """Rotate bounding boxes accordingly

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    :param bbox:
    :param size:
    :param k:
    :return:
    """
    H, W = size
    origin = (W/2, H/2)
    p1 = (bbox[:, 1], bbox[:, 0])
    p2 = (bbox[:, 3], bbox[:, 2])
    k = k % 4
    if k != 0 and len(p1) > 0:
        new_p1 = rotate_point(p1, origin, k)
        new_p2 = rotate_point(p2, origin, k)
        bbox[:, 0] = np.min([new_p1[1], new_p2[1]],axis=0)
        bbox[:, 2] = np.max([new_p1[1], new_p2[1]],axis=0)
        bbox[:, 1] = np.min([new_p1[0], new_p2[0]],axis=0)
        bbox[:, 3] = np.max([new_p1[0], new_p2[0]],axis=0)
    return bbox


def random_resize(img):
    rv = random.random()
    if rv < 0.5:
        ratio = round(rv*2, 1)
        _, H, W = img.shape
        img = transforms.resize(img, (int(ratio*H), int(ratio*W)))
    return img


def rotate_point(point, origin, k):
    x, y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = [1, 0, -1, 0][k]
    sin_rad = [0, 1, 0, -1][k]
    if k%2 == 1:
        offset_x, offset_y = offset_y, offset_x
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


def random_distort(
        img,
        brightness_delta=32,
        contrast_low=0.5, contrast_high=1.5):
    """An adjusted color related data augmentation.

    This function is a combination of four augmentation methods:
    brightness, contrast, saturation and hue.

    * brightness: Adding a random offset to the intensity of the image.
    * contrast: Multiplying the intensity of the image by a random scale.



    Args:
        img (~numpy.ndarray): An image array to be augmented. This is in
            CHW and RGB format.
        brightness_delta (float): The offset for saturation will be
            drawn from :math:`[-brightness\_delta, brightness\_delta]`.
            The default value is :obj:`32`.
        contrast_low (float): The scale for contrast will be
            drawn from :math:`[contrast\_low, contrast\_high]`.
            The default value is :obj:`0.5`.
        contrast_high (float): See :obj:`contrast_low`.
            The default value is :obj:`1.5`.

    Returns:
        An image in CHW and RGB format.

    """

    cv_img = img[::-1].transpose((1, 2, 0)).astype(np.uint8)

    def convert(img, alpha=1, beta=0):
        img = img.astype(float) * alpha + beta
        img[img < 0] = 0
        img[img > 255] = 255
        return img.astype(np.uint8)

    def brightness(cv_img, delta):
        if random.randrange(2):
            return convert(
                cv_img,
                beta=random.uniform(-delta, delta))
        else:
            return cv_img

    def contrast(cv_img, low, high):
        if random.randrange(2):
            return convert(
                cv_img,
                alpha=random.uniform(low, high))
        else:
            return cv_img

    if random.randrange(2):
        cv_img = brightness(cv_img, brightness_delta)
        cv_img = contrast(cv_img, contrast_low, contrast_high)
    else:
        cv_img = contrast(cv_img, contrast_low, contrast_high)
        cv_img = brightness(cv_img, brightness_delta)

    return cv_img.astype(np.float32).transpose((2, 0, 1))[::-1]


def random_crop_with_bbox_constraints(
        img, bbox, min_scale=0.3, max_scale=1,
        max_aspect_ratio=2, constraints=None,
        max_trial=50, return_param=False):
    """Crop an image randomly with bounding box constraints.

    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_. More details can be found in
    data augmentation section of the original paper.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        img (~numpy.ndarray): An image array to be cropped. This is in
            CHW format.
        bbox (~numpy.ndarray): Bounding boxes used for constraints.
            The shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        min_scale (float): The minimum ratio between a cropped
            region and the original image. The default value is :obj:`0.3`.
        max_scale (float): The maximum ratio between a cropped
            region and the original image. The default value is :obj:`1`.
        max_aspect_ratio (float): The maximum aspect ratio of cropped region.
            The default value is :obj:`2`.
        constaraints (iterable of tuples): An iterable of constraints.
            Each constraint should be :obj:`(min_iou, max_iou)` format.
            If you set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`,
            it means not limited.
            If this argument is not specified, :obj:`((0.1, None), (0.3, None),
            (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
        max_trial (int): The maximum number of trials to be conducted
            for each constraint. If this function
            can not find any region that satisfies the constraint in
            :math:`max\_trial` trials, this function skips the constraint.
            The default value is :obj:`50`.
        return_param (bool): If :obj:`True`, this function returns
            information of intermediate values.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`img` that is cropped from the input
        array.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **constraint** (*tuple*): The chosen constraint.
        * **y_slice** (*slice*): A slice in vertical direction used to crop \
            the input image.
        * **x_slice** (*slice*): A slice in horizontal direction used to crop \
            the input image.

    """

    if constraints is None:
        constraints = (
            (0.1, None),
            (None, 1),
        )

    _, H, W = img.shape
    params = [{
        'constraint': None, 'y_slice': slice(0, H), 'x_slice': slice(0, W)}]

    if len(bbox) == 0:
        constraints = list()

    for min_iou, max_iou in constraints:
        if min_iou is None:
            min_iou = 0
        if max_iou is None:
            max_iou = 1

        for _ in six.moves.range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(H * scale / np.sqrt(aspect_ratio))
            crop_w = int(W * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(H - crop_h)
            crop_l = random.randrange(W - crop_w)
            crop_bb = np.array((
                crop_t, crop_l, crop_t + crop_h, crop_l + crop_w))

            iou = utils.bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou and iou.sum() > 0.05:
                params.append({
                    'constraint': (min_iou, max_iou),
                    'y_slice': slice(crop_t, crop_t + crop_h),
                    'x_slice': slice(crop_l, crop_l + crop_w)})
                break

    param = random.choice(params)
    img = img[:, param['y_slice'], param['x_slice']]

    if return_param:
        return img, param
    else:
        return img
