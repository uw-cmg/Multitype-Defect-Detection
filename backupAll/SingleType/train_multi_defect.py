import numpy as np

import chainer
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain
from chainercv import transforms
from utils import MultiDefectDetectionDataset
from utils import rotate_bbox, random_resize, random_distort, random_crop_with_bbox_constraints
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class Transform(object):
    # initial faster_rcnn
    def __init__(self, faster_rcnn):
        self.faster_rcnn = faster_rcnn
    # Initial datasets, H, W stores the hight and width of the image
    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape

        # random brightness and contrast
        img = random_distort(img)

        # rotate image
        # return a tuple whose elements are rotated image, param.
        # k (int in param)represents the number of times the image is rotated by 90 degrees.
        img, params = transforms.random_rotate(img, return_param=True)
        # restore the new hight and width
        _, t_H, t_W = img.shape
        # rotate bbox based on renewed parameters
        bbox = rotate_bbox(bbox, (H, W), params['k'])

#         # Random expansion:This method randomly place the input image on
#         # a larger canvas. The size of the canvas is (rH,rW), r is a random ratio drawn from [1,max_ratio].
#         # The canvas is filled by a value fill except for the region where the original image is placed.
#         if np.random.randint(2):
#             fill_value = img.mean(axis=1).mean(axis=1).reshape(-1,1,1)
#             img, param = transforms.random_expand(img, max_ratio=2, fill=fill_value, return_param=True)
#             bbox = transforms.translate_bbox(
#                 bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

#         # Random crop
#         # crops the image with bounding box constraints
#         img, param = random_crop_with_bbox_constraints(
#             img, bbox, min_scale=0.75, max_aspect_ratio=1.25, return_param=True)
#         # this translates bounding boxes to fit within the cropped area of an image, bounding boxes whose centers are outside of the cropped area are removed.
#         bbox, param = transforms.crop_bbox(
#             bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
#             allow_outside_center=False, return_param=True)
#         #assigning new labels to the bounding boxes after cropping
#         label = label[param['index']]
#         # if the bounding boxes are all removed,
#         if bbox.shape[0] == 0:
#             img, bbox, label = in_data
#         # update the height and width of the image
#         _, t_H, t_W = img.shape

        img = self.faster_rcnn.prepare(img)
        # prepares the image to match the size of the image to be input into the RCNN
        _, o_H, o_W = img.shape
        # resize the bounding box according to the image resize
        bbox = transforms.resize_bbox(bbox, (t_H, t_W), (o_H, o_W))

        # horizontally & vertical flip
        # simutaneously flip horizontally and vertically of the image
        img, params = transforms.random_flip(
            img, x_random=True, y_random=True, return_param=True)
        # flip the bounding box with respect to the parameter
        bbox = transforms.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'], y_flip=params['y_flip'])
        
        scale = o_H / t_H

        return img, bbox, label, scale


def main():
    bbox_label_names = ('loop')

    n_itrs = 70000
    n_step = 50000
    np.random.seed(0)
    train_data = MultiDefectDetectionDataset(split='train')
    test_data = MultiDefectDetectionDataset(split='test')
    proposal_params = {'min_size': 8}

    faster_rcnn = FasterRCNNVGG16(n_fg_class=2, pretrained_model='imagenet', ratios=[0.5, 1, 2],
                                  anchor_scales=[0.5, 1, 4, 8, 16], min_size=1024, max_size=1024,
                                  proposal_creator_params=proposal_params)
    faster_rcnn.use_preset('evaluate')
    model = FasterRCNNTrainChain(faster_rcnn)
    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=1e-3, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    train_data = TransformDataset(train_data, Transform(faster_rcnn))
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size=1, n_processes=None, shared_mem=100000000)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=0)
    trainer = training.Trainer(
        updater, (n_itrs, 'iteration'), out='result')
    trainer.extend(
        extensions.snapshot_object(model.faster_rcnn, 'snapshot_model_{.updater.iteration}.npz'), 
        trigger=(n_itrs/5, 'iteration'))
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(n_step, 'iteration'))
    log_interval = 50, 'iteration'
    plot_interval = 100, 'iteration'
    print_interval = 20, 'iteration'
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/roi_loc_loss',
         'main/roi_cls_loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'validation/main/map',
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=5))
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )
    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model.faster_rcnn, use_07_metric=True,
            label_names=bbox_label_names),
        trigger=ManualScheduleTrigger(
            [100, 500, 1000, 5000, 10000, 20000, 40000, 60000, n_step, n_itrs], 'iteration'))

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
