#!/usr/bin/env python3
"""class Yolo that uses the Yolo v3 algorithm to perform object detection"""
import tensorflow.keras as K
import numpy as np


class Yolo():
    """class Yolo that uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """class Yolo that uses Yolo v3 algorithm to perform object detection
        Args:
            model_path is the path to where a Darknet Keras model is stored
            classes_path: path to where the list of class names used for the
            Darknet model, listed in order of index, can be found
            class_t: float representing the box score threshold for the initial
                filtering step
            nms_t: float representing the IOU threshold for non-max suppression
            anchors: numpy.ndarray, shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes:
                outputs: number of outputs (predictions) made by Darknet model
                anchor_boxes: number of anchor boxes used for each prediction
                2 => [anchor_box_width, anchor_box_height]
        """
        class_names = []
        with open(classes_path, 'r') as f:
            for line in f:
                class_names.append(line.strip())

        self.model = K.models.load_model(model_path)
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """outputs a list of numpy.ndarrays containing predictions from the
            Darknet model for a single image
        Args:
            outputs: list of numpy.ndarrays containing the predictions:
                Each output will have the shape
                (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
                     grid_height & grid_width =>
                     the height and width of the grid used for the output
                     anchor_boxes => the number of anchor boxes used
                     4 => (t_x, t_y, t_w, t_h)
                     1 => box_confidence
                     classes => class probabilities for all classes
            image_size: numpy.ndarray image???s original size [height, width]
        Returns: tuple of (boxes, box_confidences, box_cls_probs):
                 boxes: np shape (grid_height, grid_width, anchor_boxes, 4)
                 box_confidences: np.shape (grid_height, grid_width,
                    anchor_boxes, 1) has box confidences for each output
                 box_cls_probs:(grid_height, grid_width, anchor_boxes,
                    classes) box???s class probabilities for each output
            """
        boxes = []
        box_confidence = []
        box_cls_probs = []

        for i in range(len(outputs)):
            grid_h, grid_w, box_q, _ = outputs[i].shape

            box_conf = 1 / (1 + np.exp(-(outputs[i][:, :, :, 4:5])))
            box_confidence.append(box_conf)
            box_prob = 1 / (1 + np.exp(-(outputs[i][:, :, :, 5:])))
            box_cls_probs.append(box_prob)

            box_xy = 1 / (1 + np.exp(-(outputs[i][:, :, :, :2])))
            box_wh = np.exp(outputs[i][:, :, :, 2:4])
            anchors = self.anchors.reshape(1, 1,
                                           self.anchors.shape[0], box_q, 2)
            box_wh = box_wh * anchors[:, :, i, :, :]

            col = np.tile(np.arange(0, grid_w),
                          grid_h).reshape(grid_h, grid_w)
            row = np.tile(np.arange(0, grid_h),
                          grid_w).reshape(grid_w, grid_h).T
            col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)
            row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)
            grid = np.concatenate((col, row), axis=3)

            box_xy += grid
            box_xy /= (grid_w, grid_h)
            input_h = self.model.input.shape[2].value
            input_w = self.model.input.shape[1].value
            box_wh /= (input_w, input_h)
            box_xy -= (box_wh / 2)
            box_xy1 = box_xy
            box_xy2 = box_xy1 + box_wh
            box = np.concatenate((box_xy1, box_xy2), axis=-1)

            box[..., 0] *= image_size[1]
            box[..., 2] *= image_size[1]
            box[..., 1] *= image_size[0]
            box[..., 3] *= image_size[0]

            boxes.append(box)
        return (boxes, box_confidence, box_cls_probs)
