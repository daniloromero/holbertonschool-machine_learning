#!/usr/bin/env python3
"""class Yolo that uses the Yolo v3 algorithm to perform object detection"""
import tensorflow.keras as K


class Yolo:

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
        self.model = K.models.load_model(model_path)
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        with open(classes_path, 'r') as f:
            for line in f:
                class_names.append(line.strip())
