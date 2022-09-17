# from absl import flags
# from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from yolov3_tf2.utils import broadcast_iou

DEFAULT_MAX_BOXES = 100
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_SCORE_THRESHOLD = 0.5

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169), (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def DarknetTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    # Input: Darknet layer
    # Output shape is: (batchsize, n, n, 3, classes+5)
    #   where (n, n) depends on the YOLO layer (13,13) (26,25) (52,52)
    #   and YOLO has 3 bboxes (aka anchors) per cell
    #   and lastly (x1, y1, x2, y2, class_pred) + (..., prob_for_each_class, ...) as the last dim
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])  # Declare input, without batch dimension
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)

        x = Lambda(
            lambda x: tf.reshape(
                x,
                # Output: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
                (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)
            )
        )(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output


# As tensorflow lite doesn't support tf.size used in tf.meshgrid, 
# we reimplemented a simple meshgrid function that use basic tf function.
def _meshgrid(n_a, n_b):

    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))

    # Outputs the bbox predicted by every cell, along with the cell's objectness, and class_probabilities

    grid_size = tf.shape(pred)[1:3]

    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)
    # box_xy and box_wy are: (batch_size, grid, grid, 3, 2) where last dim has x,y or w,h
    # objectness is: (batch_size, grid, grid, 3, 1) where last dim has the pred class id
    # class_probs is: (batch_size, grid, grid, 3, num_classes) where last dim has the each class's pred prob

    # Comments use the example of batch_size=16, grid=13x13, anchors=3, classes=12

    box_xy = tf.sigmoid(box_xy)                      # tf.Tensor: shape=(16, 13, 13,  3,  2), dtype=float32
    objectness = tf.sigmoid(objectness)              # tf.Tensor: shape=(16, 13, 13,  3,  2), dtype=float32
    class_probs = tf.sigmoid(class_probs)            # tf.Tensor: shape=(16, 13, 13,  3,  1), dtype=float32
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss, tf.Tensor: shape=(16, 13, 13,  3,  4), dtype=float32

    # !!! grid[x][y] == (y, x)
    grid = _meshgrid(grid_size[1], grid_size[0])    # Creates a tuple of 2 tensors (nxn, nxn), where n is 13, 26, or 52 for YOLO
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    # grid is [gx, gy, 1, 2]
    # This is a grid representing the posible cell positions within the image
    #   Ex: x=13 is a <tf.Tensor: shape=(13, 13, 1, 2), dtype=int32>
    #   array([[[[ 0,  0]],
    #
    #           [[ 1,  0]],
    #
    #           [[ 2,  0]],
    #               ...

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)  # tf.Tensor: shape=(16, 13, 13,  3,  2), dtype=float32
    # This converts the bbox xy float center points that are cell-relative, to image-relative float center points
    # Example:
    #   If a cell's predicted bbox center is at x=0.02, y=0.15
    #   and this cell is at x=3, y=7 in the grid (assuming 13x13 grid)
    #   Then this equation above adds the cell's position in the grid to the float center points
    #   --> bbox x=(0.02 + 3), y=(0.15 + 7) = (3.02, 7.15)
    #   Then we divide by the grid size to get the float point positions on the full image
    #  --> bbox x=(3.02 / 13), y=(7.15 / 13) = (0.2323, 0.55) so about 1/4th across and 1/5 down the image

    box_wh = tf.exp(box_wh) * anchors
    # Predicted w and h are represented as log(dw) and log(dh) of the difference with the anchor's w, h
    # So we compute the exponential of the predicted values, and multiple by the anchor w, h to
    # convert back to image float w, h values

    # Compute all of the bounding box's x1, y1, x2, y2 points (still in floats)
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box

# def yolo_boxes(pred, anchors, classes):
#     # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
#     grid_size = tf.shape(pred)[1:3]
#     box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

#     box_xy = tf.sigmoid(box_xy)
#     objectness = tf.sigmoid(objectness)
#     class_probs = tf.sigmoid(class_probs)
#     pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

#     # !!! grid[x][y] == (y, x)
#     grid = _meshgrid(grid_size[1],grid_size[0])
#     grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

#     box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
#     box_wh = tf.exp(box_wh) * anchors

#     box_x1y1 = box_xy - box_wh / 2
#     box_x2y2 = box_xy + box_wh / 2
#     bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

#     return bbox, objectness, class_probs, pred_box


# def yolo_nms(outputs, anchors, masks, classes, max_boxes, iou_threshold, score_threshold):
def yolo_nms(outputs, classes, max_boxes, score_threshold, iou_threshold, soft_nms_sigma):
    # "outputs" is Tensor with 3 tuples like
    #   ( (bbox, objectness, class_probs), (...), (...) )
    # one for yolo output layer

    # print(tf.shape(outputs[0]))

    # boxes, conf, type
    b, c, t = [], [], []

    # Comments use the example of batch_size=16, grid=13x13, anchors=3, classes=12

    # For each tuple (output layer) in "outputs"
    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))  # <tf.tensor: shape=[16, 507, 4]>
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))  # <tf.tensor: shape=[16, 507, 1]>
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))  # <tf.tensor: shape=[16, 507, 12]>
        # 13x13x3 = 507, every cell in the output layer

    # (507 cells per layer) x (3 output layers) = 1521
    bbox = tf.concat(b, axis=1)         # <tf.Tensor: shape=[16 1521 4], dtype=float32>
    confidence = tf.concat(c, axis=1)   # <tf.Tensor: shape=[16 1521 1], dtype=float32>
    class_probs = tf.concat(t, axis=1)  # <tf.Tensor: shape=[16 1521 12], dtype=float32>

    # If we only have one class, do not multiply by class_prob (always 0.5)
    if classes == 1:
        scores = confidence
    else:
        scores = confidence * class_probs   # confidence is broadast to fit class_probs

    dscores = tf.squeeze(scores, axis=0)
    scores = tf.reduce_max(dscores, [1])
    bbox = tf.reshape(bbox, (-1, 4))
    classes = tf.argmax(dscores, 1)
    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=bbox,
        scores=scores,
        max_output_size=max_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        soft_nms_sigma=soft_nms_sigma
    )

    num_valid_nms_boxes = tf.shape(selected_indices)[0]

    selected_indices = tf.concat([selected_indices, tf.zeros(max_boxes - num_valid_nms_boxes, tf.int32)], 0)
    selected_scores = tf.concat([selected_scores, tf.zeros(max_boxes - num_valid_nms_boxes, tf.float32)], -1)

    boxes = tf.gather(bbox, selected_indices)
    boxes = tf.expand_dims(boxes, axis=0)
    scores = selected_scores
    scores = tf.expand_dims(scores, axis=0)
    classes = tf.gather(classes, selected_indices)
    classes = tf.expand_dims(classes, axis=0)
    valid_detections = num_valid_nms_boxes
    valid_detections = tf.expand_dims(valid_detections, axis=0)

    return boxes, scores, classes, valid_detections


def YoloV3(size: int = None, channels:int = 3, anchors: np.ndarray = yolo_anchors,
           masks: np.ndarray = yolo_anchor_masks, classes: int = 80, training: bool = False,
           max_boxes: int = 100, iou_threshold: float = 0.5,
           score_threshold: float = 0.5, soft_nms_sigma: float = 0.0
):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        # Output from all YOLO layers to use with the YoloLoss function
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    else:
        # Output bounding box predictions useable for drawing or object localizing
        boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
        boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
        boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)

        outputs = Lambda(
            # lambda x: yolo_nms(x, anchors, masks, classes, max_boxes, iou_threshold, score_threshold),
            lambda x: yolo_nms(x, classes, max_boxes, score_threshold, iou_threshold, soft_nms_sigma),
            name='yolo_nms'
        )((boxes_0[:3], boxes_1[:3], boxes_2[:3]))  # Input (bbox, objectness, class_probs) from yolo_boxes()

    # Outputs: see output of yolo_nms()
    return Model(inputs, outputs, name='yolov3')


def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False,
               max_boxes=DEFAULT_MAX_BOXES, iou_threshold=DEFAULT_IOU_THRESHOLD,
               score_threshold=DEFAULT_SCORE_THRESHOLD
):
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='yolo_darknet')(x)

    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes, max_boxes, iou_threshold, score_threshold),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            fn_output_signature=tf.float32
        )
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss
