# The image augmentation code in this module is reproduced significantly from the blog
# titled "Data Augmentation for Bounding Boxes: Rethinking Image Transforms for Object Detection",
# by Ayoosh Kathuria, published at the locations below.
#
#   https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
#   https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
#   https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
#
# With the sample code published at:
#   https://github.com/Paperspace/DataAugmentationForObjectDetection
#

import cv2
import numpy as np
import random
from typing import List, Tuple


def draw_rect(img: np.ndarray, bboxes: Tuple or List, color: int = None) -> np.ndarray:
    """Draw bouding boxes (aka cords) on an image

    Args:
        im (np.ndarray): The image to draw on
        bboxes (List): List of bounding boxes as [[xmin, ymin, xmax, ymax, class_id], ...]
        color (int, optional): RGB color for the bounding box. Defaults to None (white).

    Returns:
        np.ndarray: The image with bounding boxes drawn
    """
    img = img.copy()

    if not color:
        color = [255, 255, 255]
    for bbox in bboxes:

        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])

        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])

        img = cv2.rectangle(img.copy(), pt1, pt2, color, int(max(img.shape[:2]) / 200))
    return img


def _bbox_area(bbox):
    """From https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/

    Args:
        bbox (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def _clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image

    Derived from https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/

    Parameters
    ----------

    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`

    alpha: float
        If the fraction of a bounding box left in the image after being clipped is
        less than `alpha` the bounding box is dropped.

    Returns
    -------

    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    ar_ = (_bbox_area(bbox))

    # Remove bboxes with 9 area
    bbox = bbox[ar_ > 0]
    ar_ = ar_[ar_ > 0]

    assert np.all(ar_ > 0), f'Found zero in ar_: {ar_}'

    # x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    # y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    # x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    # y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    # print(f"ar_: \n{ar_}\n")

    # print(f"x_min: \n{x_min}")
    # print(f"y_min: \n{y_min}")
    # print(f"x_max: \n{x_max}")
    # print(f"y_max: \n{y_max}")
    # print()

    # Contrain the bbox points to be inside of the image's pixel limits
    x1_x2 = np.minimum(np.maximum(bbox[:, (0, 2)], clip_box[0] + 1), clip_box[2] - 1)
    y1_y2 = np.minimum(np.maximum(bbox[:, (1, 3)], clip_box[1] + 1), clip_box[3] - 1)

    # print(f"x1_x2: \n{x1_x2}")
    # print(f"y1_y2: \n{y1_y2}")

    # bbox_orig_hstack = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))
    # print(f"bbox_orig after hstack:\n{bbox_orig_hstack}\n")

    bbox = np.hstack((
        x1_x2[:, 0].reshape(-1, 1),
        y1_y2[:, 0].reshape(-1, 1),
        x1_x2[:, 1].reshape(-1, 1),
        y1_y2[:, 1].reshape(-1, 1),
        bbox[:, 4:])
    )
    # print(f"\nbbox after hstack:\n{bbox}\n")


    delta_area = ((ar_ - _bbox_area(bbox)) / ar_)   # Pct of bbox area removed, for each bbox
    # print(f"delta_area: \n{delta_area}\n")

    # mask = (delta_area < (1 - alpha)).astype(int)
    mask = (delta_area < alpha)   # Bool mask of which bboxes should be removed (True)
    # print(f"mask: \n{mask}\n")

    # bbox = bbox[mask == 1, :]
    bbox = bbox[mask, :]   # Keep only the bboxes that are True
    return bbox


def _rotate_im(image, angle):
    """Rotate the image.

    From: https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    return image


def _get_corners(bboxes):

    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def _rotate_box(corners, angle, cx, cy, h, w):

    """Rotate the bounding box.

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def _get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def aug_horizontal_flip(img: np.ndarray, bboxes: List, p: float = 1.0) -> Tuple[np.ndarray, List[int]]:
    """From https://blog.paperspace.com/data-augmentation-for-bounding-boxes/

    Args:
        img_file_name (str): _description_
        p (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    bboxes = bboxes.copy()
    if random.random() < p:
        img_center = np.array(img.shape[:2])[::-1] // 2    # Switch rc to xy, and calc center size
        img_center = np.hstack((img_center, img_center))    # (xcenter, ycenter, xcenter, ycenter)

        img = img[:, ::-1, :]    # Flip img's pixels horizontally

        if bboxes.size > 0:
            bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

            box_w = abs(bboxes[:, 0] - bboxes[:, 2])

            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w

    return img, bboxes


def aug_scale_random(
    img: np.ndarray, bboxes: np.ndarray, min_scale_factor: float = -0.8, max_scale_factor: float = 1.0, keep_aspect: bool = False
) -> Tuple[np.ndarray, List[int]]:
    """From https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/

    Args:
        img_file_name (str): _description_
        min_scale_factor (float, optional): _description_. Defaults to -1.0.
        max_scale_factor (float, optional): _description_. Defaults to 1.0.
        keep_aspect (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[np.ndarray, List[int]]: _description_
    """
    bboxes = bboxes.copy()

    # If scale_factor is < -1, then the resulting size will be < 0.0 and there will be no image left
    min_scale_factor = max(-1, min_scale_factor)

    img_shape = img.shape

    if keep_aspect:
        scale_x = random.uniform(min_scale_factor, max_scale_factor)
        scale_y = scale_x
    else:
        scale_x = random.uniform(min_scale_factor, max_scale_factor)
        scale_y = random.uniform(min_scale_factor, max_scale_factor)

    resize_scale_x = 1 + scale_x
    resize_scale_y = 1 + scale_y

    img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

    canvas = np.zeros(img_shape, dtype=np.uint8)
    y_lim = int(min(resize_scale_y, 1) * img_shape[0])
    x_lim = int(min(resize_scale_x, 1) * img_shape[1])
    canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]
    img = canvas

    if bboxes.size > 0:
        bboxes = bboxes.astype(np.float64)
        bboxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
        bboxes = bboxes.astype(np.int64)
        bboxes = _clip_box(bboxes, [0, 0, 1 + img_shape[1], img_shape[0]], 0.25)

    return img, bboxes


def aug_translate_random(
    img: np.ndarray, bboxes: np.ndarray, min_translate_factor: float = -0.5, max_translate_factor: float = 0.5, keep_aspect: bool = False
) -> Tuple[np.ndarray, List[int]]:
    """From https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/

    Args:
        img_file_name (str): _description_
        min_translate_factor (float, optional): _description_. Defaults to -0.2.
        max_translate_factor (float, optional): _description_. Defaults to 1.0.
        keep_aspect (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[np.ndarray, List[int]]: _description_
    """
    bboxes = bboxes.copy()

    min_translate_factor = max(-1, min_translate_factor)
    max_translate_factor = min(1, max_translate_factor)

    img_shape = img.shape

    translate_factor_x = random.uniform(min_translate_factor, max_translate_factor)
    translate_factor_y = random.uniform(min_translate_factor, max_translate_factor)

    # translate_factor_x = -0.5     # debugging neg translate
    # translate_factor_y = 0.5

    if keep_aspect:
        translate_factor_y = translate_factor_x

    canvas = np.zeros(img_shape).astype(np.uint8)

    corner_x = int(translate_factor_x * img.shape[1])
    corner_y = int(translate_factor_y * img.shape[0])

    # Change the origin to the top-left corner of the translated box
    orig_box_cords = [max(0, corner_y), max(corner_x, 0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1], corner_x + img.shape[1])]

    mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]), :]
    canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
    img = canvas

    if bboxes.size > 0:
        bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]
        # print("translated bboxes, before clipping")
        # print(bboxes)
        # print()
        bboxes = _clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.25)

    return img, bboxes


def aug_rotate_random(
    img: np.ndarray, bboxes: np.ndarray, min_angle: float = -180.0, max_angle: float = 180.0
) -> Tuple[np.ndarray, List[int]]:
    """https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/

    Args:
        img_file_name (str): _description_
        min_angle (float, optional): _description_. Defaults to -0.5.
        max_angle (float, optional): _description_. Defaults to 0.5.

    Returns:
        Tuple[np.ndarray, List[int]]: _description_
    """
    bboxes = bboxes.copy()

    angle = random.uniform(min_angle, max_angle)

    w, h = img.shape[1], img.shape[0]
    cx, cy = w // 2, h // 2

    img = _rotate_im(img, angle)

    if bboxes.size > 0:
        corners = _get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:, 4:]))
        corners[:, :8] = _rotate_box(corners[:, :8], angle, cx, cy, h, w)

        new_bbox = _get_enclosing_box(corners)

        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h

        new_bbox = new_bbox.astype(np.float64)
        new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
        new_bbox = new_bbox.astype(np.int64)
        bboxes = new_bbox
        bboxes = _clip_box(bboxes, [0, 0, w, h], 0.25)

    img = cv2.resize(img, (w, h))
    return img, bboxes


def aug_shear_random(
    img: np.ndarray, bboxes: np.ndarray, min_shear_factor: float = -0.8, max_shear_factor: float = 0.8
) -> Tuple[np.ndarray, List[int]]:
    bboxes = bboxes.copy()

    shear_factor = random.uniform(min_shear_factor, max_shear_factor)

    # img, bboxes = read_image_with_yolo_annotations(img_file_name)

    w, h = img.shape[1], img.shape[0]

    if shear_factor < 0:
        img, bboxes = aug_horizontal_flip(img, bboxes, p=1.0)

    M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])

    nW = img.shape[1] + abs(shear_factor * img.shape[0])

    img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

    if bboxes.size > 0:
        bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

    if shear_factor < 0:
        img, bboxes = aug_horizontal_flip(img, bboxes, p=1.0)

    img = cv2.resize(img, (w, h))

    if bboxes.size > 0:
        scale_factor_x = nW / w

        bboxes = bboxes.astype(np.float64)
        bboxes[:, :4] /= [scale_factor_x, 1, scale_factor_x, 1]
        bboxes = bboxes.astype(np.int64)

    return img, bboxes
