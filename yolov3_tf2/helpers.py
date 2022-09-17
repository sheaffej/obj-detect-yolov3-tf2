from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from yolov3_tf2.yolo_images import read_image_with_yolo_annotations

RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


def train_val_test_split(
    list_to_split: List[any],
    val_fraction: float, test_fraction: float = 0.0,
    shuffle: bool = True, verbose: bool = False
) -> dict:
    """Splits a list of items into 2 or 3 sets: `train`, and one or both of `validation` and `test`.
    The values of `val_fraction` and `test_fraction` must be less than 1.0. The remainder fraction
    will be the train split (`train = 1.0 - (val_fraction + test_fraction)`).

    Args:
        list_to_split (List[any]): The list to be split
        val_fraction (float): The fractional (`0.0 <= s < 1.0`) for the validation split
        test_fraction (float, optional): The fractional (`0.0 <= s < 1.0`) for the test split. Defaults to 0.0.
        shuffle (bool, optional): Set `True` to shuffle before splitting. Defaults to True.
        verbose (bool, optional): Print additional progress. Defaults to False.

    Returns:
        dict: Dictionary of {'train': train_items, 'validation': val_items, 'test': test_items}
    """
    num_train_items = int(len(list_to_split) * (1.0 - val_fraction - test_fraction))
    num_val_items = int(len(list_to_split) * val_fraction)
    num_test_items = int(len(list_to_split) * test_fraction)

    assert (num_train_items + num_val_items + num_test_items) <= len(list_to_split)
    assert num_train_items > 0

    if verbose:
        print(f"num_train_items: {num_train_items}, num_val_items: {num_val_items}, num_test_items: {num_test_items}")

    if shuffle:
        rng.shuffle(list_to_split)

    item_split_sets = {}
    if num_val_items > 0 and num_test_items > 0:
        train, val, test = np.split(list_to_split, [num_train_items, (num_train_items + num_val_items)])
        item_split_sets = {"train": train, "validation": val, "test": test}
    elif num_val_items > 0 and num_test_items == 0:
        train, val = np.split(list_to_split, [num_train_items])
        item_split_sets = {"train": train, "validation": val}
    elif num_val_items == 0 and num_test_items > 0:
        train, test = np.split(list_to_split, [num_train_items])
        item_split_sets = {"train": train, "test": test}

    return item_split_sets


def draw_image_bboxes(image: np.ndarray, bboxes: Tuple or List, color: int = None) -> np.ndarray:
    """Draw bouding boxes on an image

    Args:
        image (np.ndarray): The image one which to draw the bounding boxes
        bboxes (List): List of bounding boxes as [[xmin, ymin, xmax, ymax, class_id], ...]
        color (int, optional): RGB color for the bounding box. Defaults to None (red).

    Returns:
        np.ndarray: The image with bounding boxes drawn
    """
    image = image.copy()

    if not color:
        color = [255, 0, 0]
    for bbox in bboxes:

        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])

        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])

        image = cv2.rectangle(image, pt1, pt2, color, int(max(image.shape[:2]) / 200))
    return image


def display_image(image: np.ndarray, size: float = 10.0, dpi: int = 80, gray: bool = False):
    """Display an image with larger/custom size. The function will display as a grayscale image 
    if the image is a 2-dimensional ndarray. Or you can force it to display as grayscale using 
    the gray=True parameter.

    Args:
        image (np.ndarray): The image
        size (float, optional): The max width and height in inches (depends on the current DPI setting). Defaults to 10.0.
        dpi (int, optional): The DPI - pixels per inch. Defaults to 80.
        gray (bool, optional): Set to True to force a grayscale image. Defaults to False.
    """
    cmap = None
    if image.ndim == 2:
        cmap = "gray"

    fig = plt.figure(figsize=(size, size), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap=cmap)


def draw_yolo_annotatated_image(img_file_name: str = None, display: bool = True) -> None or np.ndarray:
    """Draw using matplotlib.pyplot.imshow the image and annotations bounding boxes.
    The YOLO annotations file is expected to exists in the same directory with the extention of .txt

    Args:
        img_file_name (str): The image file name to display. If None, the Numpy arrays are expected for `image` and `bboxes`.
        display (bool): True to display with matplotlib.pyplot.imshow. False to return the image as an numpy array.
    """

    image, bboxes = read_image_with_yolo_annotations(img_file_name)
    anno_img = draw_image_bboxes(image, bboxes, (255, 0, 0))

    if display:
        display_image(anno_img)
    else:
        return anno_img


