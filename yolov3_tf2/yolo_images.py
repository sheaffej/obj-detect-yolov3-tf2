import glob
import numpy as np
import os
import random
from typing import Callable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm


DEFAULT_DATA_DIR = './data/voc2012_raw/VOCdevkit/VOC2012/'  # path to raw PASCAL VOC dataset
DEFAULT_SPLIT_ENUM = 'train'                                # specify split as 'train' or 'val'
DEFAULT_OUTPUT_FILE = './data/voc2012_train.tfrecord'       # output dataset
DEFAULT_CLASSES_FILE = './data/voc2012.names'               # classes file name


# def build_example(image_file: str, annotations_file: str, class_map: dict, image_size: int = 416):
def build_example(image_file: str, annotations_file: str, image_size: int = 416):
    cv2_img = cv2.imread(image_file)
    cv2_img = cv2.resize(cv2_img, (image_size, image_size))
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    _, im_buf_arr = cv2.imencode(".png", cv2_img)
    img_raw = im_buf_arr.tobytes()

    height = cv2_img.shape[0]
    width = cv2_img.shape[1]
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    # classes_text = []

    with open(annotations_file, 'r') as af:
        for line in af.readlines():
            cols = line.split(' ')
            class_id = int(cols[0])
            xcenter = float(cols[1])
            ycenter = float(cols[2])
            xwidth = float(cols[3])
            yheight = float(cols[4])

            _xmin = xcenter - (xwidth / 2)
            _ymin = ycenter - (yheight / 2)
            xmin.append(_xmin)
            ymin.append(_ymin)
            xmax.append(_xmin + xwidth)
            ymax.append(_ymin + yheight)
            classes.append(class_id)
            # classes_text.append(class_map[class_id].encode('utf8'))

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_file.encode('utf8')])),
                # 'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                #     annotation['filename'].encode('utf8')])),
                # 'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                # 'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
                'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
                'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
                'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
                'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
                # 'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
                'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
                # 'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
                # 'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
                # 'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
            }
        )
    )
    return example


def yolo_img_to_tfrecords(
    data_dir: str,
    classes_file: str,
    output_dir: str,
    min_file_mb_size: int = 10,
    pct_validation: float = .1,
    pct_test: float = .0,
    file_prefix: str = "yolov3",
    image_size: int = 416,
    augmentation_fn: Callable = None,
    shuffle_seed: int = 42,
    limit: int = None
):
    """Convert images with YOLO annotations files into TFRecord data file(s)

    Args:
        data_dir (str): The directory containing the image and annotation files
        classes_file (str): The file containg the class names
        output_dir (str): The directory to write TFRecord files to. This directory will contain
        min_file_mb_size (int): Once a TFRecord file exceeds this size (approx), a new file will be created
        pct_validation (float): Fraction of the input data that will be prepared as validation data
        pct_test (float): Fraction of the input data that will be prepared as test data (no labels)
        augmentation_fn (Callable): Function to call to perform data augmentation
        limit (int): Number of input records to process before exiting
    """

    assert (pct_validation + pct_test) < 1.0

    class_map = {idx: name for idx, name in enumerate(open(classes_file).read().splitlines())}
    # print(f"Class mapping loaded: {class_map}")

    imgfiles = glob.glob(os.path.join(data_dir, 'IMG_*.png'))

    # Shuffle
    random.Random(shuffle_seed).shuffle(imgfiles)

    # Limit
    if limit:
        imgfiles = imgfiles[0:limit]
        print(f"Limiting to {limit} input images")

    # Split
    num_train_imgs = int(len(imgfiles) * (1.0 - pct_validation - pct_test))
    num_val_imgs = int(len(imgfiles) * pct_validation)
    num_test_imgs = int(len(imgfiles) * pct_test)
    assert (num_train_imgs + num_val_imgs + num_test_imgs) <= len(imgfiles)
    assert num_train_imgs > 0

    print(f"num_train_imgs: {num_train_imgs}, num_val_imgs: {num_val_imgs}, num_test_imgs: {num_test_imgs}")

    img_split_sets = {}
    if num_val_imgs > 0 and num_test_imgs > 0:
        train, val, test = np.split(imgfiles, [num_train_imgs, (num_train_imgs + num_val_imgs)])
        img_split_sets = {"train": train, "validation": val, "test": test}
    elif num_val_imgs > 0 and num_test_imgs == 0:
        train, val = np.split(imgfiles, [num_train_imgs])
        img_split_sets = {"train": train, "validation": val}
    elif num_val_imgs == 0 and num_test_imgs > 0:
        train, test = np.split(imgfiles, [num_train_imgs])
        img_split_sets = {"train": train, "test": test}

    approx_resize_img_bytes = image_size * image_size * 3 * 0.7  # Assuming each pixel 1 byte per channel

    for split_name, split_imgs in img_split_sets.items():
        print(f"{split_name}: {len(split_imgs)} images")

        split_dir = os.path.join(output_dir, split_name)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

        file_idx = 1
        file_size = 0
        outfile = None
        for img_file in tqdm(split_imgs):
            if outfile is None:
                outfile = os.path.join(split_dir, f"{file_prefix}_{split_name}_{file_idx}.tfrecord")
                writer = tf.io.TFRecordWriter(outfile)

            filebase = os.path.splitext(img_file)[0]
            annofile = f"{filebase}.txt"
            if os.path.exists(annofile):
                tf_example = build_example(image_file=img_file, annotations_file=annofile, class_map=class_map)
                writer.write(tf_example.SerializeToString())
                file_size += approx_resize_img_bytes

            if file_size > min_file_mb_size * pow(10, 6):
                writer.close()
                # print(f"TFRecord written to {outfile}")
                file_size = 0
                outfile = None
                file_idx += 1

        if outfile is not None:
            writer.close()
            # print(f"TFRecord written to {outfile}")
            outfile = None

        print()


def bbox_yolo_to_xyminmax(
    img_width: int, img_height: int,
    xcenter: float, ycenter: float,
    width: float, height: float
) -> List[int]:
    """Convert YOLO annotation format to XY min/max format

    Args:
        img_width (int): Image pixel size on the X-axis
        img_height (int): Image pixel size on the Y-axis
        xcenter (float): Fractional/float position of the box's center along the X-axis
        ycenter (float): Fractional/float position of the box's center along the Y-axis
        width (float): Fractional/float size of the box along the X-axis
        height (float): Fractional/float size of the box along the X-axis

    Returns:
        List: List of bounding boxes in the format [xmin, ymin, xmax, ymax] in absolute pixels
    """
    # These are still fractional / float values
    xmin = xcenter - (width / 2)
    ymin = ycenter - (height / 2)
    xmax = xmin + width
    ymax = ymin + height

    # Convert to absolute pixes and return
    return [
        int(xmin * img_width),
        int(ymin * img_height),
        min(int(xmax * img_width), img_width),     # And clip to max image width,
        min(int(ymax * img_height), img_height)   # And clip to max image height
    ]


def bbox_xyminmax_to_yolo(
    img_width: int, img_height: int,
    xmin: int, ymin: int, xmax: int, ymax: int
) -> List:
    xcenter_f = float(((xmin + xmax) / 2.0) / img_width)
    ycenter_f = float(((ymin + ymax) / 2.0) / img_height)
    width_f = float((xmax - xmin) / img_width)
    height_f = float((ymax - ymin) / img_height)
    return [
        xcenter_f, ycenter_f, width_f, height_f
    ]


def parse_yolo_anno_file(yolo_file: str, img_width: int, img_height: int, yolo_boxes: bool = False) -> List:
    # The expected output format is
    #   [xmin, ymin, xmax, ymax, class_id]
    # in absolute pixels (not fractional/float)
    bboxes = []
    with open(yolo_file, 'r') as f:
        # print("===== YOLO format boxes =====")
        for line in f.readlines():
            cols = line.split(' ')
            class_id = int(cols[0])
            xcenter = float(cols[1])
            ycenter = float(cols[2])
            width = float(cols[3])
            height = float(cols[4])

            bbox = []
            if yolo_boxes:
                bbox = [xcenter, ycenter, width, height]
            else:
                bbox = bbox_yolo_to_xyminmax(
                    img_width, img_height, xcenter, ycenter, width, height
                )
            bbox.append(class_id)
            bboxes.append(bbox)
    return bboxes


def write_yolo_anno_file(yolo_file: str, bboxes: List, img_width: int, img_height: int):
    with open(yolo_file, 'w') as f:
        if bboxes.size == 0:
            return  # There are no annotations for this image. Write an empty file

        assert bboxes.shape[1] == 5, f"Expecting 5 values in bbox, found {bboxes}"

        for bbox in bboxes:
            xcenter, ycenter, width, height = bbox_xyminmax_to_yolo(
                img_width, img_height,
                bbox[0], bbox[1], bbox[2], bbox[3]
            )
            class_id = bbox[4]
            f.write(f"{class_id} {xcenter} {ycenter} {width} {height}\n")


def img_to_yolo_filename(img_filename: str) -> str:
    file_no_ext = os.path.splitext(img_filename)[0]
    return f"{file_no_ext}.txt"


def read_image_with_yolo_annotations(img_filename: str, resize: Tuple[int] = None, yolo_boxes: bool = False, bgr2rgb: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Read an image file and its YOLO-format annotations. Bounding boxes returned are in the format
    of [xmin, ymin, xmax, ymax, class_id] in absolute pixel positions.

    Args:
        img_filename (str): File name of the image. The annotation file is expected at the same
                            path and file name, with .txt extension.
        resize (Tuple[int]): (w, h) to resize the image to.
        yolo_boxes (bool): False to return boxes as (x1, y1, x2, y2). True to return yolo boxes (xc, yc, w, h).

    Returns:
        Tuple[np.ndarray, List]: The image an a numpy array, and a list of bounding boxes.
    """
    img = cv2.imread(img_filename)

    if resize:
        img = cv2.resize(img, resize)

    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes = parse_yolo_anno_file(
        yolo_file=img_to_yolo_filename(img_filename),
        img_width=img.shape[1], img_height=img.shape[0],
        yolo_boxes=yolo_boxes
    )
    bboxes = np.array(bboxes)
    return img, bboxes


def write_augmented_image_files(
    new_img: np.ndarray, bboxes: np.ndarray, orig_filename: str,
    augmentation_name: str, output_dir: str, verbose: bool = False
) -> Tuple[str]:
    """Write a new image file that has been augmented

    Args:
        new_img (np.ndarray): The augmented image to write
        orig_filename (str): The file name of the original (unaugmented) image
        augmentation_type (str): A string label for the type of augmentation. Becomes part of the new image's file name
        output_dir (str): The directory in which to write the augmented file

    Returns:
        List[str, str]: The file name of the augmented image file, and yolo annotation file written

    """
    augmentation_name = augmentation_name.replace(' ', '').lower()   # clean up aug_type string
    file_basename = os.path.join(
        output_dir,
        os.path.splitext(
            os.path.basename(orig_filename))[0] + f".{augmentation_name}"   # "/path/to/IMG_XXXX.augtype"
    )
    new_img_filename = f"{file_basename}.png"
    new_yolo_filename = f"{file_basename}.txt"
    cv2.imwrite(new_img_filename, new_img)
    write_yolo_anno_file(
        yolo_file=new_yolo_filename,
        bboxes=bboxes,
        img_width=new_img.shape[1],
        img_height=new_img.shape[0]
    )

    if verbose:
        print(f"Wrote {new_img_filename}/.txt")

    return new_img_filename, new_yolo_filename


def write_training_tfrecords(
    items: list, type_name: str, output_dir: str,
    file_prefix: str = '', min_file_mb: int = 10, image_size: int = 416
):
    """Write tf.train.Example records to TFRecord files. Files are created approximately `min_file_mb` in size.

    Args:
        items (list):
        type_name (str): _description_
        output_dir (str): _description_
        file_prefix (str, optional): _description_. Defaults to ''.
        image_size (int, optional): _description_. Defaults to 416.
        min_file_mb (int, optional): _description_. Defaults to 10.
    """
    file_idx = 1
    file_size = 0
    outfile = None
    file_prefix = f"{file_prefix}_" if len(file_prefix) > 0 else file_prefix

    approx_resize_img_bytes = image_size * image_size * 3 * 0.7  # Assuming each pixel 1 byte per channel

    for img_file in tqdm(items, desc=f"{type_name.title()}"):
        if outfile is None:
            outfile = os.path.join(output_dir, f"{file_prefix}{type_name}_{file_idx}.tfrecord")
            writer = tf.io.TFRecordWriter(outfile)

        annofile = img_to_yolo_filename(img_file)

        if os.path.exists(annofile):
            tf_example = build_example(image_file=img_file, annotations_file=annofile)
            writer.write(tf_example.SerializeToString())
            file_size += approx_resize_img_bytes

        if file_size > min_file_mb * pow(10, 6):
            writer.close()
            file_size = 0
            outfile = None
            file_idx += 1

    if outfile is not None:
        writer.close()
        outfile = None
