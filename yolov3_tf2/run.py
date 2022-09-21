from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from glob import glob
import numpy as np
import os
from tqdm import tqdm
from time import perf_counter
import time
from typing import Tuple, List

import albumentations as A
import cv2
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    History
)
import wandb
from wandb.keras import WandbCallback

from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny,
    YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, freeze_all
import yolov3_tf2.dataset as dataset_module
from yolov3_tf2.yolo_images import (
    read_image_with_yolo_annotations, write_augmented_image_files, write_training_tfrecords,
)
from yolov3_tf2.image_augmentation import (
    aug_horizontal_flip, aug_rotate_random, aug_scale_random,
    aug_shear_random, aug_translate_random
)


def _load_model(
    num_classes: int,
    weights: str,
    max_boxes: int = 100,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    soft_nms_sigma: float = 0.0,
    tiny: bool = False,
    verbose=False
):
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_device, True)
        except: # noqa
            pass    # This will happen if the GPU has already been set

    model: Model    # tf.keras.Model
    if tiny:
        model = YoloV3Tiny(
            classes=num_classes, max_boxes=max_boxes,
            iou_threshold=iou_threshold, score_threshold=score_threshold
        )
    else:
        model = YoloV3(
            classes=num_classes, max_boxes=max_boxes,
            iou_threshold=iou_threshold, score_threshold=score_threshold, soft_nms_sigma=soft_nms_sigma
        )

    model.load_weights(weights).expect_partial()
    if verbose:
        print('weights loaded')

    return model


def _setup_model(
    weights: str,
    num_classes: int,
    training: bool = False,     # When true, model's output is structured for YoloLoss function. When false, output is the predicted bounding boxes (i.e. for drawing/etc)
    weights_num_classes: int = None,
    size: int = 416,
    mode: str = 'fit',
    learning_rate: float = 1e-3,
    transfer: str = 'darknet',
    tiny: bool = False
) -> Tuple[Model, Optimizer, List, List, List]:

    # Let's declare some types so our IDE is more helpful
    model: Model
    optimizer: Optimizer

    if tiny:
        model = YoloV3Tiny(size, training=training, classes=num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(size, training=training, classes=num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # Configure the model for transfer learning
    if transfer is None:
        pass  # Nothing to do
    elif transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes
        # reset top layers
        if tiny:
            model_pretrained = YoloV3Tiny(
                size, training=True, classes=weights_num_classes or num_classes)
        else:
            model_pretrained = YoloV3(
                size, training=True, classes=weights_num_classes or num_classes)
        model_pretrained.load_weights(weights)

        if transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))
        elif transfer == 'no_output':
            for layer in model.layers:
                if not layer.name.startswith('yolo_output'):
                    layer.set_weights(model_pretrained.get_layer(
                        layer.name).get_weights())
                    freeze_all(layer)
    else:
        # All other transfer require matching classes
        model.load_weights(weights)
        if transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = [YoloLoss(anchors[mask], classes=num_classes) for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss, run_eagerly=(mode == 'eager_fit'))

    return model, optimizer, loss, anchors, anchor_masks


def _prep_training_data(
    data_file_glob: str,
    classes_file: str = None,
    image_size: int = 416,
    batch_size: int = 1,
    anchors: List = yolo_anchors,
    anchor_masks: List = yolo_anchor_masks
):
    train_dataset = dataset_module.load_tfrecord_dataset(data_file_glob, classes_file, image_size)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(
        lambda x, y: (
            dataset_module.transform_images(x, image_size),
            dataset_module.transform_targets(y, anchors, anchor_masks, image_size)
        )
    )
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_dataset


def _train_val_test_split(
    list_to_split: List[any],
    rng: np.random.Generator,
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
        item_split_sets = {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}
    elif num_val_items > 0 and num_test_items == 0:
        train, val = np.split(list_to_split, [num_train_items])
        item_split_sets = {"train": train.tolist(), "validation": val.tolist()}
    elif num_val_items == 0 and num_test_items > 0:
        train, test = np.split(list_to_split, [num_train_items])
        item_split_sets = {"train": train.tolist(), "test": test.tolist()}

    return item_split_sets


def detect(
    model: Model,
    image: str,
    size: int = 416,
):

    img_raw = tf.image.decode_image(open(image, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    # boxes, scores, classes, nums = model(img)
    boxes, scores, classes, nums = model.predict(img)
    return boxes, scores, classes, nums


def detect_summary(
    image: str,
    num_classes: int,
    weights: str,
    classes_file: str,
    size: int = 416,
    output: str = None,
    max_boxes: int = 100,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    soft_nms_sigma: float = 0.0,
    tiny: bool = False,
    verbose=True
):
    class_names = [c.strip() for c in open(classes_file).readlines()]
    if verbose:
        print('classes loaded')

    model = _load_model(
        num_classes=num_classes,
        weights=weights,
        max_boxes=max_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        soft_nms_sigma=soft_nms_sigma,
        tiny=tiny,
        verbose=verbose
    )

    boxes, scores, classes, nums = detect(
        model=model,
        image=image,
        size=size,
    )

    if verbose:
        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])], np.array(scores[0][i]), np.array(boxes[0][i])))

    img_raw = tf.image.decode_image(open(image, 'rb').read(), channels=3)
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output, img)
    if verbose:
        print('output saved to: {}'.format(output))


def detect_batch(
    image_files: List[str],
    output_dir: str,
    num_classes: int,
    weights: str,
    classes_file: str,
    size: int = 416,
    max_boxes: int = 100,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    soft_nms_sigma: float = 0.0,
    tiny: bool = False,
    verbose=True
):

    class_names = [c.strip() for c in open(classes_file).readlines()]
    if verbose:
        print('classes loaded')

    model = _load_model(
        num_classes=num_classes,
        weights=weights,
        max_boxes=max_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        soft_nms_sigma=soft_nms_sigma,
        tiny=tiny,
        verbose=verbose
    )

    stats = []

    for image_file in tqdm(image_files):
        start_time = perf_counter()

        # if verbose:
        #     print(f'Detecting: {image_file}')
        basename = os.path.basename(image_file)
        anno_image_file = os.path.join(output_dir, basename)

        detect_start_time = perf_counter()
        boxes, scores, classes, nums = detect(
            model=model,
            image=image_file,
            size=size,
        )
        detect_end_time = perf_counter()

        img_raw = tf.image.decode_image(open(image_file, 'rb').read(), channels=3)
        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names, scale_font=True)
        cv2.imwrite(anno_image_file, img)

        end_time = perf_counter()
        stats.append(
            [
                end_time - start_time,                                            # Total time
                detect_end_time - detect_start_time,                              # Detection time
                (end_time - start_time) - (detect_end_time - detect_start_time)   # Non-detect time
            ]
        )

    if verbose:
        stats_np = np.array(stats)
        print(f'     Total time: min {round(stats_np[:, 0].min(), 2)}, avg {round(stats_np[:, 0].mean(), 2)}, max {round(stats_np[:, 0].max(), 2)}')
        print(f'    Detect time: min {round(stats_np[:, 1].min(), 2)}, avg {round(stats_np[:, 1].mean(), 2)}, max {round(stats_np[:, 1].max(), 2)}')
        print(f'Non-detect time: min {round(stats_np[:, 2].min(), 2)}, avg {round(stats_np[:, 2].mean(), 2)}, max {round(stats_np[:, 2].max(), 2)}')


def train(
    dataset_glob: str,
    val_dataset_glob: str,
    classes_path: str,
    num_classes: int,
    epochs: int,
    image_size: int = 416,
    batch_size: int = 1,
    train_mode: str = 'fit',
    learning_rate: float = 0.001,
    transfer: str = 'darknet',
    weights: str = None,
    weights_num_classes: int = None,
    logs_path: str = 'logs',
    multi_gpu: bool = False,
    tiny: bool = False,
    verbose: bool = True,
    use_tensorboard: bool = False,
    checkpoints_path: str = None,
    checkpoints_prefix: str = None,
    checkpoints_monitor: str = 'val_loss',
    checkpoints_initial_threshold=None,
    early_stopping_monitor: str = 'val_loss',
    early_stopping_min_delta: float = 0,
    early_stopping_patience: int = 10,
    early_stopping_baseline: float = None,
    reduce_lr_monitor: str = 'val_loss',
    reduce_lr_factor: float = 0.1,
    reduce_lr_patience: int = 10,
    reduce_lr_min_delta: float = 0.0001,
    reduce_lr_cooldown: int = 0,
    reduce_lr_min_lr: float = 0.0,
    input_batch_limit: int = None,
    wandb_project_name: str = None,
    wandb_run_name: str = None,
    wandb_notes: str = None
) -> Tuple[Model, History]:
    # Print out the arguments supplied
    args = locals()
    for k, v in args.items():
        print(f"  {k}: {v}")

    if wandb_project_name:
        wandb.init(
            project=wandb_project_name, name=wandb_run_name, notes=wandb_notes,
            sync_tensorboard=use_tensorboard
        )
        wandb.config.update(args)

    # Setup the model
    if multi_gpu:
        for physical_device in tf.config.experimental.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(physical_device, True)
            except RuntimeError:
                pass    # This will happen if the GPU device has already been set

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        batch_size = batch_size * strategy.num_replicas_in_sync

        with strategy.scope():
            model, optimizer, loss, anchors, anchor_masks = _setup_model(
                training=True, weights=weights, weights_num_classes=weights_num_classes,
                num_classes=num_classes, size=image_size,
                mode=train_mode, learning_rate=learning_rate, transfer=transfer,
                tiny=tiny
            )
    else:
        model, optimizer, loss, anchors, anchor_masks = _setup_model(
            training=True, weights=weights, weights_num_classes=weights_num_classes,
            num_classes=num_classes, size=image_size,
            mode=train_mode, learning_rate=learning_rate, transfer=transfer,
            tiny=tiny
        )

    # Build the training data pipeline
    train_dataset = _prep_training_data(
        data_file_glob=dataset_glob,
        classes_file=classes_path,
        image_size=image_size,
        anchors=anchors,
        anchor_masks=anchor_masks,
        batch_size=batch_size
    )
    if input_batch_limit:
        train_dataset = train_dataset.take(input_batch_limit)

    # Optionally, build the validation data pipeline
    if val_dataset_glob:
        val_dataset = _prep_training_data(
            data_file_glob=val_dataset_glob,
            classes_file=classes_path,
            image_size=image_size,
            anchors=anchors,
            anchor_masks=anchor_masks,
            batch_size=batch_size
        )
        if input_batch_limit:
            val_dataset = val_dataset.take(input_batch_limit)
    else:
        val_dataset = None

    # Run the training
    # if train_mode == 'eager_tf':
    #     # Eager mode is great for debugging
    #     # Non eager graph mode is recommended for real training
    #     avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    #     avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

    #     for epoch in range(1, epochs + 1):
    #         for batch, (images, labels) in enumerate(train_dataset):
    #             with tf.GradientTape() as tape:
    #                 outputs = model(images, training=True)
    #                 regularization_loss = tf.reduce_sum(model.losses)
    #                 pred_loss = []
    #                 for output, label, loss_fn in zip(outputs, labels, loss):
    #                     pred_loss.append(loss_fn(label, output))
    #                 total_loss = tf.reduce_sum(pred_loss) + regularization_loss

    #             grads = tape.gradient(total_loss, model.trainable_variables)
    #             optimizer.apply_gradients(
    #                 zip(grads, model.trainable_variables))

    #             if verbose:
    #                 print("{}_train_{}, {}, {}".format(
    #                     epoch, batch, total_loss.numpy(),
    #                     list(map(lambda x: np.sum(x.numpy()), pred_loss)))
    #                 )
    #             avg_loss.update_state(total_loss)

    #         for batch, (images, labels) in enumerate(val_dataset):
    #             outputs = model(images)
    #             regularization_loss = tf.reduce_sum(model.losses)
    #             pred_loss = []
    #             for output, label, loss_fn in zip(outputs, labels, loss):
    #                 pred_loss.append(loss_fn(label, output))
    #             total_loss = tf.reduce_sum(pred_loss) + regularization_loss

    #             if verbose:
    #                 print("{}_val_{}, {}, {}".format(
    #                     epoch, batch, total_loss.numpy(),
    #                     list(map(lambda x: np.sum(x.numpy()), pred_loss)))
    #                 )
    #             avg_val_loss.update_state(total_loss)

    #         if verbose:
    #             print("{}, train: {}, val: {}".format(
    #                 epoch,
    #                 avg_loss.result().numpy(),
    #                 avg_val_loss.result().numpy())
    #             )

    #         avg_loss.reset_states()
    #         avg_val_loss.reset_states()
    #         model.save_weights(
    #             os.path.join(checkpoints_path, f'yolov3_train_{epoch}.tf')
    #         )
    # else:
    callbacks = [
        ReduceLROnPlateau(
            verbose=1,
            monitor=reduce_lr_monitor,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_delta=reduce_lr_min_delta,
            cooldown=reduce_lr_cooldown,
            min_lr=reduce_lr_min_lr
        ),
        EarlyStopping(
            verbose=1,
            monitor=early_stopping_monitor,
            min_delta=early_stopping_min_delta,
            patience=early_stopping_patience,
            baseline=early_stopping_baseline
        ),
        ModelCheckpoint(
            monitor=checkpoints_monitor,
            filepath=os.path.join(checkpoints_path, (checkpoints_prefix + '_{epoch}.tf')),  # this is not an f-string
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            initial_value_threshold=checkpoints_initial_threshold
        )
    ]
    if use_tensorboard:
        callbacks.append(TensorBoard(log_dir=logs_path))
    if wandb_project_name:
        callbacks.append(WandbCallback(save_model=True))

    start_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_dataset
    )
    end_time = time.time() - start_time

    if verbose:
        print(f'Total Training Time: {end_time}')

    # if wandb_project_name:
    #     wandb.finish()

    return model, history


def evaluate(
    test_dataset_glob: str,
    num_classes: int,
    weights_file: str,
    weights_num_classes: int = None,
    classes_file: str = None,
    anchors: List = yolo_anchors,
    anchor_masks: List = yolo_anchor_masks,
    batch_size: int = 16,
    input_batch_limit: int = None,
    image_size: int = 416,
    train_mode: str = 'fit',
    learning_rate: float = 0.001,
    transfer: str = 'darknet',
    tiny: bool = False,
    print_result: bool = False
):

    test_dataset = _prep_training_data(
        data_file_glob=test_dataset_glob,
        classes_file=classes_file,
        image_size=image_size,
        anchors=anchors,
        anchor_masks=anchor_masks,
        batch_size=batch_size
    )
    if input_batch_limit:
        test_dataset = test_dataset.take(input_batch_limit)

    model, optimizer, loss, anchors, anchor_masks = _setup_model(
        training=True, weights=weights_file, weights_num_classes=weights_num_classes,
        num_classes=num_classes, size=image_size,
        mode=train_mode, learning_rate=learning_rate, transfer=transfer,
        tiny=tiny
    )

    result = model.evaluate(test_dataset)
    if print_result:
        print(dict(zip(model.metrics_names, result)))

    return result


# def orig_augment_images(
#     train_img_glob: str,
#     tf_output_dir: str,
#     temp_img_dir: str = '/tmp/images',
#     tf_output_prefix: str = '',
#     num_rand_augs: int = 4,
#     random_seed: int = 42,
#     image_size: int = 416,
#     min_file_mb_size: int = 30
# ):
#     rng = np.random.default_rng(random_seed)

#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # Read original training images
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     orig_img_files = glob(train_img_glob)
#     print(f"{len(orig_img_files)} images found")

#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # Augment images & write augmented images to temp directory
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     os.makedirs(temp_img_dir, exist_ok=True)

#     for img_file_name in tqdm(orig_img_files, desc='Augmenting images'):
#         orig_scaled_img, orig_bboxes = read_image_with_yolo_annotations(img_file_name, resize=(image_size, image_size))
#         # print(f"orig_bboxes: {orig_bboxes}")

#         # Write the resized original
#         write_augmented_image_files(
#             new_img=orig_scaled_img, bboxes=orig_bboxes, orig_filename=img_file_name,
#             augmentation_name='orig', output_dir=temp_img_dir
#         )

#         # Horizontal flip - Only once since it's always the same transformation
#         new_img, new_bboxes = aug_horizontal_flip(orig_scaled_img, orig_bboxes)
#         write_augmented_image_files(
#             new_img=new_img, bboxes=new_bboxes, orig_filename=img_file_name,
#             augmentation_name='hflip', output_dir=temp_img_dir
#         )

#         # Random scale
#         for i in range(1, num_rand_augs + 1):
#             new_img, new_bboxes = aug_scale_random(orig_scaled_img, orig_bboxes)
#             write_augmented_image_files(
#                 new_img=new_img, bboxes=new_bboxes, orig_filename=img_file_name,
#                 augmentation_name=f'scale{i}', output_dir=temp_img_dir
#             )

#         # Random translate
#         for i in range(1, num_rand_augs + 1):
#             new_img, new_bboxes = aug_translate_random(orig_scaled_img, orig_bboxes)
#             write_augmented_image_files(
#                 new_img=new_img, bboxes=new_bboxes, orig_filename=img_file_name,
#                 augmentation_name=f'translate{i}', output_dir=temp_img_dir
#             )

#         # Random rotate
#         for i in range(1, num_rand_augs + 1):
#             new_img, new_bboxes = aug_rotate_random(orig_scaled_img, orig_bboxes)
#             write_augmented_image_files(
#                 new_img=new_img, bboxes=new_bboxes, orig_filename=img_file_name,
#                 augmentation_name=f'rotate{i}', output_dir=temp_img_dir
#             )

#         # Random shear
#         for i in range(1, num_rand_augs + 1):
#             new_img, new_bboxes = aug_shear_random(orig_scaled_img, orig_bboxes)
#             write_augmented_image_files(
#                 new_img=new_img, bboxes=new_bboxes, orig_filename=img_file_name,
#                 augmentation_name=f'shear{i}', output_dir=temp_img_dir
#             )

#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # Split into train, validation
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     TEMP_DIR_GLOB = os.path.join(temp_img_dir, 'IMG_*.png')

#     print('Splitting augmented images into train/validation sets...')
#     img_files = glob(TEMP_DIR_GLOB)
#     splits = _train_val_test_split(img_files, val_fraction=0.1, test_fraction=0., shuffle=True, rng=rng)

#     for split_name, split_items in splits.items():
#         print(f"  {split_name}: {len(split_items)} images")

#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # Convert for tfrecords and write tfrecord files
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     print()
#     print('Converting images to tfrecord files...')
#     os.makedirs(tf_output_dir, exist_ok=True)

#     for split_name, split_items in splits.items():

#         write_training_tfrecords(
#             items=split_items, type_name=split_name,
#             output_dir=tf_output_dir, file_prefix=tf_output_prefix,
#             min_file_mb=min_file_mb_size
#         )

#     print("Process complete.")


def augment_and_prep_train_images(
    train_img_glob: str,
    transform_groups: Tuple[A.core.composition.Compose, int],
    tf_output_dir: str,
    temp_img_dir: str = '/tmp/images',
    tf_output_prefix: str = '',
    random_seed: int = 42,
    image_size: int = 416,
    keep_original: bool = True,
    min_file_mb_size: int = 30,
    num_threads: int = 1
):

    rng = np.random.default_rng(random_seed)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read original training images
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    orig_img_files = glob(train_img_glob)
    assert len(orig_img_files) > 0, f"No training image found at {train_img_glob}"
    print(f"{len(orig_img_files)} images found")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Augment images & write augmented images to temp directory
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    os.makedirs(temp_img_dir, exist_ok=True)

    def _transform_image(img_file_name: str):
        orig_scaled_img, orig_bboxes = read_image_with_yolo_annotations(
            img_file_name,
            resize=(image_size, image_size),
            yolo_boxes=True,
            bgr2rgb=True
        )

        # Write the resized original
        if keep_original:
            write_augmented_image_files(
                new_img=orig_scaled_img, bboxes=orig_bboxes, orig_filename=img_file_name,
                augmentation_name='orig', output_dir=temp_img_dir
            )

        # For each transform group
        transform: A.core.composition.Compose
        for g, t_group in enumerate(transform_groups):
            transform, num_imgs = t_group

            # Transform the number of times requested
            for i in range(num_imgs):
                transformed = transform(image=orig_scaled_img, bboxes=orig_bboxes.tolist())

                write_augmented_image_files(
                    new_img=transformed['image'], bboxes=np.array(transformed['bboxes']),
                    orig_filename=img_file_name, augmentation_name=f'g{g}-i{i}',
                    output_dir=temp_img_dir
                )

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(_transform_image, orig_img_files), total=len(orig_img_files), desc="Augmenting Images"))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Split into train, validation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TEMP_DIR_GLOB = os.path.join(temp_img_dir, 'IMG_*.png')

    print('Splitting augmented images into train/validation sets...')
    img_files = glob(TEMP_DIR_GLOB)
    splits = _train_val_test_split(img_files,  rng=rng, val_fraction=0.1, test_fraction=0., shuffle=True)

    for split_name, split_items in splits.items():
        print(f"  {split_name}: {len(split_items)} images")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert for tfrecords and write tfrecord files
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print()
    print('Converting images to tfrecord files...')
    os.makedirs(tf_output_dir, exist_ok=True)

    for split_name, split_items in splits.items():

        write_training_tfrecords(
            items=split_items, type_name=split_name,
            output_dir=tf_output_dir, file_prefix=tf_output_prefix,
            min_file_mb=min_file_mb_size
        )

    print("Process complete.")


def clean_images(img_file_glob: str):
    for img_file_name in tqdm(img_file_glob, desc="Cleaning"):
        image, bboxes = read_image_with_yolo_annotations(img_file_name)
        cv2.imwrite(img_file_name, image)


def augment_3(
    train_val_json: str,
    train_images_dir: str,
    tf_output_dir: str,
    temp_img_dir: str = '/tmp/images',
    tf_output_prefix: str = '',
    image_size: int = 416,
    keep_original: bool = True,
    min_file_mb_size: int = 30,
    num_threads: int = 1,
    max_split_files: int = None
):

    # Get list of train and validation images
    splits: dict
    with open(train_val_json, 'r') as f:
        splits = json.load(f)

    train_image_files = [os.path.join(train_images_dir, f) for f in splits['train']]
    val_image_files = [os.path.join(train_images_dir, f) for f in splits['validation']]

    if max_split_files:
        train_image_files = train_image_files[:max_split_files]
        val_image_files = val_image_files[:max_split_files]

    assert len(train_image_files) > 0, f"No training images found in {train_val_json}"
    assert len(val_image_files) > 0, f"No validation images found in {train_val_json}"
    print(f"{len(train_image_files)}/{len(val_image_files)} train/val images found")

    os.makedirs(temp_img_dir, exist_ok=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Declare transform operation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _transform_image(output_dir: str, transform_groups: list, img_file_name: str):

        orig_scaled_img, orig_bboxes = read_image_with_yolo_annotations(
            img_file_name,
            resize=(image_size, image_size),
            yolo_boxes=True,
            bgr2rgb=True
        )

        # Write the resized original
        if keep_original:
            write_augmented_image_files(
                new_img=orig_scaled_img, bboxes=orig_bboxes, orig_filename=img_file_name,
                augmentation_name='orig', output_dir=output_dir
            )

        # For each transform group
        transform: A.core.composition.Compose
        for g, t_group in enumerate(transform_groups):
            transform, num_imgs = t_group

            # Transform the number of times requested
            for i in range(num_imgs):
                transformed = transform(image=orig_scaled_img, bboxes=orig_bboxes.tolist())

                write_augmented_image_files(
                    new_img=transformed['image'], bboxes=np.array(transformed['bboxes']),
                    orig_filename=img_file_name, augmentation_name=f'g{g}-i{i}',
                    output_dir=output_dir
                )

    bbox_params = A.BboxParams(format='yolo', min_area=32, min_visibility=0.3)

    t_hflip = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
        ],
        bbox_params=bbox_params
    )

    t_rotate_w_noise = A.Compose(
        [
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=5, p=0.1),
        ],
        bbox_params=bbox_params
    )

    t_safe_crop_per = A.Compose(
        [
            A.RandomSizedBBoxSafeCrop(image_size, image_size, p=1.0),
            A.Perspective(scale=(0.05, 0.2), pad_mode=1, p=1.0),
            A.ToGray(p=0.2),
        ],
        bbox_params=bbox_params
    )

    t_reg_crop = A.Compose(
        [
            A.RandomResizedCrop(image_size, image_size, p=1.0),
            A.Perspective(scale=(0.05, 0.2), pad_mode=1, p=1.0),
        ],
        bbox_params=bbox_params
    )

    t_complex = A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Perspective(scale=(0.05, 0.3), pad_mode=1, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.5, rotate_limit=60, border_mode=1, p=0.75),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=5, p=0.2),
            A.ChannelDropout(p=0.2),
        ],
        bbox_params=bbox_params
    )

    # Prep output dirs
    train_output_dir = os.path.join(temp_img_dir, 'train')
    val_output_dir = os.path.join(temp_img_dir, 'validation')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Augment training images
    # ~~~~~~~~~~~~~~~~~~~~~~~

    transform_groups = [
        (t_hflip, 1),
        (t_rotate_w_noise, 3),
        (t_safe_crop_per, 4),
        (t_reg_crop, 8),
        (t_complex, 30)
    ]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(
            executor.map(partial(_transform_image, train_output_dir, transform_groups), train_image_files),
            total=len(train_image_files),
            desc="Augmenting training images"
        ))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # Augment validation images
    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    transform_groups = [
        (t_hflip, 1),
        (t_rotate_w_noise, 3)
    ]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(
            executor.map(partial(_transform_image, val_output_dir, transform_groups), val_image_files),
            total=len(val_image_files),
            desc="Augmenting validation images"
        ))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert for tfrecords and write tfrecord files
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print()
    print('Converting images to tfrecord files...')

    os.makedirs(tf_output_dir, exist_ok=True)

    # Assumes augmented files are PNG image files
    aug_train_images = glob(os.path.join(temp_img_dir, 'train', '*.png'))
    write_training_tfrecords(
        items=aug_train_images, type_name='train',
        output_dir=tf_output_dir, file_prefix=tf_output_prefix,
        min_file_mb=min_file_mb_size
    )

    aug_val_images = glob(os.path.join(temp_img_dir, 'validation', '*.png'))
    write_training_tfrecords(
        items=aug_val_images, type_name='validation',
        output_dir=tf_output_dir, file_prefix=tf_output_prefix,
        min_file_mb=min_file_mb_size
    )

    print("Process complete.")
