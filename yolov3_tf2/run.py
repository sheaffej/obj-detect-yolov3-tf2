import numpy as np
import os
from tqdm import tqdm
from time import perf_counter
import time
from typing import Tuple, List

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
from yolov3_tf2.dataset import transform_images  # , load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs, freeze_all
import yolov3_tf2.dataset as dataset_module


def load_model(
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

    model = load_model(
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

    model = load_model(
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


def setup_model(
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


def prep_training_data(
    data_file_glob: str,
    classes_file: str = None,
    image_size: int = 416,
    batch_size: int = 1,
    anchors: List = yolo_anchors,
    anchor_masks: List = yolo_anchor_masks
):
    train_dataset = dataset_module.load_tfrecord_dataset(data_file_glob, classes_file, image_size)
    # train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(
        lambda x, y: (
            dataset_module.transform_images(x, image_size),
            dataset_module.transform_targets(y, anchors, anchor_masks, image_size)
        )
    )
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_dataset


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
    logs_path: str = None,
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
        wandb.init(project=wandb_project_name, name=wandb_run_name, notes=wandb_notes)
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
            model, optimizer, loss, anchors, anchor_masks = setup_model(
                training=True, weights=weights, weights_num_classes=weights_num_classes,
                num_classes=num_classes, size=image_size,
                mode=train_mode, learning_rate=learning_rate, transfer=transfer,
                tiny=tiny
            )
    else:
        model, optimizer, loss, anchors, anchor_masks = setup_model(
            training=True, weights=weights, weights_num_classes=weights_num_classes,
            num_classes=num_classes, size=image_size,
            mode=train_mode, learning_rate=learning_rate, transfer=transfer,
            tiny=tiny
        )

    # Build the training data pipeline
    train_dataset = prep_training_data(
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
        val_dataset = prep_training_data(
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

    test_dataset = prep_training_data(
        data_file_glob=test_dataset_glob,
        classes_file=classes_file,
        image_size=image_size,
        anchors=anchors,
        anchor_masks=anchor_masks,
        batch_size=batch_size
    )
    if input_batch_limit:
        test_dataset = test_dataset.take(input_batch_limit)

    model, optimizer, loss, anchors, anchor_masks = setup_model(
        training=True, weights=weights_file, weights_num_classes=weights_num_classes,
        num_classes=num_classes, size=image_size,
        mode=train_mode, learning_rate=learning_rate, transfer=transfer,
        tiny=tiny
    )

    result = model.evaluate(test_dataset)
    if print_result:
        print(dict(zip(model.metrics_names, result)))

    return result
