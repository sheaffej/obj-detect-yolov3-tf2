# Yolo V3 Implemented in TensorFlow 2.0

This is my fork for my own purposes. 

Please see the parent of this fork at: https://github.com/zzh8829/yolov3-tf2

Primarily, the changes I'm making are for integration into how I will use the implementataion for training and real-time prediction as part of another project.

This adaptation carries on the MIT license from the parent repository, and therefore everyone is welcome to use my adaptation of Zihao Zhang's exellent work.

---

### Notes

- I develop using VSCode and Docker, and I'm leaving my `.devcontainer` and `.vscode` folders in case they are useful for others. I try to keep them as re-usable as possible, however you will very likely have to adjust them to re-use them on your system. For example, I mount a `DataSets` directory in `.devcontainer.json` which will cause the container to fail unless you have the exact same directory present on your system.

---

### Change Log

*21 Aug 2022*  
Moved the logic in non-package `train.py` and `detect.py` modules that were designed to be run as commmand-line applications into a `run.py` package module so that it is more natural to import and invoke the model operations from other Python code.

Removed Abseil functionality to facilitate the above work.

Added `checkpoints_path` and `logs_path` parameters to the `train` function to allow configuration of those paths.

Verified code changes works on the Colab free-tier GPU runtime.