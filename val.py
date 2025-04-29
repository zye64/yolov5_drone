# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlpackage          # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import csv
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    """
    Saves one detection result to a txt file in normalized xywh format, optionally including confidence.

    Args:
        predn (torch.Tensor): Predicted bounding boxes and associated confidence scores and classes in xyxy format, tensor
            of shape (N, 6) where N is the number of detections.
        save_conf (bool): If True, saves the confidence scores along with the bounding box coordinates.
        shape (tuple): Shape of the original image as (height, width).
        file (str | Path): File path where the result will be saved.

    Returns:
        None

    Notes:
        The xyxy bounding box format represents the coordinates (xmin, ymin, xmax, ymax).
        The xywh format represents the coordinates (center_x, center_y, width, height) and is normalized by the width and
        height of the image.

    Example:
        ```python
        predn = torch.tensor([[10, 20, 30, 40, 0.9, 1]])  # example prediction
        save_one_txt(predn, save_conf=True, shape=(640, 480), file="output.txt")
        ```
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    # 检查是否有额外维度 (theta, phi)
    has_extra_dims = predn.shape[1] > 6
    
    # 修改解包逻辑以避免多个星号表达式错误
    for p in predn.tolist():
        xyxy = p[:4]
        conf = p[4]
        cls = p[5]
        extra_dims = p[6:]
        
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        
        # 基础行格式
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        
        # 如果有额外维度(theta, phi)，添加到行末尾
        if has_extra_dims and len(extra_dims) >= 2:
            theta, phi = extra_dims[0], extra_dims[1]
            line = (*line, theta, phi)
        
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    """
    Saves a single JSON detection result, including image ID, category ID, bounding box, and confidence score.

    Args:
        predn (torch.Tensor): Predicted detections in xyxy format with shape (n, 6) where n is the number of detections.
                              The tensor should contain [x_min, y_min, x_max, y_max, confidence, class_id] for each detection.
        jdict (list[dict]): List to collect JSON formatted detection results.
        path (pathlib.Path): Path object of the image file, used to extract image_id.
        class_map (dict[int, int]): Mapping from model class indices to dataset-specific category IDs.

    Returns:
        None: Appends detection results as dictionaries to `jdict` list in-place.

    Example:
        ```python
        predn = torch.tensor([[100, 50, 200, 150, 0.9, 0], [50, 30, 100, 80, 0.8, 1]])
        jdict = []
        path = Path("42.jpg")
        class_map = {0: 18, 1: 19}
        save_one_json(predn, jdict, path, class_map)
        ```
        This will append to `jdict`:
        ```
        [
            {'image_id': 42, 'category_id': 18, 'bbox': [125.0, 75.0, 100.0, 100.0], 'score': 0.9},
            {'image_id': 42, 'category_id': 19, 'bbox': [75.0, 55.0, 50.0, 50.0], 'score': 0.8}
        ]
        ```

    Notes:
        The `bbox` values are formatted as [x, y, width, height], where x and y represent the top-left corner of the box.
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    
    # 检查是否有额外维度 (theta, phi)
    has_extra_dims = predn.shape[1] > 6
    
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        result = {
            "image_id": image_id,
            "category_id": class_map[int(p[5])],
            "bbox": [round(x, 3) for x in b],
            "score": round(p[4], 5),
        }
        
        # 如果有额外维度，添加到JSON中
        if has_extra_dims and len(p) > 6:
            # 取出theta和phi值
            theta_phi = p[6:8]
            result["theta"] = round(theta_phi[0], 5)
            result["phi"] = round(theta_phi[1], 5)
            
        jdict.append(result)


def process_batch(detections, labels, iouv):
    """
    Return a correct prediction matrix given detections and labels at various IoU thresholds.

    Args:
        detections (np.ndarray): Array of shape (N, 6) where each row corresponds to a detection with format
            [x1, y1, x2, y2, conf, class].
        labels (np.ndarray): Array of shape (M, 5) where each row corresponds to a ground truth label with format
            [class, x1, y1, x2, y2].
        iouv (np.ndarray): Array of IoU thresholds to evaluate at.

    Returns:
        correct (np.ndarray): A binary array of shape (N, len(iouv)) indicating whether each detection is a true positive
            for each IoU threshold. There are 10 IoU levels used in the evaluation.

    Example:
        ```python
        detections = np.array([[50, 50, 200, 200, 0.9, 1], [30, 30, 150, 150, 0.7, 0]])
        labels = np.array([[1, 50, 50, 200, 200]])
        iouv = np.linspace(0.5, 0.95, 10)
        correct = process_batch(detections, labels, iouv)
        ```

    Notes:
        - This function is used as part of the evaluation pipeline for object detection models.
        - IoU (Intersection over Union) is a common evaluation metric for object detection performance.
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def plot_angle_errors(save_dir):
    """Plots histograms and scatter plots from angle_errors.csv."""
    csv_path = Path(save_dir) / 'angle_errors.csv'
    if not csv_path.exists():
        LOGGER.warning(f"CSV file not found at {csv_path}, skipping angle error plotting.")
        return

    try:
        data = pd.read_csv(csv_path)
        if data.empty:
            LOGGER.warning(f"CSV file {csv_path} is empty, skipping angle error plotting.")
            return

        LOGGER.info(f"Plotting angle errors from {csv_path}...")

        # Ensure error columns exist
        if 'pitch_error_deg' not in data.columns or 'azimuth_error_deg' not in data.columns:
             LOGGER.warning(f"CSV file {csv_path} missing required error columns, skipping plotting.")
             return

        # 1. Error Histograms
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(data['pitch_error_deg'], bins=50)
        plt.title('Pitch Error Distribution (degrees)')
        plt.xlabel('Error (Predicted - Ground Truth)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.5)

        plt.subplot(1, 2, 2)
        plt.hist(data['azimuth_error_deg'], bins=50)
        plt.title('Azimuth Error Distribution (degrees)')
        plt.xlabel('Error (Predicted - Ground Truth)')
        plt.grid(True, alpha=0.5)

        hist_save_path = Path(save_dir) / 'angle_error_histograms.png'
        plt.tight_layout()
        plt.savefig(hist_save_path, dpi=200)
        plt.close()
        LOGGER.info(f"Saved error histograms to {hist_save_path}")

        # 2. Predicted vs True Scatter Plots (Raw normalized values)
        if 'pred_pitch_raw' in data.columns and 'gt_pitch_raw' in data.columns and \
           'pred_azimuth_raw' in data.columns and 'gt_azimuth_raw' in data.columns:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(data['gt_pitch_raw'], data['pred_pitch_raw'], alpha=0.5, s=5)
            plt.plot([0, 1], [0, 1], 'r--', label='Ideal') # Add identity line
            plt.title('Predicted vs True Pitch (Normalized)')
            plt.xlabel('True Pitch (Normalized)')
            plt.ylabel('Predicted Pitch (Normalized)')
            plt.grid(True)
            plt.legend()
            plt.axis('square') # Make axes equal
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            plt.subplot(1, 2, 2)
            plt.scatter(data['gt_azimuth_raw'], data['pred_azimuth_raw'], alpha=0.5, s=5)
            plt.plot([0, 1], [0, 1], 'r--', label='Ideal') # Add identity line
            plt.title('Predicted vs True Azimuth (Normalized)')
            plt.xlabel('True Azimuth (Normalized)')
            plt.ylabel('Predicted Azimuth (Normalized)')
            plt.grid(True)
            plt.legend()
            plt.axis('square') # Make axes equal
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            scatter_save_path = Path(save_dir) / 'angle_pred_vs_true_scatter.png'
            plt.tight_layout()
            plt.savefig(scatter_save_path, dpi=200)
            plt.close()
            LOGGER.info(f"Saved prediction vs true scatter plots to {scatter_save_path}")

    except Exception as e:
        LOGGER.error(f"Failed to plot angle errors: {e}")


@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
):
    """
    Evaluates a YOLOv5 model on a dataset and logs performance metrics.

    Args:
        data (str | dict): Path to a dataset YAML file or a dataset dictionary.
        weights (str | list[str], optional): Path to the model weights file(s). Supports various formats including PyTorch,
            TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TensorFlow SavedModel, TensorFlow GraphDef, TensorFlow Lite,
            TensorFlow Edge TPU, and PaddlePaddle.
        batch_size (int, optional): Batch size for inference. Default is 32.
        imgsz (int, optional): Input image size (pixels). Default is 640.
        conf_thres (float, optional): Confidence threshold for object detection. Default is 0.001.
        iou_thres (float, optional): IoU threshold for Non-Maximum Suppression (NMS). Default is 0.6.
        max_det (int, optional): Maximum number of detections per image. Default is 300.
        task (str, optional): Task type - 'train', 'val', 'test', 'speed', or 'study'. Default is 'val'.
        device (str, optional): Device to use for computation, e.g., '0' or '0,1,2,3' for CUDA or 'cpu' for CPU. Default is ''.
        workers (int, optional): Number of dataloader workers. Default is 8.
        single_cls (bool, optional): Treat dataset as a single class. Default is False.
        augment (bool, optional): Enable augmented inference. Default is False.
        verbose (bool, optional): Enable verbose output. Default is False.
        save_txt (bool, optional): Save results to *.txt files. Default is False.
        save_hybrid (bool, optional): Save label and prediction hybrid results to *.txt files. Default is False.
        save_conf (bool, optional): Save confidences in --save-txt labels. Default is False.
        save_json (bool, optional): Save a COCO-JSON results file. Default is False.
        project (str | Path, optional): Directory to save results. Default is ROOT/'runs/val'.
        name (str, optional): Name of the run. Default is 'exp'.
        exist_ok (bool, optional): Overwrite existing project/name without incrementing. Default is False.
        half (bool, optional): Use FP16 half-precision inference. Default is True.
        dnn (bool, optional): Use OpenCV DNN for ONNX inference. Default is False.
        model (torch.nn.Module, optional): Model object for training. Default is None.
        dataloader (torch.utils.data.DataLoader, optional): Dataloader object. Default is None.
        save_dir (Path, optional): Directory to save results. Default is Path('').
        plots (bool, optional): Plot validation images and metrics. Default is True.
        callbacks (utils.callbacks.Callbacks, optional): Callbacks for logging and monitoring. Default is Callbacks().
        compute_loss (function, optional): Loss function for training. Default is None.

    Returns:
        dict: Contains performance metrics including precision, recall, mAP50, and mAP50-95.
    """
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times
    loss = torch.zeros(4, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    # <<< Initialize list for storing detailed angle error data >>>
    angle_error_data = []
    # <<< End Initialization >>>

    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls, ext

        # NMS
        targets[:, 2:6] *= torch.tensor((width, height, width, height), device=device)  # to pixels, only scale x,y,w,h
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            # --- Determine num_extra_dims (Handles standalone val.py and calls from train.py) ---
            num_extra_dims = 0 # Default to 0
            final_layer = None
            model_description = f"type {type(model)}" # For logging

            try:
                # Case 1: Called standalone (model is DetectMultiBackend)
                if isinstance(model, DetectMultiBackend):
                    model_description = "DetectMultiBackend instance"
                    if hasattr(model, 'pt') and model.pt and hasattr(model, 'model'):
                        inner_model = model.model # This should be the DetectionModel instance
                        if hasattr(inner_model, 'model') and isinstance(inner_model.model, nn.Sequential) and len(inner_model.model) > 0:
                            final_layer = inner_model.model[-1]
                        else:
                            LOGGER.warning(f"WARNING ({model_description}): Could not access the expected internal sequential model structure (model.model).")
                    else:
                         LOGGER.warning(f"WARNING ({model_description}): Loaded model is not PyTorch or structure is unexpected.")

                # Case 2: Called from train.py (model is likely DetectionModel or similar nn.Module)
                elif isinstance(model, nn.Module):
                     model_description = f"nn.Module instance ({type(model).__name__})"
                     # Check if it has the expected structure directly (DetectionModel has model.model)
                     if hasattr(model, 'model') and isinstance(model.model, nn.Sequential) and len(model.model) > 0:
                         final_layer = model.model[-1]
                     # Check if it's *already* the sequential model (less common, but possible)
                     elif isinstance(model, nn.Sequential) and len(model) > 0:
                         final_layer = model[-1]
                     # Fallback: Maybe the num_extra_dims is directly on the model? (Unlikely for standard YOLOv5 structure)
                     elif hasattr(model, 'num_extra_dims'):
                         num_extra_dims = model.num_extra_dims
                         final_layer = None # Prevent further checks below if found here
                         LOGGER.info(f"Found num_extra_dims directly on model instance ({type(model).__name__}).")
                     else:
                         LOGGER.warning(f"WARNING ({model_description}): Could not access the expected internal sequential model structure (e.g., model.model).")
                else:
                    # Fallback for unexpected types
                    LOGGER.warning(f"WARNING: Unexpected model type ({type(model)}). Cannot determine num_extra_dims reliably.")


                # If we found a potential final layer, check for the attribute
                if final_layer is not None:
                    if hasattr(final_layer, 'num_extra_dims'):
                        num_extra_dims = final_layer.num_extra_dims
                    else:
                        LOGGER.warning(f"WARNING ({model_description}): Final layer ({type(final_layer).__name__}) does not have 'num_extra_dims' attribute.")

            except Exception as e:
                LOGGER.error(f"ERROR ({model_description}): An unexpected error occurred while trying to determine num_extra_dims: {e}.")

            # Log the final determined value before NMS
            if num_extra_dims == 0:
                 LOGGER.warning(f"Using num_extra_dims = 0 for Non-Max Suppression. Angle calculations will likely fail or be skipped.")
            else:
                #  LOGGER.info(f"Using num_extra_dims = {num_extra_dims} for Non-Max Suppression.")
                pass
            # --- End Determine num_extra_dims ---

            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det,
                num_extra_dims=num_extra_dims
            )

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:] # Get labels for current image, format: [cls, x, y, w, h, pitch, azimuth]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            # <<< Add check for extra dims in labels and predictions >>>
            # Prediction should have at least 8 columns: xyxy(4), conf(1), cls(1), pitch(1), azimuth(1)
            has_extra_dims_pred = num_extra_dims >= 2 and pred.shape[1] >= 8
            # Labels should have at least 8 columns: cls(1), xywh(4), pitch(1), azimuth(1)
            has_extra_dims_label = labels.shape[1] >= 7 # cls,x,y,w,h,pitch,azimuth (index 6 is pitch)
            can_calc_angle_error = has_extra_dims_pred and has_extra_dims_label
            # Temporarily disable warning spam if dims are missing, but keep the check
            # if not can_calc_angle_error and nl > 0 and npr > 0:
            #      LOGGER.warning(f"Skipping angle error calculation for {path.name}: Predictions ({pred.shape[1]} cols) or labels ({labels.shape[1]} cols) lack required dimensions.")

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes (x,y,w,h are indices 1,2,3,4)
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels with class
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)

                # <<< Angle Error Calculation Start >>>
                if can_calc_angle_error:
                    iou = box_iou(labelsn[:, 1:], predn[:, :4]) # IoU between native-space labels and preds
                    # Create class mask for matching predictions and labels
                    correct_class_mask = labels[:, 0:1] == pred[:, 5] # Compare GT class (idx 0) with Pred class (idx 5)
                    iou_threshold = 0.5 # Minimum IoU for a match
                    matched_preds_indices = set()
                    
                    # Iterate through each ground truth label
                    for l_idx in range(nl):
                        # Find predictions of the same class with IoU >= threshold
                        matches = torch.where((iou[l_idx, :] >= iou_threshold) & (pred[:, 5] == labels[l_idx, 0]))[0]
                        
                        if len(matches) > 0:
                            # If multiple preds match, pick the one with highest IoU
                            best_match_local_idx = torch.argmax(iou[l_idx, matches])
                            p_idx = matches[best_match_local_idx].item() # Get the index in the original pred tensor
                            
                            # Ensure this prediction hasn't been matched to another label
                            if p_idx not in matched_preds_indices:
                                matched_preds_indices.add(p_idx)
                                
                                # Extract angles (pitch at index 6, azimuth at index 7)
                                p_pitch_raw = pred[p_idx, 6].item()
                                p_azimuth_raw = pred[p_idx, 7].item()
                                l_pitch_raw = labels[l_idx, 5].item() # Pitch is at index 5 in labels (cls,x,y,w,h,pitch,azi)
                                l_azimuth_raw = labels[l_idx, 6].item()# Azimuth is at index 6 in labels

                                # Calculate and store raw absolute error
                                pitch_err_raw = p_pitch_raw - l_pitch_raw
                                azimuth_err_raw = p_azimuth_raw - l_azimuth_raw
                                pitch_err_deg = (p_pitch_raw * 180) - (l_pitch_raw * 180)
                                azimuth_err_deg = (p_azimuth_raw * 360) - (l_azimuth_raw * 360)

                                # Create match data dictionary
                                match_data = {
                                    "image_path": path.name,
                                    "pred_box_x1": predn[p_idx, 0].item(), "pred_box_y1": predn[p_idx, 1].item(),
                                    "pred_box_x2": predn[p_idx, 2].item(), "pred_box_y2": predn[p_idx, 3].item(),
                                    "pred_conf": pred[p_idx, 4].item(), "pred_cls": int(pred[p_idx, 5].item()),
                                    "pred_pitch_raw": p_pitch_raw, "pred_azimuth_raw": p_azimuth_raw,
                                    "gt_box_x1": labelsn[l_idx, 1].item(), "gt_box_y1": labelsn[l_idx, 2].item(),
                                    "gt_box_x2": labelsn[l_idx, 3].item(), "gt_box_y2": labelsn[l_idx, 4].item(),
                                    "gt_cls": int(labels[l_idx, 0].item()),
                                    "gt_pitch_raw": l_pitch_raw, "gt_azimuth_raw": l_azimuth_raw,
                                    "iou": iou[l_idx, p_idx].item(),
                                    "pitch_error_raw": pitch_err_raw, "azimuth_error_raw": azimuth_err_raw,
                                    "pitch_error_deg": pitch_err_deg, "azimuth_error_deg": azimuth_err_deg,
                                }
                                angle_error_data.append(match_data)
                # <<< Angle Error Calculation End >>>

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                (save_dir / "labels").mkdir(parents=True, exist_ok=True)
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred

        callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
        if not os.path.exists(anno_json):
            anno_json = os.path.join(data["path"], "annotations", "instances_val2017.json")
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements("pycocotools>=2.0.6")
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    # <<< Write Angle Error Data to CSV >>>
    if angle_error_data:
        csv_path = save_dir / 'angle_errors.csv'
        LOGGER.info(f"Saving angle error data to {csv_path}...")
        try:
            fieldnames = [
                "image_path", "pred_box_x1", "pred_box_y1", "pred_box_x2", "pred_box_y2",
                "pred_conf", "pred_cls", "pred_pitch_raw", "pred_azimuth_raw",
                "gt_box_x1", "gt_box_y1", "gt_box_x2", "gt_box_y2", "gt_cls",
                "gt_pitch_raw", "gt_azimuth_raw", "iou",
                "pitch_error_raw", "azimuth_error_raw",
                "pitch_error_deg", "azimuth_error_deg"
            ]
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(angle_error_data)

            # Calculate and Print MAE from the collected data
            all_pitch_err_deg = [abs(d['pitch_error_deg']) for d in angle_error_data]
            all_azimuth_err_deg = [abs(d['azimuth_error_deg']) for d in angle_error_data]
            mae_pitch_deg = np.mean(all_pitch_err_deg) if all_pitch_err_deg else 0
            mae_azimuth_deg = np.mean(all_azimuth_err_deg) if all_azimuth_err_deg else 0
            LOGGER.info(f"Pitch MAE (deg): {mae_pitch_deg:.2f}, Azimuth MAE (deg): {mae_azimuth_deg:.2f}")

            # <<< Call plotting function after saving CSV >>>
            if plots: # Only plot if plots are enabled
                plot_angle_errors(save_dir)
            # <<< End plotting call >>>

        except Exception as e:
            LOGGER.error(f"Failed to write angle errors to CSV: {e}")
    else:
        LOGGER.info("No matched predictions found with extra dimensions to calculate pitch/azimuth errors or save to CSV.")
    # <<< End Write CSV >>>

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    final_loss = (loss.cpu() / len(dataloader)).tolist()
    # Note: Returning MAE values here might require adjusting downstream code (e.g., train.py)
    # For now, just printing MAE. Modify return statement if needed.
    return (mp, mr, map50, map, *final_loss), maps, t


def parse_opt():
    """
    Parse command-line options for configuring YOLOv5 model inference.

    Args:
        data (str, optional): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        weights (list[str], optional): List of paths to model weight files. Default is 'yolov5s.pt'.
        batch_size (int, optional): Batch size for inference. Default is 32.
        imgsz (int, optional): Inference image size in pixels. Default is 640.
        conf_thres (float, optional): Confidence threshold for predictions. Default is 0.001.
        iou_thres (float, optional): IoU threshold for Non-Max Suppression (NMS). Default is 0.6.
        max_det (int, optional): Maximum number of detections per image. Default is 300.
        task (str, optional): Task type - options are 'train', 'val', 'test', 'speed', or 'study'. Default is 'val'.
        device (str, optional): Device to run the model on. e.g., '0' or '0,1,2,3' or 'cpu'. Default is empty to let the system choose automatically.
        workers (int, optional): Maximum number of dataloader workers per rank in DDP mode. Default is 8.
        single_cls (bool, optional): If set, treats the dataset as a single-class dataset. Default is False.
        augment (bool, optional): If set, performs augmented inference. Default is False.
        verbose (bool, optional): If set, reports mAP by class. Default is False.
        save_txt (bool, optional): If set, saves results to *.txt files. Default is False.
        save_hybrid (bool, optional): If set, saves label+prediction hybrid results to *.txt files. Default is False.
        save_conf (bool, optional): If set, saves confidences in --save-txt labels. Default is False.
        save_json (bool, optional): If set, saves results to a COCO-JSON file. Default is False.
        project (str, optional): Project directory to save results to. Default is 'runs/val'.
        name (str, optional): Name of the directory to save results to. Default is 'exp'.
        exist_ok (bool, optional): If set, existing directory will not be incremented. Default is False.
        half (bool, optional): If set, uses FP16 half-precision inference. Default is False.
        dnn (bool, optional): If set, uses OpenCV DNN for ONNX inference. Default is False.

    Returns:
        argparse.Namespace: Parsed command-line options.

    Notes:
        - The '--data' parameter is checked to ensure it ends with 'coco.yaml' if '--save-json' is set.
        - The '--save-txt' option is set to True if '--save-hybrid' is enabled.
        - Args are printed using `print_args` to facilitate debugging.

    Example:
        To validate a trained YOLOv5 model on a COCO dataset:
        ```python
        $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640
        ```
        Different model formats could be used instead of `yolov5s.pt`:
        ```python
        $ python val.py --weights yolov5s.pt yolov5s.torchscript yolov5s.onnx yolov5s_openvino_model yolov5s.engine
        ```
        Additional options include saving results in different formats, selecting devices, and more.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/mydata.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 tasks like training, validation, testing, speed, and study benchmarks based on provided options.

    Args:
        opt (argparse.Namespace): Parsed command-line options.
            This includes values for parameters like 'data', 'weights', 'batch_size', 'imgsz', 'conf_thres',
            'iou_thres', 'max_det', 'task', 'device', 'workers', 'single_cls', 'augment', 'verbose', 'save_txt',
            'save_hybrid', 'save_conf', 'save_json', 'project', 'name', 'exist_ok', 'half', and 'dnn', essential
            for configuring the YOLOv5 tasks.

    Returns:
        None

    Examples:
        To validate a trained YOLOv5 model on the COCO dataset with a specific weights file, use:
        ```python
        $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640
        ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
