"""Verbatim frozen host functions; extracted without changing their arithmetic."""
from types import SimpleNamespace
import sys
import types
import cv2
import numpy as np
import torch
from PIL import Image

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

audit = SimpleNamespace(require=require)
base = SimpleNamespace(INPUT_SIDE=768)

def letterbox_image(rgb, size):
    height, width = rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    output = np.zeros((size, size, 3), dtype=np.uint8)
    output[top : top + resized_height, left : left + resized_width] = resized
    return output, float(scale), int(left), int(top)


def normalized_images(values, device):
    normalized = values.astype(np.float32) / 255.0
    normalized = (
        normalized - np.asarray([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
    ) / np.asarray([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
    return torch.from_numpy(np.ascontiguousarray(normalized)).to(device, non_blocking=False)


def cxcywh_to_xyxy(boxes):
    center = boxes[..., :2]
    half = boxes[..., 2:] * 0.5
    return np.concatenate([center - half, center + half], axis=-1)


def restore_original_box(box, width, height):
    scale = min(base.INPUT_SIDE / width, base.INPUT_SIDE / height)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (base.INPUT_SIDE - rw) // 2, (base.INPUT_SIDE - rh) // 2
    offset = np.asarray([left, top, left, top], np.float32)
    result = (np.asarray(box, np.float32) * np.float32(base.INPUT_SIDE) - offset) / np.float32(scale)
    require(result.dtype == np.float32 and np.isfinite(result).all(), "Invalid FP32 original box")
    return result


def finite_array(value, name):
    audit.require(value.dtype == np.float32 and np.isfinite(value).all(), f"{name} must be finite FP32")


def square20_fp32(box):
    finite_array(box, "original AABB")
    center = (box[:2] + box[2:]) * np.float32(0.5)
    side = max(box[2] - box[0], box[3] - box[1], np.float32(2.0)) * np.float32(1.4)
    return np.concatenate((center - side * np.float32(0.5), center + side * np.float32(0.5))).astype(np.float32)


def square_transform_fp32(roi):
    finite_array(roi, "square20 ROI")
    width, height = roi[2] - roi[0], roi[3] - roi[1]
    audit.require(width > 0 and height > 0, "Invalid square20 extent")
    side = max(64, int(round(float(width))))
    sx, sy = np.float32(side - 1) / width, np.float32(side - 1) / height
    # Axis-aligned form of the frozen four-corner square20 homography.
    matrix = np.asarray([[sx, 0, -sx * roi[0]], [0, sy, -sy * roi[1]], [0, 0, 1]], np.float32)
    finite_array(matrix, "square20 transform")
    return side, matrix


def transform_points_fp32(points, matrix):
    finite_array(points, "prompt corners")
    homogeneous = np.concatenate((points, np.ones((len(points), 1), np.float32)), axis=1)
    transformed = homogeneous @ matrix.T
    result = transformed[:, :2] / transformed[:, 2:3]
    finite_array(result, "transformed prompt corners")
    return result


def resize_rgb_fp32(crop, height, width):
    finite_array(crop, "square20 crop pixels")
    # Keep Pillow's frozen bilinear filter, using float-mode channels to avoid RGB8 rounding.
    channels = [np.asarray(Image.fromarray(crop[:, :, channel]).resize(
        (width, height), resample=Image.Resampling.BILINEAR), dtype=np.float32) for channel in range(3)]
    result = np.stack(channels, axis=-1)
    finite_array(result, "resized crop pixels")
    return result


def transform_points(points, matrix):
    source = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(source, matrix)[0].astype(np.float64)


def install_optional_training_stubs():
    mmdet = types.ModuleType("mmdet")
    models = types.ModuleType("mmdet.models")
    dense_heads = types.ModuleType("mmdet.models.dense_heads")
    necks = types.ModuleType("mmdet.models.necks")
    unused_rpn = type("UnusedRPN", (object,), {})
    dense_heads.RPNHead = unused_rpn
    dense_heads.CenterNetUpdateHead = unused_rpn
    necks.FPN = type("UnusedFPN", (object,), {})
    mmdet.models = models
    models.dense_heads = dense_heads
    models.necks = necks
    sys.modules.update(
        {
            "mmdet": mmdet,
            "mmdet.models": models,
            "mmdet.models.dense_heads": dense_heads,
            "mmdet.models.necks": necks,
        }
    )
    projects = types.ModuleType("projects")
    efficient_det = types.ModuleType("projects.EfficientDet")
    efficient_det.efficientdet = None
    projects.EfficientDet = efficient_det
    sys.modules.update({"projects": projects, "projects.EfficientDet": efficient_det})
    mmengine = types.ModuleType("mmengine")
    mmengine.ConfigDict = dict
    sys.modules["mmengine"] = mmengine
