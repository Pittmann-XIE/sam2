#!/usr/bin/env python3
"""
Visualize cam2 region-aware placement masks.

Green marks the object/cuboid placement region from:
    observations/images/cam2_rgb_aug_object_mask

Yellow marks the container placement region from:
    observations/images/cam2_rgb_aug_container_mask

By default this reads:
    <dataset_dir>/cam2_line_masks.h5

If --episode-path is provided, the masks are overlaid on a cam2 reference frame.
Otherwise the script writes a black-background mask visualization.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


OBJECT_MASK_KEY = "observations/images/cam2_rgb_aug_object_mask"
CONTAINER_MASK_KEY = "observations/images/cam2_rgb_aug_container_mask"
DEFAULT_IMAGE_KEYS = (
    "observations/images/cam2_rgb_aug",
    "observations/images/cam2_rgb",
)


def read_mask(root, key, frame_index):
    if key not in root:
        raise KeyError(f"Missing required mask dataset: {key}")

    dataset = root[key]
    if dataset.ndim == 2:
        mask = dataset[()]
    elif dataset.ndim == 3:
        if dataset.shape[0] == 0:
            raise ValueError(f"Mask dataset is empty: {key}")
        index = min(max(frame_index, 0), dataset.shape[0] - 1)
        mask = dataset[index]
    else:
        raise ValueError(f"Expected 2D or 3D mask for {key}; got shape {dataset.shape}")

    return (np.asarray(mask) > 0)


def normalize_image(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected image with shape HxW or HxWxC; got {image.shape}")

    image = image[:, :, :3]
    if image.dtype == np.uint8:
        return image

    image_f = image.astype(np.float32)
    min_value = float(np.nanmin(image_f))
    max_value = float(np.nanmax(image_f))
    if min_value >= 0.0 and max_value <= 1.0:
        return np.clip(image_f * 255.0, 0, 255).astype(np.uint8)
    if max_value > min_value:
        scaled = (image_f - min_value) / (max_value - min_value) * 255.0
        return np.clip(scaled, 0, 255).astype(np.uint8)
    return np.zeros_like(image_f, dtype=np.uint8)


def read_reference_image(episode_path, frame_index, image_keys):
    if episode_path is None:
        return None

    with h5py.File(episode_path, "r") as root:
        selected_key = next((key for key in image_keys if key in root), None)
        if selected_key is None:
            raise KeyError(
                f"None of the requested image datasets exist in {episode_path}: {image_keys}"
            )

        dataset = root[selected_key]
        if dataset.ndim == 4:
            if dataset.shape[0] == 0:
                raise ValueError(f"Image dataset is empty: {selected_key}")
            index = min(max(frame_index, 0), dataset.shape[0] - 1)
            image = dataset[index]
        elif dataset.ndim in (2, 3):
            image = dataset[()]
        else:
            raise ValueError(
                f"Expected 2D, 3D, or 4D image dataset for {selected_key}; got {dataset.shape}"
            )

    return normalize_image(image)


def resize_mask_if_needed(mask, shape_hw, name):
    if mask.shape == shape_hw:
        return mask
    if cv2 is None:
        raise ModuleNotFoundError(
            f"Mask {name} has shape {mask.shape}, image has shape {shape_hw}, "
            "and cv2 is required to resize masks."
        )
    resized = cv2.resize(
        mask.astype(np.uint8),
        (shape_hw[1], shape_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def make_visualization(object_mask, container_mask, base_image, alpha):
    height, width = object_mask.shape
    if container_mask.shape != (height, width):
        raise ValueError(
            "Object and container masks must have the same shape; got "
            f"{object_mask.shape} and {container_mask.shape}"
        )

    if base_image is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        canvas = base_image.copy()
        object_mask = resize_mask_if_needed(object_mask, canvas.shape[:2], "object")
        container_mask = resize_mask_if_needed(container_mask, canvas.shape[:2], "container")

    color_layer = np.zeros_like(canvas)
    color_layer[object_mask] = np.array([0, 255, 0], dtype=np.uint8)
    color_layer[container_mask] = np.array([255, 255, 0], dtype=np.uint8)

    colored_pixels = object_mask | container_mask
    output = canvas.copy()
    output[colored_pixels] = (
        (1.0 - alpha) * canvas[colored_pixels].astype(np.float32)
        + alpha * color_layer[colored_pixels].astype(np.float32)
    ).astype(np.uint8)
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize cam2 region-aware placement masks as green/yellow overlays."
    )
    parser.add_argument(
        "--dataset-dir",
        default="/mnt/Ego2Exo/line_straight_random_container_scripted_trimmed_cropped_all",
        help="Dataset folder containing cam2_line_masks.h5.",
    )
    parser.add_argument(
        "--mask-path",
        default=None,
        help="Optional explicit path to cam2_line_masks.h5.",
    )
    parser.add_argument(
        "--episode-path",
        default=None,
        help="Optional episode_*.h5 path used as the background image.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index for time-indexed masks/images. Default: 0",
    )
    parser.add_argument(
        "--output",
        default="cam2_region_placement_masks.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Overlay opacity from 0.0 to 1.0. Default: 0.65",
    )
    parser.add_argument(
        "--image-key",
        action="append",
        default=None,
        help=(
            "Image dataset key to try for the optional background. "
            "Can be repeated. Defaults to cam2_rgb_aug then cam2_rgb."
        ),
    )
    return parser.parse_args()


def main():
    if cv2 is None:
        raise ModuleNotFoundError(
            "Missing required module: cv2. Run this script in the same environment "
            "used for the dataset tools."
        )

    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0.0 and 1.0")

    mask_path = Path(args.mask_path) if args.mask_path else Path(args.dataset_dir) / "cam2_line_masks.h5"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

    with h5py.File(mask_path, "r") as root:
        object_mask = read_mask(root, OBJECT_MASK_KEY, args.frame_index)
        container_mask = read_mask(root, CONTAINER_MASK_KEY, args.frame_index)

    image_keys = tuple(args.image_key) if args.image_key else DEFAULT_IMAGE_KEYS
    base_image = read_reference_image(args.episode_path, args.frame_index, image_keys)
    visualization = make_visualization(object_mask, container_mask, base_image, args.alpha)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    print(f"Wrote {output_path}")
    print(f"  green: {OBJECT_MASK_KEY} ({int(object_mask.sum())} pixels)")
    print(f"  yellow: {CONTAINER_MASK_KEY} ({int(container_mask.sum())} pixels)")


if __name__ == "__main__":
    main()
