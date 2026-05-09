#!/usr/bin/env python3
"""
Interactively create cam2 object/container filled region masks for every .h5 file in a folder.

For each file, the script opens one reference frame from cam2, lets the user
place ordered keypoints for two masks, fills each mask polygon, and writes:

    observations/images/cam2_rgb_aug_object_mask
    observations/images/cam2_rgb_aug_container_mask

By default the reference image is read from observations/images/cam2_rgb_aug.
If that dataset is missing, the script falls back to observations/images/cam2_rgb.

Usage:
    python3 sam2/12_create_cam2_line_masks_from_keypoints.py \
        --folder /mnt/Ego2Exo/your_dataset_folder

Controls in the annotation window:
    left click       add keypoint
    backspace/delete undo last keypoint
    enter           accept current mask
    c               clear current mask
    escape          skip current file
"""

import argparse
from pathlib import Path

import numpy as np

plt = None

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import h5py
except ModuleNotFoundError:
    h5py = None


IMAGE_DATASET = "observations/images/cam2_rgb_aug"
FALLBACK_IMAGE_DATASET = "observations/images/cam2_rgb"
OBJECT_MASK_DATASET = "observations/images/cam2_rgb_aug_object_mask"
CONTAINER_MASK_DATASET = "observations/images/cam2_rgb_aug_container_mask"


def list_h5_files(folder):
    folder_path = Path(folder).expanduser()
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    return sorted(folder_path.glob("*.h5"))


def normalize_image_for_display(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0]
    if image.dtype == np.uint8:
        return image

    image_float = image.astype(np.float32)
    if image_float.size == 0:
        return image_float

    min_value = float(np.nanmin(image_float))
    max_value = float(np.nanmax(image_float))
    if max_value <= 1.0 and min_value >= 0.0:
        return np.clip(image_float, 0.0, 1.0)
    if max_value > min_value:
        return ((image_float - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    return np.zeros_like(image_float, dtype=np.uint8)


def get_reference_frame(image_dataset, frame_index):
    if image_dataset.ndim == 4:
        if len(image_dataset) == 0:
            raise ValueError("Image dataset is empty")
        clipped_index = min(max(frame_index, 0), len(image_dataset) - 1)
        return image_dataset[clipped_index], clipped_index
    if image_dataset.ndim in (2, 3):
        return image_dataset[()], None
    raise ValueError(
        f"Expected image dataset with 2, 3, or 4 dimensions; got shape {image_dataset.shape}"
    )


def select_image_dataset(root, image_dataset, fallback_image_dataset):
    if image_dataset in root:
        return image_dataset
    if fallback_image_dataset and fallback_image_dataset in root:
        return fallback_image_dataset
    return None


class KeypointCollector:
    def __init__(self, image, title, close_path):
        self.image = normalize_image_for_display(image)
        self.title = title
        self.close_path = close_path
        self.points = []
        self.accepted = False
        self.skipped = False

        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.ax.imshow(self.image)
        self.ax.set_title(self._title_text())
        self.ax.axis("off")
        self.line_artist = None
        self.scatter_artist = None
        self.text_artists = []

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _title_text(self):
        return (
            f"{self.title}\n"
            "left click: add | backspace/delete: undo | enter: accept | c: clear | escape: skip"
        )

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.points.append((int(round(event.xdata)), int(round(event.ydata))))
        self.redraw()

    def on_key(self, event):
        if event.key in ("backspace", "delete"):
            if self.points:
                self.points.pop()
                self.redraw()
        elif event.key == "c":
            self.points = []
            self.redraw()
        elif event.key == "enter":
            self.accepted = True
            plt.close(self.fig)
        elif event.key == "escape":
            self.skipped = True
            plt.close(self.fig)

    def redraw(self):
        if self.line_artist is not None:
            self.line_artist.remove()
            self.line_artist = None
        if self.scatter_artist is not None:
            self.scatter_artist.remove()
            self.scatter_artist = None
        for artist in self.text_artists:
            artist.remove()
        self.text_artists = []

        if self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            draw_xs = xs + [xs[0]] if self.close_path and len(xs) > 2 else xs
            draw_ys = ys + [ys[0]] if self.close_path and len(ys) > 2 else ys
            (self.line_artist,) = self.ax.plot(draw_xs, draw_ys, color="lime", linewidth=2)
            self.scatter_artist = self.ax.scatter(xs, ys, color="red", s=28)

            for i, (x, y) in enumerate(self.points, start=1):
                self.text_artists.append(
                    self.ax.text(
                        x + 3,
                        y + 3,
                        str(i),
                        color="yellow",
                        fontsize=9,
                        bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
                    )
                )

        self.fig.canvas.draw_idle()

    def collect(self):
        plt.show()
        return self.points, self.accepted, self.skipped


def collect_keypoints(image, label, file_path, close_path):
    collector = KeypointCollector(
        image=image,
        title=f"{file_path.name}: draw {label} mask",
        close_path=close_path,
    )
    points, accepted, skipped = collector.collect()
    if skipped:
        return None
    if not accepted:
        return None
    return points


def rasterize_points(points, height, width, line_thickness, close_path):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not points:
        return mask

    clipped_points = np.array(
        [[np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)] for x, y in points],
        dtype=np.int32,
    )

    if len(clipped_points) == 1:
        cv2.circle(mask, tuple(clipped_points[0]), max(1, line_thickness // 2), 255, -1)
    elif close_path and len(clipped_points) >= 3:
        cv2.fillPoly(
            mask,
            [clipped_points],
            color=255,
            lineType=cv2.LINE_AA,
        )
    else:
        cv2.polylines(
            mask,
            [clipped_points],
            isClosed=close_path,
            color=255,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    return mask


def expand_mask_to_image_shape(mask_2d, image_dataset):
    if image_dataset.ndim == 4:
        return np.repeat(mask_2d[np.newaxis, :, :], image_dataset.shape[0], axis=0)
    return mask_2d


def write_mask_dataset(root, dataset_name, mask, overwrite, compression):
    if dataset_name in root:
        if not overwrite:
            raise ValueError(
                f"Dataset already exists: {dataset_name}. Use --overwrite to replace it."
            )
        del root[dataset_name]

    parent_name, leaf_name = dataset_name.rsplit("/", 1)
    parent = root.require_group(parent_name)
    parent.create_dataset(
        leaf_name,
        data=mask,
        dtype=np.uint8,
        compression=compression,
    )


def assert_outputs_can_be_written(root, dataset_names, overwrite):
    if overwrite:
        return
    existing_names = [name for name in dataset_names if name in root]
    if existing_names:
        joined_names = ", ".join(existing_names)
        raise ValueError(f"Output dataset already exists: {joined_names}. Use --overwrite.")


def annotate_file(file_path, args):
    print(f"\nOpening {file_path}")
    with h5py.File(file_path, "r+") as root:
        selected_image_dataset = select_image_dataset(
            root,
            args.image_dataset,
            args.fallback_image_dataset,
        )
        if selected_image_dataset is None:
            print(
                "  Skipping: missing image dataset "
                f"{args.image_dataset} and fallback {args.fallback_image_dataset}"
            )
            return False

        image_dataset = root[selected_image_dataset]
        reference_image, actual_frame = get_reference_frame(image_dataset, args.frame_index)
        height, width = reference_image.shape[:2]
        if actual_frame is not None:
            print(f"  Using reference frame {actual_frame} from {selected_image_dataset}")
        else:
            print(f"  Using single image from {selected_image_dataset}")

        object_points = collect_keypoints(
            reference_image,
            "object",
            file_path,
            args.close_path,
        )
        if object_points is None:
            print("  File skipped before writing object/container masks.")
            return False

        container_points = collect_keypoints(
            reference_image,
            "container",
            file_path,
            args.close_path,
        )
        if container_points is None:
            print("  File skipped before writing object/container masks.")
            return False

        object_mask = rasterize_points(
            object_points,
            height,
            width,
            args.line_thickness,
            args.close_path,
        )
        container_mask = rasterize_points(
            container_points,
            height,
            width,
            args.line_thickness,
            args.close_path,
        )

        object_dataset = expand_mask_to_image_shape(object_mask, image_dataset)
        container_dataset = expand_mask_to_image_shape(container_mask, image_dataset)

        assert_outputs_can_be_written(
            root,
            [args.object_mask_dataset, args.container_mask_dataset],
            args.overwrite,
        )

        write_mask_dataset(
            root,
            args.object_mask_dataset,
            object_dataset,
            args.overwrite,
            args.compression,
        )
        write_mask_dataset(
            root,
            args.container_mask_dataset,
            container_dataset,
            args.overwrite,
            args.compression,
        )

        print(
            f"  Wrote {args.object_mask_dataset} with shape {object_dataset.shape} "
            f"and {args.container_mask_dataset} with shape {container_dataset.shape}"
        )
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create cam2_rgb_aug object/container line masks from clicked keypoints."
    )
    parser.add_argument(
        "--folder",
        default='/mnt/Ego2Exo/line_straight_random_container_scripted_trimmed_cropped_all',
        help="Folder containing .h5 files.",
    )
    parser.add_argument(
        "--image-dataset",
        default=IMAGE_DATASET,
        help=f"Preferred image dataset to display. Default: {IMAGE_DATASET}",
    )
    parser.add_argument(
        "--fallback-image-dataset",
        default=FALLBACK_IMAGE_DATASET,
        help=f"Fallback image dataset if --image-dataset is missing. Default: {FALLBACK_IMAGE_DATASET}",
    )
    parser.add_argument(
        "--object-mask-dataset",
        default=OBJECT_MASK_DATASET,
        help=f"Output object mask dataset. Default: {OBJECT_MASK_DATASET}",
    )
    parser.add_argument(
        "--container-mask-dataset",
        default=CONTAINER_MASK_DATASET,
        help=f"Output container mask dataset. Default: {CONTAINER_MASK_DATASET}",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Reference frame index when the image dataset is time-indexed. Default: 0",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=8,
        help="Mask line thickness in pixels. Default: 8",
    )
    parser.add_argument(
        "--close-path",
        action="store_true",
        default=True,
        help="Connect the last keypoint back to the first and fill the polygon. Default: enabled",
    )
    parser.add_argument(
        "--open-path",
        dest="close_path",
        action="store_false",
        help="Draw an open polyline instead of a filled polygon.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing mask datasets if present.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        choices=("gzip", "lzf", "none"),
        help="HDF5 compression for written masks. Default: gzip",
    )
    return parser.parse_args()


def main():
    global plt

    args = parse_args()
    missing_modules = []
    try:
        import matplotlib.pyplot as loaded_plt
        plt = loaded_plt
    except ModuleNotFoundError:
        missing_modules.append("matplotlib")

    if h5py is None:
        missing_modules.append("h5py")
    if cv2 is None:
        missing_modules.append("opencv-python/cv2")
    if missing_modules:
        raise ModuleNotFoundError(
            "Missing required module(s): "
            + ", ".join(missing_modules)
            + ". Run this script in the same environment used for the dataset tools."
        )

    if args.line_thickness < 1:
        raise ValueError("--line-thickness must be >= 1")
    if args.compression == "none":
        args.compression = None

    h5_files = list_h5_files(args.folder)
    if not h5_files:
        print(f"No .h5 files found in {args.folder}")
        return

    print(f"Found {len(h5_files)} .h5 files in {args.folder}")
    written = 0
    for file_path in h5_files:
        try:
            if annotate_file(file_path, args):
                written += 1
        except Exception as exc:
            print(f"  Error while processing {file_path}: {exc}")

    print(f"\nDone. Wrote masks for {written}/{len(h5_files)} files.")


if __name__ == "__main__":
    main()
