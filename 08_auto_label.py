# # V1: save cropped images and masks into hdf5
# import os
# import cv2
# import h5py
# import json
# import torch
# import shutil
# import argparse
# import numpy as np

# from sam2.build_sam import build_sam2_video_predictor

# def main(args):
#     dataset_folder = args.dataset_folder
#     annotations_file = args.annotations_file
#     # Unique temp directory to prevent collisions if running multiple instances
#     temp_frame_dir = f'./temp_sam2_frames_{args.episode.replace(".hdf5", "")}_{args.camera}'
    
#     CROP_SIZE = (224, 224) 
#     PADDING = 10 
    
#     with open(annotations_file, 'r') as f:
#         all_annotations = json.load(f)

#     # Validate episode and camera existence
#     if args.episode not in all_annotations:
#         print(f"Error: {args.episode} not found in annotations.")
#         return
#     if args.camera not in all_annotations[args.episode]:
#         print(f"Error: {args.camera} not found in {args.episode}.")
#         return
        
#     clicks = all_annotations[args.episode][args.camera]
#     if not clicks:
#         print(f"No clicks for {args.episode} -> {args.camera}. Skipping.")
#         return

#     filepath = os.path.join(dataset_folder, args.episode)
#     episode_name = os.path.splitext(args.episode)[0]
    
#     # --- Check for existing datasets if skip_done is enabled ---
#     if args.skip_done:
#         if os.path.exists(filepath):
#             with h5py.File(filepath, 'r') as h5f:
#                 # Handle both list (old) and dict (new) formats for checking
#                 parsed_ids = [str(i + 1) for i in range(len(clicks))] if isinstance(clicks, list) else clicks.keys()
#                 is_fully_processed = True
#                 for obj_id_str in parsed_ids:
#                     expected_dataset = f'observations/images/{args.camera}_cropped_{obj_id_str}'
#                     if expected_dataset not in h5f:
#                         is_fully_processed = False
#                         break
                
#                 if is_fully_processed:
#                     print(f"Skipping {args.episode} -> {args.camera} - already processed.")
#                     return

#     # Setup Device
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         # Matmul and cuDNN optimizations for Ampere+ GPUs (RTX 30 series, A100, etc.)
#         if torch.cuda.get_device_properties(0).major >= 8:
#             torch.backends.cuda.matmul.allow_tf32 = True
#             torch.backends.cudnn.allow_tf32 = True
#     else:
#         device = torch.device("cpu")

#     predictor = build_sam2_video_predictor(
#         "configs/sam2.1/sam2.1_hiera_l.yaml", 
#         "./checkpoints/sam2.1_hiera_large.pt", 
#         device=device
#     )

#     with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#         with h5py.File(filepath, 'a') as h5f:
#             # --- Backward Compatibility Fix for annotation formats ---
#             if isinstance(clicks, list):
#                 parsed_clicks = {str(i + 1): pt for i, pt in enumerate(clicks)}
#             else:
#                 parsed_clicks = clicks
                
#             print(f"Processing {args.episode} -> {args.camera}")
#             dataset_path = f'observations/images/{args.camera}'
#             video_array = h5f[dataset_path][:] 
#             num_frames, img_h, img_w, _ = video_array.shape
            
#             # Setup YOLO directory
#             yolo_cam_dir = os.path.join(args.yolo_dir, episode_name, args.camera)
#             if args.save_yolo:
#                 os.makedirs(yolo_cam_dir, exist_ok=True)

#             # Dump HDF5 frames to temp JPEG files for SAM 2 consumption
#             os.makedirs(temp_frame_dir, exist_ok=True)
#             for idx, frame in enumerate(video_array):
#                 # SAM 2 expects BGR if using OpenCV-style loading
#                 frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#                 cv2.imwrite(os.path.join(temp_frame_dir, f"{idx:05d}.jpg"), frame_bgr)
            
#             inference_state = predictor.init_state(video_path=temp_frame_dir)
#             predictor.reset_state(inference_state)
            
#             # Add prompts for all objects
#             for obj_id_str, pt in parsed_clicks.items():
#                 ann_obj_id = int(obj_id_str) 
#                 predictor.add_new_points_or_box(
#                     inference_state=inference_state,
#                     frame_idx=0,
#                     obj_id=ann_obj_id,
#                     points=np.array([pt], dtype=np.float32),
#                     labels=np.array([1], np.int32)
#                 )
            
#             # Prepare storage for resized crops
#             cropped_videos = {
#                 int(obj_id_str): np.zeros((num_frames, CROP_SIZE[1], CROP_SIZE[0], 3), dtype=np.uint8) 
#                 for obj_id_str in parsed_clicks.keys()
#             }
            
#             # Tracking and extraction loop
#             for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#                 raw_frame = video_array[out_frame_idx]
#                 yolo_lines = []
                
#                 for i, out_obj_id in enumerate(out_obj_ids):
#                     mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
#                     y_indices, x_indices = np.where(mask > 0)
                    
#                     if len(y_indices) > 0 and len(x_indices) > 0:
#                         # 1. Calculate the TIGHT bounding box (with PADDING) for YOLO
#                         x_min_t = max(0, x_indices.min() - PADDING)
#                         x_max_t = min(img_w - 1, x_indices.max() + PADDING)
#                         y_min_t = max(0, y_indices.min() - PADDING)
#                         y_max_t = min(img_h - 1, y_indices.max() + PADDING)
                        
#                         # 2. Save YOLO annotations using the tight box
#                         if args.save_yolo:
#                             box_w = x_max_t - x_min_t
#                             box_h = y_max_t - y_min_t
#                             x_center_norm = (x_min_t + box_w / 2.0) / img_w
#                             y_center_norm = (y_min_t + box_h / 2.0) / img_h
#                             box_w_norm = box_w / img_w
#                             box_h_norm = box_h / img_h
                            
#                             class_id = int(out_obj_id) - 1
#                             yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {box_w_norm:.6f} {box_h_norm:.6f}\n")
                        
#                         # 3. Create a SQUARE crop for ResNet (Aspect Ratio Preservation)
#                         # We calculate the side of the square based on the larger dimension
#                         side = max(x_max_t - x_min_t, y_max_t - y_min_t)
#                         cx, cy = (x_min_t + x_max_t) / 2, (y_min_t + y_max_t) / 2
                        
#                         x_min_s = int(max(0, cx - side / 2))
#                         x_max_s = int(min(img_w - 1, cx + side / 2))
#                         y_min_s = int(max(0, cy - side / 2))
#                         y_max_s = int(min(img_h - 1, cy + side / 2))
                        
#                         # Extract the squared region and resize to target ResNet size
#                         crop = raw_frame[y_min_s:y_max_s, x_min_s:x_max_s]
#                         crop_resized = cv2.resize(crop, CROP_SIZE) # Standard 224x224
#                         cropped_videos[out_obj_id][out_frame_idx] = crop_resized
                
#                 # Write YOLO file for the current frame
#                 if args.save_yolo and yolo_lines:
#                     yolo_txt_path = os.path.join(yolo_cam_dir, f"{out_frame_idx:05d}.txt")
#                     with open(yolo_txt_path, 'w') as f:
#                         f.writelines(yolo_lines)
            
#             # Save the processed crops back into the HDF5 file
#             for obj_id, crop_array in cropped_videos.items():
#                 target_dataset_name = f'observations/images/{args.camera}_cropped_{obj_id}'
#                 if target_dataset_name in h5f:
#                     del h5f[target_dataset_name]
#                 h5f.create_dataset(target_dataset_name, data=crop_array)
#                 print(f"Saved {target_dataset_name}")
            
#             # Final Cleanup: Remove temp JPEG frames before process ends
#             shutil.rmtree(temp_frame_dir)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="SAM 2 Single-Camera Squared Cropping")
#     parser.add_argument("--episode", type=str, required=True, help="Specific episode file (e.g., episode_52.hdf5)")
#     parser.add_argument("--camera", type=str, required=True, help="Specific camera (e.g., cam1_rgb)")
#     parser.add_argument("--dataset_folder", type=str, default='/mnt/Ego2Exo/pick_teleop_4/todo/smooth/aloha_100')
#     parser.add_argument("--annotations_file", type=str, default='dataset_annotations.json')
#     parser.add_argument("--skip_done", action="store_true", help="Skip if cropped datasets already exist in HDF5")
#     parser.add_argument("--save_yolo", action="store_true", default=True, help="Save normalized YOLO bounding boxes")
#     parser.add_argument("--yolo_dir", type=str, default='/mnt/Ego2Exo//mnt/Ego2Exo/pick_teleop_4/todo/smooth/aloha_100/yolo_labels', help="Root directory for YOLO labels")
    
#     args = parser.parse_args()
#     main(args)
    
    

## V2: extract and save masks
import os
import cv2
import h5py
import json
import torch
import shutil
import argparse
import numpy as np

from sam2.build_sam import build_sam2_video_predictor

def main(args):
    dataset_folder = args.dataset_folder
    annotations_file = args.annotations_file
    # Unique temp directory to prevent collisions if running multiple instances
    temp_frame_dir = f'./temp_sam2_frames_{args.episode.replace(".hdf5", "")}_{args.camera}'
    
    PADDING = 10 
    
    with open(annotations_file, 'r') as f:
        all_annotations = json.load(f)

    # Validate episode and camera existence
    if args.episode not in all_annotations:
        print(f"Error: {args.episode} not found in annotations.")
        return
    if args.camera not in all_annotations[args.episode]:
        print(f"Error: {args.camera} not found in {args.episode}.")
        return
        
    clicks = all_annotations[args.episode][args.camera]
    if not clicks:
        print(f"No clicks for {args.episode} -> {args.camera}. Skipping.")
        return

    filepath = os.path.join(dataset_folder, args.episode)
    episode_name = os.path.splitext(args.episode)[0]
    
    # --- Check for existing datasets if skip_done is enabled ---
    if args.skip_done:
        if os.path.exists(filepath):
            with h5py.File(filepath, 'r') as h5f:
                # Handle both list (old) and dict (new) formats for checking
                parsed_ids = [str(i + 1) for i in range(len(clicks))] if isinstance(clicks, list) else clicks.keys()
                is_fully_processed = True
                for obj_id_str in parsed_ids:
                    # Updated to check for the mask dataset name
                    expected_dataset = f'observations/images/{args.camera}_{obj_id_str}_mask'
                    if expected_dataset not in h5f:
                        is_fully_processed = False
                        break
                
                if is_fully_processed:
                    print(f"Skipping {args.episode} -> {args.camera} - already processed.")
                    return

    # Setup Device based on the provided argument
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
        
        # --- CRITICAL FIX ---
        # Explicitly set the active CUDA device context. 
        # This prevents custom kernels (like FlashAttention in SAM 2) 
        # from accidentally launching on cuda:0 when tensors are on cuda:1.
        torch.cuda.set_device(device)
        
        # Matmul and cuDNN optimizations for Ampere+ GPUs
        if torch.cuda.get_device_properties(device).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        if args.device.startswith("cuda"):
            print(f"Warning: {args.device} requested but CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", 
        "./checkpoints/sam2.1_hiera_large.pt", 
        device=device
    )

    # Use dynamic device type ('cuda' or 'cpu') for autocast context
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        with h5py.File(filepath, 'a') as h5f:
            # --- Backward Compatibility Fix for annotation formats ---
            if isinstance(clicks, list):
                parsed_clicks = {str(i + 1): pt for i, pt in enumerate(clicks)}
            else:
                parsed_clicks = clicks
                
            print(f"Processing {args.episode} -> {args.camera} on {device}")
            dataset_path = f'observations/images/{args.camera}'
            video_array = h5f[dataset_path][:] 
            num_frames, img_h, img_w, _ = video_array.shape
            
            # Setup YOLO directory
            yolo_cam_dir = os.path.join(args.yolo_dir, episode_name, args.camera)
            if args.save_yolo:
                os.makedirs(yolo_cam_dir, exist_ok=True)

            # Dump HDF5 frames to temp JPEG files for SAM 2 consumption
            os.makedirs(temp_frame_dir, exist_ok=True)
            for idx, frame in enumerate(video_array):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(temp_frame_dir, f"{idx:05d}.jpg"), frame_bgr)
            
            inference_state = predictor.init_state(video_path=temp_frame_dir)
            predictor.reset_state(inference_state)
            
            # Add prompts for all objects
            for obj_id_str, pt in parsed_clicks.items():
                ann_obj_id = int(obj_id_str) 
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=ann_obj_id,
                    points=np.array([pt], dtype=np.float32),
                    labels=np.array([1], np.int32)
                )
            
            # Prepare storage for precise segmentation masks (uint8 format)
            mask_videos = {
                int(obj_id_str): np.zeros((num_frames, img_h, img_w), dtype=np.uint8) 
                for obj_id_str in parsed_clicks.keys()
            }
            
            # Tracking and extraction loop
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                yolo_lines = []
                
                for i, out_obj_id in enumerate(out_obj_ids):
                    # Extract the mask and save directly to storage
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    mask_videos[out_obj_id][out_frame_idx] = mask.astype(np.uint8)
                    
                    if args.save_yolo:
                        y_indices, x_indices = np.where(mask > 0)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            x_min_t = max(0, x_indices.min() - PADDING)
                            x_max_t = min(img_w - 1, x_indices.max() + PADDING)
                            y_min_t = max(0, y_indices.min() - PADDING)
                            y_max_t = min(img_h - 1, y_indices.max() + PADDING)
                            
                            box_w = x_max_t - x_min_t
                            box_h = y_max_t - y_min_t
                            x_center_norm = (x_min_t + box_w / 2.0) / img_w
                            y_center_norm = (y_min_t + box_h / 2.0) / img_h
                            box_w_norm = box_w / img_w
                            box_h_norm = box_h / img_h
                            
                            class_id = int(out_obj_id) - 1
                            yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {box_w_norm:.6f} {box_h_norm:.6f}\n")
                
                if args.save_yolo and yolo_lines:
                    yolo_txt_path = os.path.join(yolo_cam_dir, f"{out_frame_idx:05d}.txt")
                    with open(yolo_txt_path, 'w') as f:
                        f.writelines(yolo_lines)
            
            # Save the full mask arrays to the HDF5 dataset
            for obj_id, mask_array in mask_videos.items():
                target_dataset_name = f'observations/images/{args.camera}_{obj_id}_mask'
                if target_dataset_name in h5f:
                    del h5f[target_dataset_name]
                
                # Using gzip compression for masks since 0s and 1s compress highly efficiently
                h5f.create_dataset(target_dataset_name, data=mask_array, compression="gzip")
                print(f"Saved precise mask to {target_dataset_name}")
            
            shutil.rmtree(temp_frame_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 2 Single-Camera Mask Extraction")
    parser.add_argument("--episode", type=str, required=True, help="Specific episode file (e.g., episode_52.hdf5)")
    parser.add_argument("--camera", type=str, required=True, help="Specific camera (e.g., cam1_rgb)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device to use (e.g., 'cuda:0', 'cuda:1', or 'cpu')")
    parser.add_argument("--dataset_folder", type=str, default='/mnt/Ego2Exo/line_rotate_scripted_trimmed/')
    parser.add_argument("--annotations_file", type=str, default='dataset_annotations.json')
    parser.add_argument("--skip_done", action="store_true", help="Skip if mask datasets already exist in HDF5")
    parser.add_argument("--save_yolo", action="store_true", default=True, help="Save normalized YOLO bounding boxes")
    parser.add_argument("--yolo_dir", type=str, default='/mnt/Ego2Exo/line_rotate_scripted_trimmed/yolo_labels', help="Root directory for YOLO labels")
    
    args = parser.parse_args()
    main(args)


