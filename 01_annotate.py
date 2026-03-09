# import os
# import cv2
# import h5py
# import json

# clicked_points = []

# def get_points_from_user(event, x, y, flags, param):
#     """Callback function to capture mouse click coordinates."""
#     global clicked_points
#     if event == cv2.EVENT_LBUTTONDOWN:
#         clicked_points.append([x, y])

# def main():
#     global clicked_points
#     dataset_folder = '/mnt/Ego2Exo/pick_teleop_4/todo/smooth/aloha_100'
#     annotations_file = 'dataset_annotations.json'
#     cameras = ['cam1_rgb', 'cam2_rgb', 'aria_rgb']
    
#     # Load existing annotations to allow resuming where you left off
#     all_annotations = {}
#     if os.path.exists(annotations_file):
#         with open(annotations_file, 'r') as f:
#             all_annotations = json.load(f)
#             print(f"Loaded existing progress from {annotations_file}")

#     valid_files = [f for f in os.listdir(dataset_folder) if f.endswith('.hdf5')]
#     quit_flag = False
    
#     for filename in valid_files:
#         if quit_flag:
#             break
            
#         filepath = os.path.join(dataset_folder, filename)
#         if filename not in all_annotations:
#             all_annotations[filename] = {}
            
#         try:
#             with h5py.File(filepath, 'r') as f:
#                 for cam in cameras:
#                     # Skip if this camera view was already annotated in a previous session
#                     if cam in all_annotations[filename]:
#                         print(f"Skipping {filename} -> {cam} (already annotated).")
#                         continue
                        
#                     dataset_path = f'observations/images/{cam}'
#                     if dataset_path not in f:
#                         continue
                    
#                     # Assuming frames are stored as (N, H, W, C) numpy arrays
#                     first_frame = f[dataset_path][0] 
#                     first_img_cv2 = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
                    
#                     window_name = f"Annotating: {filename} | {cam}"
#                     cv2.namedWindow(window_name)
#                     cv2.setMouseCallback(window_name, get_points_from_user)
                    
#                     print(f"\n--- Now viewing: {filename} ({cam}) ---")
#                     print("Controls:")
#                     print("  Left Click : Add point")
#                     print("  'z'        : Undo last point")
#                     print("  'c'        : Clear all points")
#                     print("  's'        : Skip this camera")
#                     print("  'Enter'    : Confirm and move to next")
#                     print("  'q'        : Quit and save progress")
                    
#                     clicked_points = [] 
#                     skip_camera = False
                    
#                     while True:
#                         temp_img = first_img_cv2.copy()
                        
#                         # Draw instructions directly on the image canvas
#                         cv2.putText(temp_img, "Enter: Confirm | z: Undo | c: Clear | s: Skip | q: Quit", 
#                                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    
#                         for i, pt in enumerate(clicked_points):
#                             cv2.circle(temp_img, tuple(pt), 5, (0, 255, 0), -1)
#                             cv2.putText(temp_img, f"ID:{i+1}", (pt[0]+10, pt[1]-10), 
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
#                         cv2.imshow(window_name, temp_img)
#                         key = cv2.waitKey(20) & 0xFF
                        
#                         if key == 13 or key == 32: # Enter or Space
#                             break
#                         elif key == ord('z'): # Undo
#                             if clicked_points:
#                                 clicked_points.pop()
#                         elif key == ord('c'): # Clear
#                             clicked_points.clear()
#                         elif key == ord('s'): # Skip
#                             skip_camera = True
#                             clicked_points.clear()
#                             break
#                         elif key == ord('q'): # Quit
#                             quit_flag = True
#                             break
                            
#                     cv2.destroyAllWindows()
                    
#                     if quit_flag:
#                         break
                        
#                     # Save the results
#                     if not skip_camera:
#                         all_annotations[filename][cam] = list(clicked_points)
#                     else:
#                         all_annotations[filename][cam] = [] # Mark as skipped
                        
#                     # Incremental save: Write to JSON immediately
#                     with open(annotations_file, 'w') as out_f:
#                         json.dump(all_annotations, out_f, indent=4)
                        
#         except Exception as e:
#             print(f"Error reading {filename}: {e}")

#     print("\nAnnotation process finished or halted.")
#     print(f"All progress is safely saved to {annotations_file}")

# if __name__ == "__main__":
#     main()


### 

import os
import cv2
import h5py
import json
import argparse

# Use a dictionary to map specific explicit IDs to coordinates: {obj_id: [x, y]}
clicked_points = {}
active_obj_id = 1

def get_points_from_user(event, x, y, flags, param):
    """Callback function to capture mouse click coordinates for the active ID."""
    global clicked_points, active_obj_id
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points[active_obj_id] = [x, y]

def main(args):
    global clicked_points, active_obj_id
    
    cameras = args.camera_names
    
    # Load existing annotations
    all_annotations = {}
    if os.path.exists(args.annotations_file):
        with open(args.annotations_file, 'r') as f:
            all_annotations = json.load(f)
            print(f"Loaded existing progress from {args.annotations_file}")

    # Determine which files to process
    if args.episode:
        valid_files = [args.episode]
        if not os.path.exists(os.path.join(args.dataset_folder, args.episode)):
            print(f"Error: {args.episode} not found in {args.dataset_folder}")
            return
        print(f"Targeting specific episode: {args.episode} (Will overwrite existing data)")
    else:
        valid_files = [f for f in os.listdir(args.dataset_folder) if f.endswith('.hdf5')]

    quit_flag = False
    
    for filename in valid_files:
        if quit_flag:
            break
            
        filepath = os.path.join(args.dataset_folder, filename)
        if filename not in all_annotations:
            all_annotations[filename] = {}
            
        try:
            with h5py.File(filepath, 'r') as f:
                for cam in cameras:
                    # Skip if already annotated AND we are not forcing a redo of a specific episode
                    if cam in all_annotations[filename] and not args.episode:
                        print(f"Skipping {filename} -> {cam} (already annotated).")
                        continue
                        
                    dataset_path = f'observations/images/{cam}'
                    if dataset_path not in f:
                        print(f"Camera {cam} not found in {filename}, skipping.")
                        continue
                    
                    first_frame = f[dataset_path][0] 
                    first_img_cv2 = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
                    
                    window_name = f"Annotating: {filename} | {cam}"
                    cv2.namedWindow(window_name)
                    cv2.setMouseCallback(window_name, get_points_from_user)
                    
                    clicked_points = {} 
                    active_obj_id = 1
                    skip_camera = False
                    
                    while True:
                        temp_img = first_img_cv2.copy()
                        
                        # UI Overlays
                        cv2.putText(temp_img, f"ACTIVE ID: {active_obj_id} (Press 1-9 to change)", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(temp_img, "Enter: Confirm | z: Delete Active | c: Clear All | s: Skip | q: Quit", 
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    
                        for obj_id, pt in clicked_points.items():
                            color = (0, 255, 0) if obj_id == active_obj_id else (255, 0, 0)
                            cv2.circle(temp_img, tuple(pt), 5, color, -1)
                            cv2.putText(temp_img, f"ID:{obj_id}", (pt[0]+10, pt[1]-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                        cv2.imshow(window_name, temp_img)
                        key = cv2.waitKey(20) & 0xFF
                        
                        if key == 13 or key == 32: # Enter or Space
                            break
                        elif ord('1') <= key <= ord('9'): # Change Active ID
                            active_obj_id = int(chr(key))
                        elif key == ord('z'): # Delete point for Active ID
                            if active_obj_id in clicked_points:
                                del clicked_points[active_obj_id]
                        elif key == ord('c'): # Clear all points
                            clicked_points.clear()
                        elif key == ord('s'): # Skip camera
                            skip_camera = True
                            clicked_points.clear()
                            break
                        elif key == ord('q'): # Quit
                            quit_flag = True
                            break
                            
                    cv2.destroyAllWindows()
                    
                    if quit_flag:
                        break
                        
                    # Save the results (convert to string keys for JSON compatibility)
                    if not skip_camera:
                        all_annotations[filename][cam] = {str(k): v for k, v in clicked_points.items()}
                    else:
                        all_annotations[filename][cam] = {} # Mark as skipped
                        
                    # Incremental save
                    with open(args.annotations_file, 'w') as out_f:
                        json.dump(all_annotations, out_f, indent=4)
                        
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print("\nAnnotation process finished or halted.")
    print(f"Progress is safely saved to {args.annotations_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default='/mnt/Ego2Exo/sim_transfer_cube_scripted', help="Folder containing .h5 files")
    parser.add_argument("--annotations_file", type=str, default='dataset_annotations_cube.json', help="Output JSON file")
    parser.add_argument("--episode", type=str, default=None, help="Specific .h5 filename to redo/annotate (e.g. 'episode_01.h5')")
    parser.add_argument("--camera_names", nargs='+', default=['top'], help="List of camera names to process (e.g. --camera_names cam1_rgb cam2_rgb)")
    args = parser.parse_args()
    
    main(args)