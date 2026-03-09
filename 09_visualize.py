import os
import cv2
import h5py
import argparse
import numpy as np

def main(args):
    filepath = os.path.join(args.dataset_folder, args.episode)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    with h5py.File(filepath, 'r') as h5f:
        # 1. Identify all cropped datasets for the specified camera
        base_path = 'observations/images'
        if base_path not in h5f:
            print(f"Error: '{base_path}' not found in {args.episode}")
            return
            
        all_keys = h5f[base_path].keys()
        prefix = f"{args.camera}_cropped_"
        cropped_keys = [k for k in all_keys if k.startswith(prefix)]
        
        if not cropped_keys:
            print(f"No cropped datasets found for camera '{args.camera}' in {args.episode}.")
            print(f"Available keys in {base_path}: {list(all_keys)}")
            return
            
        print(f"Found {len(cropped_keys)} tracked object(s) for {args.camera}.")
        
        # 2. Load all crop arrays into memory
        # They are small (224x224), so loading all frames to RAM is fast and allows smooth scrubbing
        crops_data = {}
        for key in sorted(cropped_keys):
            obj_id = key.split('_')[-1]
            crops_data[obj_id] = h5f[f"{base_path}/{key}"][:]
            
        # Get the total number of frames from the first object
        num_frames = len(list(crops_data.values())[0])
        
        # 3. Setup OpenCV Window and Trackbar
        window_name = f"Cropped Viewer | {args.episode} -> {args.camera} | Press 'Q' to quit"
        cv2.namedWindow(window_name)
        
        def update_frame(val):
            """Callback function triggered whenever the slider moves."""
            frame_idx = val
            display_images = []
            
            for obj_id, crop_array in crops_data.items():
                # Extract frame and convert RGB (HDF5) to BGR (OpenCV)
                img = crop_array[frame_idx]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Add a text label to identify the object ID
                cv2.putText(
                    img_bgr, 
                    f"Obj {obj_id}", 
                    (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )
                display_images.append(img_bgr)
            
            # Stitch all cropped images side-by-side horizontally
            combined_img = np.hstack(display_images)
            cv2.imshow(window_name, combined_img)

        # Create the slider (trackbar)
        cv2.createTrackbar("Frame", window_name, 0, num_frames - 1, update_frame)
        
        # Initialize the window with the first frame
        update_frame(0)
        
        print(f"Showing {num_frames} frames. Scrub the slider to navigate.")
        print("Press 'q' or 'ESC' in the image window to close.")
        
        # 4. Wait for user to exit
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]: # 27 is the ESC key
                break
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrub through generated HDF5 crops")
    parser.add_argument("--episode", type=str, default='episode_10.hdf5', help="Specific episode file (e.g., episode_52.hdf5)")
    parser.add_argument("--camera", type=str, default='top', help="Specific camera (e.g., cam1_rgb)")
    parser.add_argument("--dataset_folder", type=str, default='/mnt/Ego2Exo/sim_transfer_cube_scripted/', help="Path to the HDF5 dataset folder")
    
    args = parser.parse_args()
    main(args)