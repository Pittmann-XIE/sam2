# generate_sh.py
import json

annotations_file = '/home/pengtao/thesis/ws_ros2humble-main_lab/sam2/dataset_annotations_line_straight_random_container_scripted.json'
output_script = 'run_all.sh'
dataset_path = '/mnt/Ego2Exo/line_straight_random_container_scripted_trimmed_cropped_all'
device = 'cuda:1'

with open(annotations_file, 'r') as f:
    data = json.load(f)

with open(output_script, 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("# Auto-generated SAM 2 processing script\n\n")
    
    # Exit immediately if a command exits with a non-zero status
    f.write("set -e\n\n") 
    
    for episode, cameras in data.items():
        for camera, clicks in cameras.items():
            if clicks:  # Only write the command if there are actual clicks
                # Adding --skip_done so you can easily kill and restart the bash script
                f.write(f"python3 08_auto_label.py --episode {episode} --camera {camera} --annotations_file {annotations_file} --dataset_folder {dataset_path} --save_yolo --yolo_dir {dataset_path}/yolo_labels --device {device} \n")

print(f"Successfully generated {output_script}!")
print(f"Run it using: bash {output_script}")