# generate_sh.py
import json

annotations_file = 'dataset_annotations.json'
output_script = 'run_all.sh'

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
                f.write(f"python3 08_auto_label.py --episode {episode} --camera {camera}\n")

print(f"Successfully generated {output_script}!")
print(f"Run it using: bash {output_script}")