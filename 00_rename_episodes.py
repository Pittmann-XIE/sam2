import argparse
import re
from pathlib import Path

def rename_hdf5_files(folder_path, shift):
    directory = Path(folder_path)
    
    if not directory.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        return

    # Pattern to match 'episode_X.hdf5' and capture X
    pattern = re.compile(r'episode_(\d+)\.hdf5')

    # Collect all matching files and their current IDs
    files_to_rename = []
    for file in directory.glob('episode_*.hdf5'):
        match = pattern.match(file.name)
        if match:
            file_id = int(match.group(1))
            files_to_rename.append((file, file_id))

    # Sort files to avoid overwriting during the shift
    # If shift is positive, rename highest IDs first. If negative, lowest first.
    files_to_rename.sort(key=lambda x: x[1], reverse=(shift > 0))

    print(f"Starting rename operation in: {directory}")
    
    for file_path, old_id in files_to_rename:
        new_id = old_id + shift
        new_name = f"episode_{new_id}.hdf5"
        new_path = directory / new_name
        
        # Security check: Don't overwrite if the file already exists
        if new_path.exists():
            print(f"Skipping {file_path.name} -> {new_name}: Destination already exists.")
            continue
            
        file_path.rename(new_path)
        print(f"Renamed: {file_path.name} -> {new_name}")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shift HDF5 episode filenames by an integer.")
    parser.add_argument("--folder", default='/mnt/Ego2Exo/pick_teleop_5/todo/smooth/aloha_100', type=str, help="Path to the folder containing .hdf5 files")
    parser.add_argument("--shift", default=111 ,type=int, help="Integer value to shift the IDs by")
    
    args = parser.parse_args()
    rename_hdf5_files(args.folder, args.shift)