import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_h5_masks(file_path, frame_idx=0):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        images_grp = f['observations/images']
        
        # Define the cameras we want to visualize
        cameras = ['cam1', 'cam2', 'cam3']
        num_frames = images_grp['cam1_rgb'].shape[0]

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        plt.subplots_adjust(bottom=0.15)

        def update_plots(idx):
            for i, cam in enumerate(cameras):
                # 1. Original RGB
                rgb = images_grp[f'{cam}_rgb'][idx]
                
                # 2. Mask 1 Overlay
                mask1 = images_grp[f'{cam}_rgb_1_mask'][idx]
                # 3. Mask 2 Overlay
                mask2 = images_grp[f'{cam}_rgb_2_mask'][idx]

                # Plotting
                axes[i, 0].imshow(rgb)
                axes[i, 0].set_title(f'{cam} RGB')
                
                # Mask 1 visualization (Green overlay)
                axes[i, 1].imshow(rgb)
                axes[i, 1].imshow(mask1, alpha=0.5, cmap='Greens')
                axes[i, 1].set_title(f'{cam} Mask 1')

                # Mask 2 visualization (Red overlay)
                axes[i, 2].imshow(rgb)
                axes[i, 2].imshow(mask2, alpha=0.5, cmap='Reds')
                axes[i, 2].set_title(f'{cam} Mask 2')

                for ax in axes[i]:
                    ax.axis('off')
            
            fig.suptitle(f"Frame {idx} / {num_frames - 1}", fontsize=16)
            fig.canvas.draw_idle()

        # Initial plot
        update_plots(frame_idx)

        # Add a slider for frame navigation
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=frame_idx, valfmt='%d')

        def on_change(val):
            update_plots(int(val))

        slider.on_changed(on_change)
        plt.show()

if __name__ == "__main__":
    h5_path = '/mnt/Ego2Exo/line_rotate_scripted_distractors_trimmed/episode_0.h5'
    visualize_h5_masks(h5_path)