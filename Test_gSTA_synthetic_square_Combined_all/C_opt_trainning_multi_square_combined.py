# %%
pre_seq_length = 10
aft_seq_length = 20

# %%
import pickle

# with open('../reformatedNDDs/dataset_exp_sep_9_split_4_input_10_output_20_original.pkl', 'rb') as f:
# with open('../reformatedNDDs/dataset_exp_sep_9_split_4_input_10_output_20_notbl.pkl', 'rb') as f:
# with open('../reformatedNDDs/dataset_exp_sep_9_split_4_input_10_output_20_bl.pkl', 'rb') as f:
with open('../reformatedNDDs/dataset_16k_20k_al_256_11282024.pkl', 'rb') as f:
    dataset = pickle.load(f)

train_x, train_y = dataset['X_train'], dataset['Y_train']
print(f"X_train: {dataset['X_train'].shape}")
print(f"Y_train: {dataset['Y_train'].shape}")
print(f"X_test: {dataset['X_test'].shape}")
print(f"Y_test: {dataset['Y_test'].shape}")
print(f"X_val: {dataset['X_val'].shape}")
print(f"Y_val: {dataset['Y_val'].shape}")

# %%
import openstl
import importlib
importlib.reload(openstl)

from openstl.utils import show_video_line
import numpy as np

# show the given frames from an example
example_idx = np.random.randint(0, len(train_x))
show_video_line(train_x[example_idx], ncols=pre_seq_length, vmax=1.0, cbar=False, out_path=None, format='png', use_rgb=False)
# show the future frames from an example
show_video_line(train_y[example_idx], ncols=aft_seq_length, vmax=1.0, cbar=False, out_path=None, format='png', use_rgb=False)

# %%
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, Y, normalize=False, data_name='custom'):
        super(CustomDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.mean = None
        self.std = None
        self.data_name = data_name

        if normalize:
            # get the mean/std values along the channel dimension
            mean = data.mean(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            std = data.std(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            data = (data - mean) / std
            self.mean = mean
            self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels

# %%
X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset['X_train'], dataset[
    'X_val'], dataset['X_test'], dataset['Y_train'], dataset['Y_val'], dataset['Y_test']

# # Select the first few samples for each dataset split
# X_train = X_train[:500, :, :, :245, :270]
# Y_train = Y_train[:500, :, :, :245, :270]
# X_val = X_val[:100, :, :, :245, :270]
# Y_val = Y_val[:100, :, :, :245, :270]
# X_test = X_test[:50, :, :, :245, :270]
# Y_test = Y_test[:50, :, :, :245, :270]

# X_train = np.transpose(X_train, (0, 1, 3, 2, 4))
# Y_train = np.transpose(Y_train, (0, 1, 3, 2, 4))
# X_val = np.transpose(X_val, (0, 1, 3, 2, 4))
# Y_val = np.transpose(Y_val, (0, 1, 3, 2, 4))
# X_test = np.transpose(X_test, (0, 1, 3, 2, 4))
# Y_test = np.transpose(Y_test, (0, 1, 3, 2, 4))

# Function to scale each image individually to [0, 1] based on its own min and max values
def scale_image_to_01(data):
    # Scale each image in the data independently
    min_vals = data.min(axis=(2, 3, 4), keepdims=True)  # Min per image
    max_vals = data.max(axis=(2, 3, 4), keepdims=True)  # Max per image
    return (data - min_vals) / (max_vals - min_vals + 1e-8)  # Add small value to avoid division by zero

# Apply scaling to each dataset
X_train = scale_image_to_01(X_train)
Y_train = scale_image_to_01(Y_train)
X_val = scale_image_to_01(X_val)
Y_val = scale_image_to_01(Y_val)
X_test = scale_image_to_01(X_test)
Y_test = scale_image_to_01(Y_test)

# # Verify scaling by checking min and max values for a few images
# print(f"X_train range for first image: {X_train[0].min()} - {X_train[0].max()}")
# print(f"Y_train range for first image: {Y_train[0].min()} - {Y_train[0].max()}")
# print(f"X_val range for first image: {X_val[0].min()} - {X_val[0].max()}")
# print(f"Y_val range for first image: {Y_val[0].min()} - {Y_val[0].max()}")
# print(f"X_test range for first image: {X_test[0].min()} - {X_test[0].max()}")
# print(f"Y_test range for first image: {Y_test[0].min()} - {Y_test[0].max()}")

# Assuming X_train, Y_train, etc., are numpy arrays with dimensions matching the expected input
train_set = CustomDataset(X=X_train, Y=Y_train)
val_set = CustomDataset(X=X_val, Y=Y_val)
test_set = CustomDataset(X=X_test, Y=Y_test)
# Concatenate train, val, and test sets for the final test set
# X_all = np.concatenate([X_train, X_val, X_test], axis=0)
# Y_all = np.concatenate([Y_train, Y_val, Y_test], axis=0)
# test_set = CustomDataset(X=X_all, Y=Y_all)

# %%
batch_size = 2

dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=True, pin_memory=True)

# %%
custom_training_config = {
    'pre_seq_length': pre_seq_length,
    'aft_seq_length': aft_seq_length,
    'total_length': pre_seq_length + aft_seq_length,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 250,
    'lr': 0.001,   
    'metrics': ['mse', 'mae'],
    # 'metrics': ['mse', 'mae', 'perceptual'],
    'warmup_lr': 1e-6,  # Matching benchmark warmup learning rate
    'warmup_epoch': 5,  # Matching benchmark warmup epochs
    'sched': 'onecycle',  # Matching benchmark scheduler
    'min_lr': 1e-6,  # Matching benchmark minimum learning rate
    'final_div_factor': 1e4,  # Matching benchmark onecycle final divisor
    'decay_epoch': 100,  # Matching benchmark decay epoch
    'decay_rate': 0.1,  # Matching benchmark decay rate
    # 'use_augment': False,
    'ex_name': 'custom_exp',
    'dataname': 'custom',
    # 'dataname': 'mmnist',
    # 'in_shape': [10, 1, 32, 32],
    'in_shape': [10, 1, 256, 256],
    # 'in_shape': [10, 1, 64, 64],
    # 'in_shape': [20, 1, 256, 256],
    'drop': 0.25,
}

custom_model_config = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'method': 'SimVP',
    # 'model_type': 'swin',
    # Users can either using a config file or directly set these hyperparameters 
    # 'config_file': 'configs/custom/example_model.py',
    
    # Here, we directly set these parameters
    'model_type': 'gSTA',
    'N_S': 4,
    'N_T': 16,
    'hid_S': 64,
    'hid_T': 256
}

# %%
from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser

args = create_parser().parse_args([])
config = args.__dict__

args.pretrained_ckpt = None
# args.pretrained_ckpt = "./work_dirs/custom_exp/checkpoints/best-epoch=00-val_loss=0.009.ckpt"

args.combined = True
args.alpha = 0.5 # mse strength

# update the training config
config.update(custom_training_config)
# update the model config
config.update(custom_model_config)
# fulfill with default values
default_values = default_parser()
for attribute in default_values.keys():
    if config[attribute] is None:
        config[attribute] = default_values[attribute]
        
exp = BaseExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test), strategy='auto')

# # %%
# torch.set_float32_matmul_precision('high')

# print('>'*35 + ' training ' + '<'*35)
# exp.train()

# print('>'*35 + ' testing  ' + '<'*35)
# exp.test()

# %%
import numpy as np
from openstl.utils import show_video_line

# Load the inputs, predictions, and true values
inputs = np.load('./work_dirs/custom_exp/saved/inputs.npy')
preds = np.load('./work_dirs/custom_exp/saved/preds.npy')
trues = np.load('./work_dirs/custom_exp/saved/trues.npy')

# %%
# Generate a random index for the example
example_idx = np.random.randint(0, len(inputs))

# Show the frames for the input, prediction, and ground truth
show_video_line(inputs[example_idx], ncols=10, vmax=0.6, cbar=False, out_path=None, format='png', use_rgb=False)
show_video_line(preds[example_idx], ncols=20, vmax=0.6, cbar=False, out_path=None, format='png', use_rgb=False)
show_video_line(trues[example_idx], ncols=20, vmax=0.6, cbar=False, out_path=None, format='png', use_rgb=False)

# %%
import numpy as np
import matplotlib.pyplot as plt

def ComputeTestError(prediction, target):
    num_pixels = np.prod(target.shape)
    squared_error = np.sum((target - prediction) ** 2) / num_pixels
    phi_gt_max = np.max(target)
    phi_gt_min = np.min(target)
    mre = np.sqrt(squared_error) / (phi_gt_max - phi_gt_min) * 100
    
    return mre

def TestErrorPlot(pred, true, figsize=(10, 6), dpi=200, out_path="./statistics_testdata.png"):
    """Plot the test error for the given predictions and true data, save as PNG."""
    error_List = []
    testID_List = []
    count = 1

    # Get the number of cases and number of comparisons from pred.shape
    num_cases = pred.shape[0]
    num_comparisons = pred.shape[1]

    # Loop through each case in the prediction and true arrays
    for i in range(num_cases):
        # Use a qualitative colormap for contrasting colors (e.g., tab20 colormap)
        color = plt.cm.tab20(i % 20)  # Ensures up to 20 distinct colors
        
        # For each case, we calculate the error for all comparisons (inferred from pred.shape[1])
        for j in range(num_comparisons):
            # Compute test error for each comparison
            tmp_error = ComputeTestError(pred[i, j], true[i, j])
            error_List.append(tmp_error)
            testID_List.append(count)

            # Vary transparency (alpha) based on comparison position in the case
            alpha_value = (j + 1) / num_comparisons  # Later points are more transparent
            plt.plot(count, tmp_error, 'o', color=color, markersize=4, alpha=alpha_value)
            count += 1

    # Convert lists to numpy arrays
    testID_List = np.asarray(testID_List)
    error_List = np.asarray(error_List)
    avg_error = np.average(error_List)
    
    # Plot the average error line across all cases
    plt.axhline(avg_error, color='red', linestyle='--', linewidth=2, label=f'Average Error: {avg_error:.4f}%')

    # Add labels and title
    plt.xlabel('Samples (color = case, darker = later prediction)', fontsize=12)
    plt.ylabel('Mean Relative Error (%)', fontsize=12)
    plt.title('Accuracy Statistics Plot for Test Dataset', fontsize=14)

    # Add gridlines for better readability
    plt.grid(True)

    # Add a legend for the average error
    plt.legend(loc='upper right', fontsize=8)  # Smaller legend font size
    
    # Save the figure with the specified size and resolution
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, format='png')

    # Display the average error in the terminal
    print(f'Average Error: {avg_error:.4f}%')
    print(f'Max error index: {np.argmax(error_List)}')

# Call the TestErrorPlot function
TestErrorPlot(preds, trues, figsize=(20, 8), dpi=300, out_path="./test_error_plot.png")

# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import imageio

def show_video_gif_multiple_withError(prev, true, pred, vmax=1.0, vmin=0.0, cmap='jet', norm=None, out_path=None, use_rgb=False):
    """Generate gif with a video sequence and plot absolute error along with mean relative error using provided MRE formula."""
    
    def swap_axes(x):
        if len(x.shape) > 3:
            return x.swapaxes(1, 2).swapaxes(2, 3)
        else:
            return x

    prev, true, pred = map(swap_axes, [prev, true, pred])
    prev_frames = prev.shape[0]
    frames = prev_frames + true.shape[0]
    images = []
    
    for i in range(frames):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 9))  # Larger figsize for higher resolution
        for t, ax in enumerate(axes):
            if t == 0:
                plt.text(0.3, 1.05, 'Ground Truth', fontsize=15, color='green', transform=ax.transAxes)
                if i < prev_frames:
                    frame = prev[i]
                else:
                    frame = true[i - prev_frames]
                im = ax.imshow(frame, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label('Pixel Value', fontsize=10)
                
            elif t == 1:
                plt.text(0.2, 1.05, 'Predicted Frames', fontsize=15, color='red', transform=ax.transAxes)
                if i < prev_frames:
                    # frame = prev[i]
                    frame = np.zeros_like(prev[i])
                else:
                    frame = pred[i - prev_frames]
                frame = frame
                im = ax.imshow(frame, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label('Pixel Value', fontsize=10)
                
            elif t == 2:
                plt.text(0.2, 1.05, 'Absolute Error', fontsize=15, color='blue', transform=ax.transAxes)
                if i < prev_frames:
                    # Plot prev - prev (which should result in all zeros)
                    abs_error = np.zeros_like(prev[i])
                    mre = 0.0  # No error, as we are comparing the same frames
                else:
                    # Plot absolute error for the remaining frames
                    abs_error = np.abs(true[i - prev_frames] - pred[i - prev_frames])
                    
                    # Calculate MRE using the provided formula
                    phi_gt = true[i - prev_frames]
                    phi_pred = pred[i - prev_frames]
                    num_pixels = np.prod(phi_gt.shape)
                    squared_error = np.sum((phi_gt - phi_pred) ** 2) / num_pixels
                    phi_gt_max = np.max(phi_gt)
                    phi_gt_min = np.min(phi_gt)
                    mre = np.sqrt(squared_error) / (phi_gt_max - phi_gt_min) * 100

                im = ax.imshow(abs_error, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label('Absolute Error', fontsize=10)
                
                # Use ax.text() to manually place MRE below the third subplot
                ax.text(0.5, -0.1, f'Mean Relative Error: {mre:.4f}%', 
                        fontsize=12, color='blue', ha='center', transform=ax.transAxes)

            ax.axis('off')
        
        # Save the frame to the temporary image and append to images list for GIF creation
        plt.savefig(f'./tmp_frame_{i}.png', bbox_inches='tight', format='png', dpi=300)  # Higher DPI
        images.append(imageio.imread(f'./tmp_frame_{i}.png'))
        plt.close()

    # Remove temporary files after GIF creation
    if out_path is not None:
        if not out_path.endswith('gif'):
            out_path = out_path + '.gif'
        
        # Create GIF using the frames and set it to loop infinitely (loop=0)
        imageio.mimsave(out_path, images, duration=0.1, loop=0)  # loop=0 for infinite looping GIF

    # Optionally, clean up the temporary files after GIF creation
    for i in range(frames):
        os.remove(f'./tmp_frame_{i}.png')


# %%
# # Import the function for generating GIFs
# from openstl.utils import show_video_gif_multiple

for i in range(len(inputs)):
    example_idx = i

    # Modify the output filename to include the random index
    output_gif_filename = f'./prediction_gif/example_{example_idx}.gif'
    # show_video_gif_multiple(inputs[example_idx], trues[example_idx], preds[example_idx], out_path=output_gif_filename)
    show_video_gif_multiple_withError(inputs[example_idx], trues[example_idx], preds[example_idx], out_path=output_gif_filename, cmap='gray')

    print(f"GIF saved as {output_gif_filename}")

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

def show_video_gif_multiple_withError(prev, true, pred, vmax=1.0, vmin=0.0, cmap='jet', norm=None, out_path=None, use_rgb=False):
    """Generate a 3-row figure with ground truth, prediction, and error, handling initial 10 frames in prev and subsequent 20 frames in true/pred."""
    
    def swap_axes(x):
        if len(x.shape) > 3:
            return x.swapaxes(1, 2).swapaxes(2, 3)
        else:
            return x

    prev, true, pred = map(swap_axes, [prev, true, pred])
    prev_frames = prev.shape[0]  # prev contains the first 10 frames
    true_frames = true.shape[0]  # true contains the next 20 frames
    pred_frames = pred.shape[0]  # pred contains the next 20 frames

    # Sample ground truth: 3 frames from 0-9 in prev, 5 frames from 10-29 in true
    sampled_gt_indices_prev = [0, 5, 9]  # Frames 0, 5, 9 from prev
    sampled_gt_indices_true = list(np.linspace(10, 29, 5, dtype=int))  # 5 frames from true (10-29)
    
    # Sample prediction: 5 frames from pred (10-29)
    sampled_pred_indices = list(np.linspace(0, pred_frames - 1, 5, dtype=int))  # Sampling within pred frames (0 to 19 in pred)

    # Ensure output directory exists
    if not os.path.exists('prediction_gif'):
        os.makedirs('prediction_gif')
    
    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(24, 7))  # 3 rows, 8 columns
    fig.subplots_adjust(hspace=0.1, wspace=0.05)  # Reduce the white space between subplots

    for row, ax_row in enumerate(axes):
        if row == 0:  # Ground Truth Row
            for i, ax in enumerate(ax_row):
                if i < 3:
                    # Plot frames from 0-9 from prev (first 3)
                    frame = prev[sampled_gt_indices_prev[i]]
                    ax.imshow(frame, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
                    ax.set_title(f'Input {sampled_gt_indices_prev[i]}', fontsize=10)  # Smaller font size
                else:
                    # Plot 5 sampled frames from 10-29 from true
                    frame = true[sampled_gt_indices_true[i-3] - 10]  # Offset by -10 to match true frames starting from 10
                    ax.imshow(frame, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
                    ax.set_title(f'Ground Truth {sampled_gt_indices_true[i-3]}', fontsize=10)  # Smaller font size
                
                # Remove x and y axis numbers
                ax.set_xticks([])  # Remove x-axis numbers
                ax.set_yticks([])  # Remove y-axis numbers
        
        elif row == 1:  # Prediction Row
            for i, ax in enumerate(ax_row):
                if i < 3:
                    # Leave first 3 subplots empty for predictions (since pred starts at frame 10)
                    ax.axis('off')
                else:
                    # Plot 5 sampled frames from pred (10-29 in true corresponds to 0-19 in pred)
                    frame = pred[sampled_pred_indices[i-3]]
                    ax.imshow(frame, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
                    ax.set_title(f'Prediction {sampled_pred_indices[i-3] + 10}', fontsize=8)  # Smaller font size

                # Remove x and y axis numbers
                ax.set_xticks([])  # Remove x-axis numbers
                ax.set_yticks([])  # Remove y-axis numbers
        
        elif row == 2:  # Error Row
            for i, ax in enumerate(ax_row):
                if i < 3:
                    # Leave first 3 subplots empty for errors (since no prediction exists for these frames)
                    ax.axis('off')
                else:
                    # Plot absolute error for sampled frames from 10-29 in true and pred (0 to 19 in pred)
                    gt_frame = true[sampled_gt_indices_true[i-3] - 10]  # Offset by -10 to match true frames starting from 10
                    pred_frame = pred[sampled_pred_indices[i-3]]
                    abs_error = np.abs(gt_frame - pred_frame)
                    ax.imshow(abs_error, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
                    ax.set_title(f'Absolute Error {sampled_pred_indices[i-3] + 10}', fontsize=10)  # Smaller font size

                    # Calculate MRE (Mean Relative Error)
                    num_pixels = np.prod(gt_frame.shape)
                    squared_error = np.sum((gt_frame - pred_frame) ** 2) / num_pixels
                    phi_gt_max = np.max(gt_frame)
                    phi_gt_min = np.min(gt_frame)
                    mre = np.sqrt(squared_error) / (phi_gt_max - phi_gt_min) * 100

                    # Display MRE below the subplot
                    ax.text(0.5, -0.1, f'MRE: {mre:.4f}%', fontsize=10, color='blue', ha='center', transform=ax.transAxes)  # Smaller text size

                # Remove x and y axis numbers
                ax.set_xticks([])  # Remove x-axis numbers
                ax.set_yticks([])  # Remove y-axis numbers

    plt.show()
    # Save the figure for each case
    if out_path is not None:
        output_path = os.path.join('prediction_gif', out_path + '.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved as {output_path}")
    
    plt.close()


# Example of how to call the modified function:
for i in range(len(inputs)):
    example_idx = i

    # Modify the output filename to include the example index
    output_png_filename = f'example_{example_idx}'
    show_video_gif_multiple_withError(inputs[example_idx], trues[example_idx], preds[example_idx], out_path=output_png_filename)

    print(f"Figure saved as {output_png_filename}.png in prediction_gif folder")


