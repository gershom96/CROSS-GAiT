import  matplotlib.pyplot as plt
import numpy as np

def display_images(image_data, image_mean, image_std):
    """
    Reconstruct and display a single image.
    
    Args:
        image_data (numpy array): Image data with shape (C, H, W).
        image_mean (list): Mean used for normalization for each channel.
        image_std (list): Standard deviation used for normalization for each channel.
    """
    # Convert to numpy array if not already
    image_data = np.array(image_data)
    image_mean = np.array(image_mean)
    image_std = np.array(image_std)
    # Re-scale the image data by multiplying by the std and adding the mean
    image_data = (image_data * image_std[:, None, None]) + image_mean[:, None, None]

    # Clip the values to the valid range [0, 1]
    image_data = np.clip(image_data, 0, 1)

    # Convert the image from (C, H, W) to (H, W, C)
    image_data = image_data.transpose(1, 2, 0)

    # Convert to uint8 format (0-255) for displaying
    # image_data = (image_data * 255).astype(np.uint8)

    # Display the image
    plt.figure(figsize=(5, 5))
    plt.imshow(image_data)
    plt.axis('off')
    plt.show()

def plot_curves(time_series, time_series_mean, time_series_std):
    """
    Plot 18 time series curves on the same plot.
    
    Args:
        time_series (numpy array): Time series data with shape (18, 100).
    """
    plt.figure(figsize=(10, 6))
    
    for i in range(time_series.shape[0]):
        plt.plot(((time_series[i]*time_series_std[i]) + time_series_mean[i]), label=f'Channel {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data (18 Channels)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
    plt.show()