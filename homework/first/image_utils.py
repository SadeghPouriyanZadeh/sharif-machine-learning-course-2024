import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ImageEditor:
    """
    ImageEditor is a class that provides various image editing operations such as displaying images, downsampling, average pooling, and interpolation.

    Attributes:
        image (numpy.ndarray): The loaded image in RGB format.
        image_gray (numpy.ndarray): The grayscale version of the loaded image.
        reduction_results (dict): A dictionary to store reduced images by different methods.

    Methods:
        __init__: Initializes the ImageEditor with the provided image path.
        downsample: Reduces the image dimensions by skipping pixels (downsampling).
        average_pool: Reduces the image dimensions using average pooling.
        interpolate: Resizes the image using bilinear interpolation.
        display_all_reductions: Displays the original image and all reduced images side by side for comparison using Plotly.
        display_single_reduction: Displays the original image and a single reduced image side by side for comparison using Matplotlib.
        save_image: Saves an image to the specified path.
    """

    def __init__(self, image_path) -> None:
        """
        Initializes the object with an image and its grayscale version.

        Args:
            image_path (str): The file path to the image.
        """

        # check if the image file exists in the given path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found at the given path: {image_path}")

        # check if the image file extension is ending with jpg, jpeg, or png
        if image_path.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            raise ValueError("The file is not an image file")

        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.reduction_results = {}

    def downsample(self, factor: int) -> dict:
        """
        Reduce the image dimensions by skipping pixels (downsampling).

        Args:
            factor (int): Factor by which to reduce dimensions.

        Returns:
            dict: A dictionary containing the reduction method, reduction factor, and the reduced image.
        """
        # Check if the factor is valid
        if factor <= 0:
            raise ValueError("The reduction factor must be greater than 0.")

        # Perform downsampling by skipping pixels based on the factor provided
        reduced_image = self.image[::factor, ::factor]
        reduction_result = {
            "reduction_method": "downsampling",
            "reduction_factor": factor,
            "reduced_image": reduced_image,
        }
        self.reduction_results["downsampling"] = reduction_result
        return reduction_result

    def average_pool(self, factor: int) -> dict:
        """
        Reduce the image dimensions using average pooling.

        Args:
            factor (int): Factor by which to reduce dimensions.

        Returns:
            dict: A dictionary containing the reduction method, reduction factor, and the reduced image.
        """
        # Check if the factor is valid
        if factor <= 0:
            raise ValueError("The reduction factor must be greater than 0.")

        original_height, original_width, channels = self.image.shape

        # Calculate the dimensions of the reduced image
        new_height = original_height // factor
        new_width = original_width // factor

        # Initialize the reduced image
        reduced_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        # Perform average pooling
        for i in range(new_height):
            for j in range(new_width):
                for c in range(channels):
                    # Calculate the average value of the pixels in the pooling window
                    pooling_window = self.image[
                        i * factor : (i + 1) * factor, j * factor : (j + 1) * factor, c
                    ]
                    reduced_image[i, j, c] = np.mean(pooling_window)

        # Clip the values to ensure they are within the valid range [0, 255]
        reduced_image = np.clip(reduced_image, 0, 255).astype(np.uint8)

        reduction_result = {
            "reduction_method": "average_pooling",
            "reduction_factor": factor,
            "reduced_image": reduced_image,
        }
        self.reduction_results["average_pooling"] = reduction_result
        return reduction_result

    def interpolate(self, new_width: int, new_height: int) -> dict:
        """
        Resize an image using bilinear interpolation.
        Parameters:
        image (numpy.ndarray): The input image to be resized.
                               It should be a 3D array with shape (height, width, channels).
        new_width (int): The desired width of the resized image.
        new_height (int): The desired height of the resized image.
        Returns:
        numpy.ndarray: The resized image with shape (new_height, new_width, channels).
        The function performs bilinear interpolation to resize the input image to the specified dimensions.
        It calculates the corresponding pixel values in the original image and uses the surrounding pixels
        to compute the new pixel values in the resized image.
        """

        original_height, original_width, channels = self.image.shape

        # Create an output image with the new dimensions
        resized_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        # Calculate the scaling factors
        x_scale = original_width / new_width
        y_scale = original_height / new_height

        # Apply bilinear interpolation
        for i in range(new_height):
            for j in range(new_width):
                # Find the corresponding pixel in the original image
                x = j * x_scale
                y = i * y_scale

                # Calculate the coordinates of the 4 surrounding pixels
                x0 = int(np.floor(x))
                x1 = min(x0 + 1, original_width - 1)
                y0 = int(np.floor(y))
                y1 = min(y0 + 1, original_height - 1)

                # Calculate the weights
                dx = x - x0
                dy = y - y0

                # Perform bilinear interpolation
                for c in range(channels):
                    pixel_value = (
                        (1 - dx) * (1 - dy) * self.image[y0, x0, c]
                        + dx * (1 - dy) * self.image[y0, x1, c]
                        + (1 - dx) * dy * self.image[y1, x0, c]
                        + dx * dy * self.image[y1, x1, c]
                    )
                    resized_image[i, j, c] = np.clip(pixel_value, 0, 255)

        reduction_result = {
            "reduction_method": "interpolation",
            "reduction_factor": (new_width, new_height),
            "reduced_image": resized_image,
        }
        self.reduction_results["interpolation"] = reduction_result
        return reduction_result

    def display_single_reduction(self, reduction_result: dict) -> None:
        """
        Display the original image and a single reduced image side by side for comparison using Matplotlib.

        Args:
            reduction_result (dict): Dictionary containing the reduced image and reduction details.

        Raises:
            ValueError: If no reduced images are found or if the original image is not loaded.
        """
        # Check if reduced images are available
        if not hasattr(self, "reduction_results"):
            raise ValueError(
                "No reduced images found. Please run a reduction method first."
            )
        # Check if original image is available
        if not hasattr(self, "image"):
            raise ValueError(
                "No original image found. Please run a reduction method first."
            )

        # Create a side-by-side comparison of the original and reduced images
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Display the original image
        axes[0].imshow(self.image)
        axes[0].set_title("Original Image\nImage Shape: " f"{self.image.shape}\n")
        axes[0].axis("off")

        # Display the reduced image
        axes[1].imshow(reduction_result["reduced_image"])
        axes[1].set_title(
            f"Reduced Image:\n"
            f"{reduction_result['reduction_method'].capitalize()}\n"
            f"Reduction Factor: {reduction_result['reduction_factor']}\n"
            f"Image Shape: {reduction_result['reduced_image'].shape}"
        )
        axes[1].axis("off")

        plt.show()

    def display_all_reductions(self) -> None:
        """
        Display the original image and all reduced images side by side for comparison using Plotly.

        This method creates a subplot with the original image and all the reduced images
        (downsampled, average pooled, and interpolated) for visual comparison. It uses Plotly
        to create an interactive visualization.

        Raises:
            ValueError: If no reduced images are found or if the original image is not loaded.
        """
        # Check if reduced images are available
        if not self.reduction_results:
            raise ValueError(
                "No reduced images found. Please run reduction methods first."
            )
        # Check if original image is available
        if not hasattr(self, "image"):
            raise ValueError("No original image found. Please load an image first.")

        fig = make_subplots(
            rows=1,
            cols=4,
            subplot_titles=(
                "Original Image",
                "Downsampled Image",
                "Average Pooled Image",
                "Interpolated Image",
            ),
        )

        # Add original image
        fig.add_trace(go.Image(z=self.image), row=1, col=1)

        # Add downsampled image
        if "downsampling" in self.reduction_results:
            fig.add_trace(
                go.Image(z=self.reduction_results["downsampling"]["reduced_image"]),
                row=1,
                col=2,
            )

        # Add average pooled image
        if "average_pooling" in self.reduction_results:
            fig.add_trace(
                go.Image(z=self.reduction_results["average_pooling"]["reduced_image"]),
                row=1,
                col=3,
            )

        # Add interpolated image
        if "interpolation" in self.reduction_results:
            fig.add_trace(
                go.Image(z=self.reduction_results["interpolation"]["reduced_image"]),
                row=1,
                col=4,
            )

        # Update layout
        fig.update_layout(
            title_text="Comparison of Original and Reduced Images",
            title_x=0.5,
            showlegend=False,
        )

        fig.show()

    def save_image(self, image: np.ndarray, path: str) -> None:
        """
        Save an image to the specified path.

        Args:
            image (ndarray): The image to save.
            path (str): The path to save the image.
        """
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Image saved to {path}")
