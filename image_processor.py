"""
CS 5001
Spring 2025
Final Project – Milestone 2: Object-Oriented Image Processing & Basic Filters
This file contains the ImageProcessor class, which provides methods for processing images.
Author: 
Ziming "Alan" Yi
"""


import cv2
import os  # Import os module for file and directory operations
import numpy as np


class ImageProcessor:
    # List of supported image file extensions
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    def __init__(self, image_path):
        """
        Initialize the ImageProcessor with an image file
        
        Args:
            image_path (str): Path to the image file
            
        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the image cannot be loaded or is invalid
            TypeError: If the file format is not supported
        """
        # os.path.exists(): Check if a file or directory exists at the specified path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # os.path.isfile(): Check if the path points to a file (not a directory)
        if not os.path.isfile(image_path):
            raise ValueError(f"Path is not a file: {image_path}")
        
        # Check if the file has a supported extension
        _, file_extension = os.path.splitext(image_path)
        if file_extension.lower() not in self.SUPPORTED_FORMATS:
            raise TypeError(f"Unsupported file format: {file_extension}. Supported formats are: {', '.join(self.SUPPORTED_FORMATS)}")
            
        self.image_path = image_path
        # cv2.imread(): Reads an image from a file into a numpy array
        # Returns None if the image cannot be loaded
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}. The file may be corrupted or in an unsupported format.")
            
        if self.image.size == 0:
            raise ValueError("Loaded image is empty")
    
    def grayscale_conversion(self):
        """
        convert image to grayscale

        Returns:
            numpy.ndarray: grayscale image

        Raises:
            ValueError: If the image is already grayscale
        """
        # Check if the image is already grayscale (2D array)
        if len(self.image.shape) == 2:
            raise ValueError("Image is already in grayscale")

        # Check if the image is effectively grayscale (3 identical channels)
        # This happens when a grayscale image is loaded by cv2.imread as BGR
        bgr_channels = cv2.split(self.image)
        if np.array_equal(bgr_channels[0], bgr_channels[1]) and np.array_equal(bgr_channels[1], bgr_channels[2]):
            raise ValueError("Image is already in grayscale")

        # cv2.cvtColor(): Converts an image from one color space to another
        # COLOR_BGR2GRAY: Converts BGR color space to grayscale
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def adjust_brightness(self, factor):
        """
        adjust image brightness

        Args:
            factor (float): brightness adjustment factor, greater than 1 increases brightness, less than 1 decreases brightness

        Returns:
            numpy.ndarray: adjusted image

        Raises:
            ValueError: If factor is not a positive number
        """
        if not isinstance(factor, (int, float)):
            raise ValueError("Brightness factor must be a number")

        if factor <= 0:
            raise ValueError("Brightness factor must be positive")

        # cv2.convertScaleAbs(): Scales, calculates absolute values, and converts the result to 8-bit
        # alpha: scale factor (brightness multiplier)
        # beta: delta added to the scaled values (0 means no additional brightness)
        return cv2.convertScaleAbs(self.image, alpha=factor, beta=0)

    def flip_image(self, direction='horizontal'):
        """
        flip image

        Args:
            direction (str): 'horizontal' or 'vertical'

        Returns:
            numpy.ndarray: flipped image

        Raises:
            ValueError: If direction is invalid
        """
        if not isinstance(direction, str):
            raise ValueError("Direction must be a string")

        direction = direction.lower()
        if direction not in ['horizontal', 'vertical']:
            raise ValueError("Direction must be 'horizontal' or 'vertical'")

        # cv2.flip(): Flips a 2D array around vertical, horizontal, or both axes
        # 1: flip around y-axis (horizontal flip)
        # 0: flip around x-axis (vertical flip)
        if direction == 'horizontal':
            return cv2.flip(self.image, 1)
        elif direction == 'vertical':
            return cv2.flip(self.image, 0)

    def gaussian_blur(self, kernel_size=5):
        """
        Apply Gaussian blur to the image

        Args:
            kernel_size (int): Size of the Gaussian kernel (must be odd)

        Returns:
            numpy.ndarray: blurred image

        Raises:
            ValueError: If kernel_size is not a positive odd number
        """
        if not isinstance(kernel_size, int):
            raise ValueError("Kernel size must be an integer")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd number")

        return cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)

    def cartoonize(self):
        """
        Apply cartoon effect to the image

        Returns:
            numpy.ndarray: cartoonized image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply median blur to reduce noise
        gray = cv2.medianBlur(gray, 5)
        
        # Detect edges using adaptive threshold
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 9, 9)
        
        # Apply bilateral filter to smooth the image while preserving edges
        color = cv2.bilateralFilter(self.image, 9, 300, 300)
        
        # Combine the edges with the color image
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        return cartoon

    def sepia(self):
        """
        Apply sepia filter to the image

        Returns:
            numpy.ndarray: sepia-toned image
        """
        # Create sepia matrix
        sepia_matrix = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
        
        # Apply sepia filter
        sepia_image = cv2.transform(self.image, sepia_matrix)
        
        # Clip values to [0, 255]
        sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
        
        return sepia_image
