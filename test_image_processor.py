"""
CS 5001
Spring 2025
Final Project – Milestone 2: Object-Oriented Image Processing & Basic Filters

This file contains unit tests for the ImageProcessor class using pytest.

Author: 
Ziming "Alan" Yi
"""

import pytest
import os
import cv2
import numpy as np
from image_processor import ImageProcessor


def valid_image_path():
    """Fixture to provide a valid test image path"""
    # Create a simple test image if it doesn't exist
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        # Create a simple 100x100 RGB image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some color to make it non-grayscale
        img[30:70, 30:70] = [0, 0, 255]  # Red square
        cv2.imwrite(test_image_path, img)
    return test_image_path

valid_image_path = pytest.fixture(valid_image_path)


def grayscale_image_path():
    """Fixture to provide a grayscale test image path"""
    test_image_path = "test_gray_image.jpg"
    if not os.path.exists(test_image_path):
        # Create a simple 100x100 grayscale image
        img = np.zeros((100, 100), dtype=np.uint8)
        # Add some gray values
        img[30:70, 30:70] = 128  # Gray square
        cv2.imwrite(test_image_path, img)
    return test_image_path

grayscale_image_path = pytest.fixture(grayscale_image_path)


def test_init_with_valid_image(valid_image_path):
    """Test initialization with a valid image path"""
    processor = ImageProcessor(valid_image_path)
    assert processor.image_path == valid_image_path
    assert processor.image is not None
    assert processor.image.shape[2] == 3  # Should be a color image


def test_init_with_nonexistent_image():
    """Test initialization with a non-existent image path"""
    with pytest.raises(FileNotFoundError):
        ImageProcessor("nonexistent_image.jpg")


def test_init_with_invalid_path():
    """Test initialization with an invalid path (directory)"""
    # Create a temporary directory
    os.makedirs("test_dir", exist_ok=True)
    with pytest.raises(ValueError):
        ImageProcessor("test_dir")
    os.rmdir("test_dir")


def test_init_with_unsupported_format():
    """Test initialization with an unsupported file format"""
    # Create a temporary text file
    test_file = "test_file.txt"
    with open(test_file, "w") as f:
        f.write("This is not an image file")
    
    try:
        with pytest.raises(TypeError, match="Unsupported file format"):
            ImageProcessor(test_file)
    finally:
        # Clean up the test file
        if os.path.exists(test_file):
            os.remove(test_file)


def test_grayscale_conversion(valid_image_path):
    """Test grayscale conversion with a color image"""
    processor = ImageProcessor(valid_image_path)
    gray_image = processor.grayscale_conversion()
    assert len(gray_image.shape) == 2  # Grayscale image has 2 dimensions


def test_grayscale_already_grayscale(grayscale_image_path):
    """Test grayscale conversion with an already grayscale image"""
    processor = ImageProcessor(grayscale_image_path)
    with pytest.raises(ValueError, match="Image is already in grayscale"):
        processor.grayscale_conversion()


# Test brightness adjustment
def test_brightness_increase(valid_image_path):
    """Test increasing brightness"""
    processor = ImageProcessor(valid_image_path)
    original_mean = np.mean(processor.image)
    brightened = processor.adjust_brightness(1.5)
    assert np.mean(brightened) > original_mean


def test_brightness_decrease(valid_image_path):
    """Test decreasing brightness"""
    processor = ImageProcessor(valid_image_path)
    original_mean = np.mean(processor.image)
    darkened = processor.adjust_brightness(0.5)
    assert np.mean(darkened) < original_mean


def test_brightness_invalid_factor(valid_image_path):
    """Test brightness adjustment with invalid factor"""
    processor = ImageProcessor(valid_image_path)
    with pytest.raises(ValueError, match="Brightness factor must be positive"):
        processor.adjust_brightness(-1)

    with pytest.raises(ValueError, match="Brightness factor must be a number"):
        processor.adjust_brightness("invalid")


# Test image flipping
def test_horizontal_flip(valid_image_path):
    """Test horizontal flipping"""
    processor = ImageProcessor(valid_image_path)
    # Create a reference point in the image
    x, y = 30, 50
    color = processor.image[y, x].copy()

    flipped = processor.flip_image("horizontal")

    # After horizontal flip, the reference point should be at (width-x-1, y)
    width = processor.image.shape[1]
    flipped_x = width - x - 1
    assert np.array_equal(flipped[y, flipped_x], color)


def test_vertical_flip(valid_image_path):
    """Test vertical flipping"""
    processor = ImageProcessor(valid_image_path)
    # Create a reference point in the image
    x, y = 30, 50
    color = processor.image[y, x].copy()

    flipped = processor.flip_image("vertical")

    # After vertical flip, the reference point should be at (x, height-y-1)
    height = processor.image.shape[0]
    flipped_y = height - y - 1
    assert np.array_equal(flipped[flipped_y, x], color)


def test_flip_invalid_direction(valid_image_path):
    """Test flipping with invalid direction"""
    processor = ImageProcessor(valid_image_path)
    with pytest.raises(ValueError, match="Direction must be 'horizontal' or 'vertical'"):
        processor.flip_image("diagonal")

    with pytest.raises(ValueError, match="Direction must be a string"):
        processor.flip_image(123)


# Cleanup test files after all tests
def teardown_module(module):
    """Remove test files after all tests are done"""
    test_files = ["test_image.jpg", "test_gray_image.jpg"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)


def test_gaussian_blur(self):
    """Test Gaussian blur filter"""
    # Test with default kernel size
    blurred = self.processor.gaussian_blur()
    self.assertEqual(blurred.shape, self.processor.image.shape)
    
    # Test with custom kernel size
    blurred = self.processor.gaussian_blur(7)
    self.assertEqual(blurred.shape, self.processor.image.shape)
    
    # Test invalid kernel size
    with self.assertRaises(ValueError):
        self.processor.gaussian_blur(4)  # Even number
    with self.assertRaises(ValueError):
        self.processor.gaussian_blur(0)  # Zero
    with self.assertRaises(ValueError):
        self.processor.gaussian_blur(-1)  # Negative


def test_cartoonize(self):
    """Test cartoonize filter"""
    cartoon = self.processor.cartoonize()
    self.assertEqual(cartoon.shape, self.processor.image.shape)
    
    # Test with grayscale image
    gray_processor = ImageProcessor(self.gray_image_path)
    cartoon = gray_processor.cartoonize()
    self.assertEqual(cartoon.shape, gray_processor.image.shape)


def test_sepia(self):
    """Test sepia filter"""
    sepia = self.processor.sepia()
    self.assertEqual(sepia.shape, self.processor.image.shape)
    
    # Test with grayscale image
    gray_processor = ImageProcessor(self.gray_image_path)
    sepia = gray_processor.sepia()
    self.assertEqual(sepia.shape, gray_processor.image.shape)
