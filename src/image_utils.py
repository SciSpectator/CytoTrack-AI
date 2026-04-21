"""
Cell Tracker - Image Processing Utilities
==========================================
Brightness, contrast, gamma, and color filter adjustments.
"""

import cv2
import numpy as np
from typing import Tuple


def apply_brightness_contrast(image: np.ndarray, brightness: int = 0, 
                              contrast: float = 1.0) -> np.ndarray:
    """
    Apply brightness and contrast adjustments.
    
    Args:
        image: Input BGR image
        brightness: Brightness offset (-100 to 100)
        contrast: Contrast multiplier (0.5 to 3.0)
        
    Returns:
        Adjusted image
    """
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def apply_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction.
    
    Args:
        image: Input image
        gamma: Gamma value (0.1 to 3.0, 1.0 = no change)
        
    Returns:
        Gamma corrected image
    """
    if gamma == 1.0:
        return image
    
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_filter(image: np.ndarray, filter_mode: int = 0) -> np.ndarray:
    """
    Apply color filter to image.
    
    Args:
        image: Input BGR image
        filter_mode: Filter type
            0 = None (original)
            1 = Grayscale
            2 = Sepia
            3 = False Color (Jet colormap)
            4 = Invert
            5 = CLAHE (Contrast Limited Adaptive Histogram Equalization)
            6 = Edge Enhancement
            
    Returns:
        Filtered image
    """
    if filter_mode == 0:
        return image
    
    elif filter_mode == 1:  # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    elif filter_mode == 2:  # Sepia
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia = cv2.transform(image, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    
    elif filter_mode == 3:  # False Color (Jet)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    elif filter_mode == 4:  # Invert
        return cv2.bitwise_not(image)
    
    elif filter_mode == 5:  # CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    elif filter_mode == 6:  # Edge Enhancement
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    return image


def apply_all_adjustments(image: np.ndarray, brightness: int = 0,
                         contrast: float = 1.0, gamma: float = 1.0,
                         filter_mode: int = 0) -> np.ndarray:
    """
    Apply all image adjustments in sequence.
    
    Args:
        image: Input BGR image
        brightness: Brightness offset (-100 to 100)
        contrast: Contrast multiplier (0.5 to 3.0)
        gamma: Gamma value (0.1 to 3.0)
        filter_mode: Color filter (0-6)
        
    Returns:
        Processed image
    """
    # Apply brightness and contrast
    result = apply_brightness_contrast(image, brightness, contrast)
    
    # Apply gamma
    result = apply_gamma(result, gamma)
    
    # Apply color filter
    result = apply_filter(result, filter_mode)
    
    return result


# Filter names for display
FILTER_NAMES = [
    "0: None",
    "1: Grayscale",
    "2: Sepia",
    "3: False Color (Jet)",
    "4: Invert",
    "5: CLAHE (Enhance)",
    "6: Edge Enhancement"
]


class ImageSettings:
    """Store and manage image display settings."""
    
    def __init__(self):
        self.brightness = 0      # -100 to 100
        self.contrast = 1.0      # 0.5 to 3.0
        self.gamma = 1.0         # 0.1 to 3.0
        self.filter_mode = 0     # 0 to 6
    
    def reset(self):
        """Reset all settings to defaults."""
        self.brightness = 0
        self.contrast = 1.0
        self.gamma = 1.0
        self.filter_mode = 0
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply current settings to an image."""
        return apply_all_adjustments(
            image,
            self.brightness,
            self.contrast,
            self.gamma,
            self.filter_mode
        )
    
    def increase_brightness(self, amount: int = 10):
        """Increase brightness."""
        self.brightness = min(100, self.brightness + amount)
    
    def decrease_brightness(self, amount: int = 10):
        """Decrease brightness."""
        self.brightness = max(-100, self.brightness - amount)
    
    def increase_contrast(self, amount: float = 0.1):
        """Increase contrast."""
        self.contrast = min(3.0, self.contrast + amount)
    
    def decrease_contrast(self, amount: float = 0.1):
        """Decrease contrast."""
        self.contrast = max(0.5, self.contrast - amount)
    
    def increase_gamma(self, amount: float = 0.1):
        """Increase gamma."""
        self.gamma = min(3.0, self.gamma + amount)
    
    def decrease_gamma(self, amount: float = 0.1):
        """Decrease gamma."""
        self.gamma = max(0.1, self.gamma - amount)
    
    def next_filter(self):
        """Switch to next filter."""
        self.filter_mode = (self.filter_mode + 1) % len(FILTER_NAMES)
    
    def prev_filter(self):
        """Switch to previous filter."""
        self.filter_mode = (self.filter_mode - 1) % len(FILTER_NAMES)
    
    def get_info_string(self) -> str:
        """Get settings info string for display."""
        return (f"Brightness: {self.brightness:+d} | "
                f"Contrast: {self.contrast:.1f} | "
                f"Gamma: {self.gamma:.1f} | "
                f"Filter: {FILTER_NAMES[self.filter_mode]}")
