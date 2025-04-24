"""
Image preprocessing functionality for the ML Platform.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import io
import base64

def resize_image(image: Union[Image.Image, np.ndarray], size: Tuple[int, int], 
                keep_aspect_ratio: bool = True) -> Image.Image:
    """
    Resize an image to the specified size.
    
    Args:
        image: PIL Image or numpy array
        size: Target size (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if keep_aspect_ratio:
        image = ImageOps.contain(image, size)
    else:
        image = image.resize(size, Image.LANCZOS)
    
    return image

def normalize_image(image: Union[Image.Image, np.ndarray], 
                   mean: Optional[List[float]] = None, 
                   std: Optional[List[float]] = None) -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        image: PIL Image or numpy array
        mean: Channel-wise mean values (default: [0.485, 0.456, 0.406] for RGB)
        std: Channel-wise standard deviations (default: [0.229, 0.224, 0.225] for RGB)
        
    Returns:
        Normalized numpy array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to float32 and scale to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Use ImageNet defaults if mean and std are not provided
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    # Ensure mean and std have the right shape for broadcasting
    mean_array = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std_array = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    
    # Apply normalization
    normalized_image = (image - mean_array) / std_array
    
    return normalized_image

def augment_image(image: Image.Image, augmentations: List[str], 
                 parameters: Optional[Dict[str, Any]] = None) -> Image.Image:
    """
    Apply a series of augmentations to an image.
    
    Args:
        image: PIL Image to augment
        augmentations: List of augmentation types to apply
        parameters: Dictionary of parameters for each augmentation
        
    Returns:
        Augmented PIL Image
    """
    parameters = parameters or {}
    result_image = image.copy()
    
    for aug in augmentations:
        if aug == 'rotate':
            angle = parameters.get('rotate_angle', 30)
            result_image = result_image.rotate(angle, Image.BICUBIC, expand=True)
        
        elif aug == 'flip_horizontal':
            result_image = ImageOps.mirror(result_image)
        
        elif aug == 'flip_vertical':
            result_image = ImageOps.flip(result_image)
        
        elif aug == 'crop':
            left = parameters.get('crop_left', 0)
            top = parameters.get('crop_top', 0)
            right = parameters.get('crop_right', result_image.width)
            bottom = parameters.get('crop_bottom', result_image.height)
            result_image = result_image.crop((left, top, right, bottom))
        
        elif aug == 'brightness':
            factor = parameters.get('brightness_factor', 1.5)
            enhancer = ImageEnhance.Brightness(result_image)
            result_image = enhancer.enhance(factor)
        
        elif aug == 'contrast':
            factor = parameters.get('contrast_factor', 1.5)
            enhancer = ImageEnhance.Contrast(result_image)
            result_image = enhancer.enhance(factor)
        
        elif aug == 'saturation':
            factor = parameters.get('saturation_factor', 1.5)
            enhancer = ImageEnhance.Color(result_image)
            result_image = enhancer.enhance(factor)
        
        elif aug == 'blur':
            radius = parameters.get('blur_radius', 2)
            result_image = result_image.filter(ImageFilter.GaussianBlur(radius))
        
        elif aug == 'sharpen':
            result_image = result_image.filter(ImageFilter.SHARPEN)
        
    return result_image

def image_to_tensor(image: Union[Image.Image, np.ndarray], 
                   expand_dims: bool = True) -> np.ndarray:
    """
    Convert an image to a tensor format suitable for ML models.
    
    Args:
        image: PIL Image or numpy array
        expand_dims: Whether to add batch dimension
        
    Returns:
        Numpy array in tensor format
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Handle grayscale images
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    
    # Ensure channels-last format (H, W, C)
    if image.shape[-1] not in [1, 3, 4]:
        # Assume channels-first format
        image = np.transpose(image, (1, 2, 0))
    
    # Add batch dimension if needed
    if expand_dims:
        image = np.expand_dims(image, axis=0)
    
    return image

def decode_image_data(encoded_data: str) -> Image.Image:
    """
    Decode base64 encoded image data.
    
    Args:
        encoded_data: Base64 encoded image data
        
    Returns:
        PIL Image
    """
    # Remove data URL prefix if present
    if ';base64,' in encoded_data:
        encoded_data = encoded_data.split(';base64,')[1]
    
    # Decode base64 data
    image_data = base64.b64decode(encoded_data)
    
    # Open image from binary data
    image = Image.open(io.BytesIO(image_data))
    
    return image

def encode_image_data(image: Image.Image, format: str = 'PNG') -> str:
    """
    Encode an image as base64 data.
    
    Args:
        image: PIL Image to encode
        format: Image format to use
        
    Returns:
        Base64 encoded image data
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    
    # Encode as base64
    encoded_data = base64.b64encode(buffer.getvalue()).decode('ascii')
    
    # Create data URL
    mime_type = f"image/{format.lower()}"
    data_url = f"data:{mime_type};base64,{encoded_data}"
    
    return data_url
