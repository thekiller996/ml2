import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from typing import List, Dict, Any, Union, Tuple
import os

class ImagePreprocessor:
    """Class for image preprocessing operations"""
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize image to specified dimensions."""
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to range [0,1]."""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def apply_image_filters(image: np.ndarray, filters: List[str], params: Dict[str, Any] = None) -> np.ndarray:
        """Apply specified filters to an image.
        
        Args:
            image: Input image as numpy array
            filters: List of filter names to apply ('blur', 'sharpen', 'edge', etc.)
            params: Dictionary of parameters for filters
            
        Returns:
            Processed image
        """
        if params is None:
            params = {}
            
        pil_image = Image.fromarray(np.uint8(image))
        
        for filter_name in filters:
            if filter_name == 'blur':
                radius = params.get('blur_radius', 2)
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
            
            elif filter_name == 'sharpen':
                factor = params.get('sharpen_factor', 2.0)
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(factor)
            
            elif filter_name == 'edge':
                pil_image = pil_image.filter(ImageFilter.FIND_EDGES)
            
            elif filter_name == 'emboss':
                pil_image = pil_image.filter(ImageFilter.EMBOSS)
            
            elif filter_name == 'contour':
                pil_image = pil_image.filter(ImageFilter.CONTOUR)
            
            elif filter_name == 'brightness':
                factor = params.get('brightness_factor', 1.5)
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(factor)
            
            elif filter_name == 'contrast':
                factor = params.get('contrast_factor', 1.5)
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(factor)
                
        return np.array(pil_image)
    
    @staticmethod
    def create_augmentation_pipeline(augmentations: List[str], params: Dict[str, Any] = None) -> A.Compose:
        """Create an image augmentation pipeline.
        
        Args:
            augmentations: List of augmentation names
            params: Dictionary of parameters for augmentations
            
        Returns:
            Albumentations Compose object with the specified augmentations
        """
        if params is None:
            params = {}
            
        aug_list = []
        
        for aug_name in augmentations:
            if aug_name == 'horizontal_flip':
                aug_list.append(A.HorizontalFlip(p=params.get('hflip_prob', 0.5)))
                
            elif aug_name == 'vertical_flip':
                aug_list.append(A.VerticalFlip(p=params.get('vflip_prob', 0.5)))
                
            elif aug_name == 'rotate':
                limit = params.get('rotate_limit', 45)
                aug_list.append(A.Rotate(limit=limit, p=params.get('rotate_prob', 0.5)))
                
            elif aug_name == 'shift':
                shift_limit = params.get('shift_limit', 0.1)
                aug_list.append(A.ShiftScaleRotate(
                    shift_limit=shift_limit,
                    scale_limit=0,
                    rotate_limit=0,
                    p=params.get('shift_prob', 0.5)
                ))
                
            elif aug_name == 'scale':
                scale_limit = params.get('scale_limit', 0.2)
                aug_list.append(A.ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=scale_limit,
                    rotate_limit=0,
                    p=params.get('scale_prob', 0.5)
                ))
                
            elif aug_name == 'blur':
                blur_limit = params.get('blur_limit', 7)
                aug_list.append(A.Blur(blur_limit=blur_limit, p=params.get('blur_prob', 0.5)))
                
            elif aug_name == 'brightness_contrast':
                brightness = params.get('brightness_limit', 0.2)
                contrast = params.get('contrast_limit', 0.2)
                aug_list.append(A.RandomBrightnessContrast(
                    brightness_limit=brightness,
                    contrast_limit=contrast,
                    p=params.get('bc_prob', 0.5)
                ))
                
            elif aug_name == 'noise':
                aug_list.append(A.GaussNoise(
                    var_limit=params.get('noise_var', (10.0, 50.0)),
                    p=params.get('noise_prob', 0.5)
                ))
                
            elif aug_name == 'cutout':
                aug_list.append(A.Cutout(
                    num_holes=params.get('cutout_holes', 8),
                    max_h_size=params.get('cutout_height', 8),
                    max_w_size=params.get('cutout_width', 8),
                    p=params.get('cutout_prob', 0.5)
                ))
        
        return A.Compose(aug_list)
    
    @staticmethod
    def apply_augmentations(images: List[np.ndarray], pipeline: A.Compose, times: int = 1) -> List[np.ndarray]:
        """Apply augmentation pipeline to a list of images.
        
        Args:
            images: List of images to augment
            pipeline: Albumentations Compose pipeline
            times: Number of augmented versions to create per original image
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for image in images:
            augmented_images.append(image)  # Keep the original
            
            for _ in range(times):
                augmented = pipeline(image=image)['image']
                augmented_images.append(augmented)
                
        return augmented_images
    
    @staticmethod
    def extract_image_features(image: np.ndarray, method: str) -> np.ndarray:
        """Extract features from an image using various methods.
        
        Args:
            image: Input image
            method: Feature extraction method ('histogram', 'hog', 'orb')
            
        Returns:
            Feature vector
        """
        if method == 'histogram':
            hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
            return np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
            
        elif method == 'hog':
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Resize for consistency
            gray = cv2.resize(gray, (128, 128))
            
            # Calculate HOG features
            win_size = (128, 128)
            block_size = (16, 16)
            block_stride = (8, 8)
            cell_size = (8, 8)
            nbins = 9
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            features = hog.compute(gray)
            return features.flatten()
            
        elif method == 'orb':
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Create ORB detector
            orb = cv2.ORB_create(nfeatures=100)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if descriptors is None:
                return np.zeros(3200)  # Return zeros if no features detected
                
            # Ensure consistent size by padding or truncating
            max_size = 3200
            if descriptors.size > max_size:
                return descriptors.flatten()[:max_size]
            else:
                return np.pad(descriptors.flatten(), (0, max_size - descriptors.size), 'constant')
        
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
    
    @staticmethod
    def batch_process_images(image_dir: str, operations: List[Dict[str, Any]], 
                             output_dir: str = None) -> List[np.ndarray]:
        """Process a batch of images with a sequence of operations.
        
        Args:
            image_dir: Directory containing images
            operations: List of operations to apply, each a dict with 'type' and params
            output_dir: Directory to save processed images (optional)
            
        Returns:
            List of processed images
        """
        processed_images = []
        
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get list of image files
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            
            # Apply each operation in sequence
            for op in operations:
                op_type = op['type']
                
                if op_type == 'resize':
                    image = ImagePreprocessor.resize_image(
                        image, op.get('width', 224), op.get('height', 224))
                    
                elif op_type == 'normalize':
                    image = ImagePreprocessor.normalize_image(image)
                    
                elif op_type == 'filters':
                    image = ImagePreprocessor.apply_image_filters(
                        image, op.get('filters', []), op.get('params', {}))
                    
                elif op_type == 'augment':
                    # For single image augmentation
                    pipeline = ImagePreprocessor.create_augmentation_pipeline(
                        op.get('augmentations', []), op.get('params', {}))
                    augmented = pipeline(image=image)['image']
                    image = augmented
            
            processed_images.append(image)
            
            # Save if output directory specified
            if output_dir:
                output_path = os.path.join(output_dir, f"processed_{img_file}")
                # Convert back to BGR for saving with OpenCV
                save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, save_img)
                
        return processed_images

# Helper function to load and preprocess image dataset
def load_image_dataset(image_dir: str, target_size: Tuple[int, int] = (224, 224), 
                       normalize: bool = True) -> List[np.ndarray]:
    """Load images from a directory with basic preprocessing.
    
    Args:
        image_dir: Directory containing images
        target_size: Size to resize images to
        normalize: Whether to normalize pixel values
        
    Returns:
        List of preprocessed images
    """
    images = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Resize
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize if requested
        if normalize:
            image = image.astype(np.float32) / 255.0
            
        images.append(image)
        
    return images