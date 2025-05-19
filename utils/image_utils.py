# Image processing utilities for Agent_4o

import io
import base64
import numpy as np
import streamlit as st
from PIL import Image, ExifTags

import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

# Try to import optional dependencies with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    st.warning("OpenCV (cv2) not found. Some image processing features will be disabled.")
    CV2_AVAILABLE = False

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    st.warning("imagehash library not found. Duplicate detection will be disabled.")
    IMAGEHASH_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    st.warning("scikit-learn not found. Color analysis features will be disabled.")
    SKLEARN_AVAILABLE = False

def resize_image(img, max_size=800):
    """
    Resize an image while maintaining aspect ratio
    
    Args:
        img: PIL Image object
        max_size: Maximum width or height
        
    Returns:
        Resized PIL Image
    """
    width, height = img.size
    
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int(height * (max_size / width))
    else:
        if height > max_size:
            new_height = max_size
            new_width = int(width * (max_size / height))
    
    if width > max_size or height > max_size:
        try:
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except AttributeError:
            # For older PIL versions
            return img.resize((new_width, new_height), Image.LANCZOS)
    
    return img

def create_thumbnail(img_file, size=(200, 200)):
    """
    Create a thumbnail from an image file
    
    Args:
        img_file: BytesIO or file-like object containing image data
        size: Thumbnail size as (width, height)
        
    Returns:
        PIL Image thumbnail
    """
    img_file.seek(0)
    img = Image.open(img_file).convert('RGB')
    img.thumbnail(size, Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.Resampling.LANCZOS)
    return img

def convert_image_format(img, format="JPEG"):
    """
    Convert image to desired format
    
    Args:
        img: PIL Image object
        format: Target format (JPEG, PNG, etc.)
        
    Returns:
        BytesIO object containing the converted image
    """
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer

def image_to_base64(img):
    """
    Convert PIL Image to base64 string
    
    Args:
        img: PIL Image object
        
    Returns:
        Base64 encoded string of image
    """
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def compute_image_hash(img_file, hash_type="phash"):
    """
    Compute perceptual hash of image for similarity comparison
    
    Args:
        img_file: BytesIO or file-like object containing image data
        hash_type: Type of hash ('phash', 'dhash', 'whash', 'average_hash')
        
    Returns:
        ImageHash object
    """
    if not IMAGEHASH_AVAILABLE:
        st.warning("imagehash library not available. Cannot compute image hash.")
        return None
        
    img_file.seek(0)
    img = Image.open(img_file).convert('RGB')
    
    if hash_type == "phash":
        return imagehash.phash(img)
    elif hash_type == "dhash":
        return imagehash.dhash(img)
    elif hash_type == "whash":
        return imagehash.whash(img)
    else:
        return imagehash.average_hash(img)

def extract_dominant_color(img_file, n_colors=1):
    """
    Extract dominant color(s) from image
    
    Args:
        img_file: BytesIO or file-like object containing image data
        n_colors: Number of dominant colors to extract
        
    Returns:
        List of RGB colors as tuples
    """
    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn not available. Cannot extract dominant colors.")
        return []
    
    img_file.seek(0)
    img = Image.open(img_file).convert('RGB')
    img = img.resize((100, 100))  # Resize for faster processing
    
    # Convert to numpy array and reshape
    pixels = np.array(img).reshape(-1, 3)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels)
    
    # Get the colors
    colors = kmeans.cluster_centers_.astype(int)
    
    # Convert to RGB tuples
    return [tuple(color) for color in colors]

def is_grayscale(img_file, threshold=10):
    """
    Determine if an image is grayscale/black and white
    
    Args:
        img_file: BytesIO or file-like object containing image data
        threshold: Threshold for color channel difference
        
    Returns:
        Boolean indicating if image is grayscale
    """
    img_file.seek(0)
    img = Image.open(img_file).convert('RGB')
    
    # Get a sample of pixels for efficiency
    img = img.resize((100, 100))
    img_array = np.array(img)
    
    # Calculate difference between RGB channels
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    rg_diff = np.abs(r - g).mean()
    rb_diff = np.abs(r - b).mean()
    gb_diff = np.abs(g - b).mean()
    
    # If all channel differences are below threshold, image is grayscale
    return rg_diff < threshold and rb_diff < threshold and gb_diff < threshold

def get_image_format(img_file):
    """
    Get the format of an image file
    
    Args:
        img_file: BytesIO or file-like object containing image data
        
    Returns:
        String representing image format
    """
    img_file.seek(0)
    img = Image.open(img_file)
    return img.format

def extract_image_patches(img, patch_size=128, stride=64):
    """
    Extract patches from image for detailed analysis
    
    Args:
        img: PIL Image object
        patch_size: Size of each square patch
        stride: Step size between patches
        
    Returns:
        List of PIL Image patches
    """
    width, height = img.size
    patches = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = img.crop((x, x + patch_size, y, y + patch_size))
            patches.append(patch)
    return patches

def detect_faces(img):
    """
    Detect faces in image using OpenCV
    
    Args:
        img: PIL Image object
        
    Returns:
        Tuple of (number of faces, list of face rectangles)
    """
    if not CV2_AVAILABLE:
        st.warning("OpenCV not available. Cannot detect faces.")
        return 0, []
    
    try:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Load face detector
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        return len(faces), faces
    except Exception as e:
        st.warning(f"Face detection error: {str(e)}")
        return 0, []

def calculate_brightness(img):
    """
    Calculate average brightness of image
    
    Args:
        img: PIL Image object
        
    Returns:
        Float representing average brightness (0-255)
    """
    # Convert to grayscale
    if img.mode != 'L':
        img_gray = img.convert('L')
    else:
        img_gray = img
    
    # Calculate mean brightness
    brightness = np.mean(np.array(img_gray))
    return brightness

def estimate_blur(img):
    """
    Estimate blur level using variance of Laplacian
    
    Args:
        img: PIL Image object
        
    Returns:
        Float representing blur estimate (higher = less blurry)
    """
    if not CV2_AVAILABLE:
        st.warning("OpenCV not available. Cannot estimate blur.")
        return 0
    
    try:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Calculate variance of Laplacian
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Higher values indicate less blur
        return blur_value
    except Exception as e:
        st.warning(f"Blur estimation error: {str(e)}")
        return 0

def compress_image(img, quality=85):
    """
    Compress image to reduce size
    
    Args:
        img: PIL Image object
        quality: JPEG compression quality (0-100)
        
    Returns:
        Compressed PIL Image
    """
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def get_image_foreground_mask(img):
    """
    Extract foreground mask using GrabCut algorithm
    
    Args:
        img: PIL Image object
        
    Returns:
        Numpy array with foreground mask (1=foreground, 0=background)
    """
    if not CV2_AVAILABLE:
        st.warning("OpenCV not available. Cannot extract foreground mask.")
        return None
    
    try:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Create initial mask
        mask = np.zeros(img_cv.shape[:2], np.uint8)
        
        # Set up the GrabCut algorithm
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define rectangle containing foreground
        height, width = img_cv.shape[:2]
        margin = min(width, height) // 8
        rectangle = (margin, margin, width - margin, height - margin)
        
        # Apply GrabCut
        cv2.grabCut(img_cv, mask, rectangle, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create mask where definite/probable foreground is marked
        mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
        
        return mask2
    except Exception as e:
        st.warning(f"Foreground extraction error: {str(e)}")
        return None