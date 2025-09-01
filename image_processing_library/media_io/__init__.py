"""
Input/Output module for handling image and video file operations.
"""

from .image_io import (
    load_image,
    load_image_pil,
    load_image_cv2,
    save_image,
    save_image_pil,
    save_image_cv2,
    numpy_to_pil,
    pil_to_numpy,
    get_image_info,
    validate_image_array,
    ImageIOError,
    SUPPORTED_FORMATS
)

from .video_io import (
    VideoReader,
    VideoWriter,
    VideoInfo,
    VideoIOError,
    load_video,
    save_video,
    get_video_info,
    extract_frames,
    process_video_frames,
    create_video_from_images,
    SUPPORTED_VIDEO_FORMATS,
    VIDEO_CODECS
)

__all__ = [
    # Image I/O
    'load_image',
    'load_image_pil', 
    'load_image_cv2',
    'save_image',
    'save_image_pil',
    'save_image_cv2',
    'numpy_to_pil',
    'pil_to_numpy',
    'get_image_info',
    'validate_image_array',
    'ImageIOError',
    'SUPPORTED_FORMATS',
    
    # Video I/O
    'VideoReader',
    'VideoWriter',
    'VideoInfo',
    'VideoIOError',
    'load_video',
    'save_video',
    'get_video_info',
    'extract_frames',
    'process_video_frames',
    'create_video_from_images',
    'SUPPORTED_VIDEO_FORMATS',
    'VIDEO_CODECS'
]