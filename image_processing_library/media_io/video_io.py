"""
Video input/output functionality.

This module provides functions for video frame extraction, processing, and writing
with support for various codecs and progress tracking.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, Callable, Iterator, List
import os
from dataclasses import dataclass

# Common video codecs
VIDEO_CODECS = {
    '.mp4': 'mp4v',
    '.avi': 'XVID', 
    '.mov': 'mp4v',
    '.mkv': 'XVID',
    '.webm': 'VP80'
}

# Supported video formats
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}


@dataclass
class VideoInfo:
    """Information about a video file."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    file_path: str
    file_size: int


class VideoIOError(Exception):
    """Exception raised for video I/O operations."""
    pass


class VideoReader:
    """
    Video reader class for frame-by-frame processing with progress tracking.
    """
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize video reader.
        
        Args:
            file_path: Path to the video file
            
        Raises:
            VideoIOError: If the video cannot be opened
        """
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise VideoIOError(f"Video file not found: {self.file_path}")
        
        if self.file_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
            raise VideoIOError(f"Unsupported video format: {self.file_path.suffix}")
        
        self.cap = cv2.VideoCapture(str(self.file_path))
        
        if not self.cap.isOpened():
            raise VideoIOError(f"Failed to open video: {self.file_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        self.current_frame = 0
        self._progress_callback: Optional[Callable[[float], None]] = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def close(self):
        """Close the video reader."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def set_progress_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def _update_progress(self) -> None:
        """Update progress and call callback if set."""
        if self._progress_callback and self.frame_count > 0:
            progress = self.current_frame / self.frame_count
            self._progress_callback(progress)
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read the next frame from the video.
        
        Returns:
            numpy array containing the frame in RGB format, or None if no more frames
        """
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.current_frame += 1
        self._update_progress()
        
        return frame_rgb
    
    def read_frames(self, start_frame: int = 0, end_frame: Optional[int] = None) -> Iterator[np.ndarray]:
        """
        Read frames from the video as an iterator.
        
        Args:
            start_frame: Starting frame number (0-based)
            end_frame: Ending frame number (exclusive), None for all frames
            
        Yields:
            numpy arrays containing frames in RGB format
        """
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.current_frame = start_frame
        
        if end_frame is None:
            end_frame = self.frame_count
        
        while self.current_frame < end_frame:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame
    
    def read_all_frames(self) -> np.ndarray:
        """
        Read all frames from the video into a 4D numpy array.
        
        Returns:
            numpy array with shape (frames, height, width, channels)
            
        Warning:
            This loads all frames into memory and may consume large amounts of RAM
        """
        frames = []
        
        for frame in self.read_frames():
            frames.append(frame)
        
        if not frames:
            raise VideoIOError("No frames could be read from the video")
        
        return np.array(frames)
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by frame number.
        
        Args:
            frame_number: Frame number to retrieve (0-based)
            
        Returns:
            numpy array containing the frame in RGB format, or None if frame doesn't exist
        """
        if frame_number < 0 or frame_number >= self.frame_count:
            return None
        
        # Set position to the desired frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
        
        return self.read_frame()
    
    def get_info(self) -> VideoInfo:
        """
        Get information about the video.
        
        Returns:
            VideoInfo object containing video metadata
        """
        return VideoInfo(
            width=self.width,
            height=self.height,
            fps=self.fps,
            frame_count=self.frame_count,
            duration=self.duration,
            codec=self.cap.get(cv2.CAP_PROP_FOURCC),
            file_path=str(self.file_path),
            file_size=self.file_path.stat().st_size
        )


class VideoWriter:
    """
    Video writer class for creating video files with codec support.
    """
    
    def __init__(self, file_path: Union[str, Path], width: int, height: int, 
                 fps: float, codec: Optional[str] = None):
        """
        Initialize video writer.
        
        Args:
            file_path: Path where to save the video
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
            codec: Video codec (auto-detected from extension if None)
            
        Raises:
            VideoIOError: If the video writer cannot be initialized
        """
        self.file_path = Path(file_path)
        self.width = width
        self.height = height
        self.fps = fps
        
        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect codec from file extension
        if codec is None:
            ext = self.file_path.suffix.lower()
            codec = VIDEO_CODECS.get(ext, 'mp4v')
        
        # Convert codec string to fourcc
        if isinstance(codec, str):
            if len(codec) == 4:
                fourcc = cv2.VideoWriter_fourcc(*codec)
            else:
                # Handle common codec names
                codec_map = {
                    'h264': cv2.VideoWriter_fourcc(*'H264'),
                    'xvid': cv2.VideoWriter_fourcc(*'XVID'),
                    'mjpg': cv2.VideoWriter_fourcc(*'MJPG'),
                    'mp4v': cv2.VideoWriter_fourcc(*'mp4v')
                }
                fourcc = codec_map.get(codec.lower(), cv2.VideoWriter_fourcc(*'mp4v'))
        else:
            fourcc = codec
        
        self.writer = cv2.VideoWriter(
            str(self.file_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise VideoIOError(f"Failed to initialize video writer for {self.file_path}")
        
        self.frame_count = 0
        self._progress_callback: Optional[Callable[[float, int], None]] = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def close(self):
        """Close the video writer."""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.release()
    
    def set_progress_callback(self, callback: Callable[[float, int], None]) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a single frame to the video.
        
        Args:
            frame: numpy array containing the frame in RGB format
            
        Raises:
            VideoIOError: If the frame cannot be written
        """
        if frame.shape[:2] != (self.height, self.width):
            raise VideoIOError(
                f"Frame size {frame.shape[:2]} doesn't match video size ({self.height}, {self.width})"
            )
        
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        
        # Ensure frame is uint8
        if frame_bgr.dtype != np.uint8:
            frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
        
        self.writer.write(frame_bgr)
        self.frame_count += 1
        
        if self._progress_callback:
            # Progress is based on frames written (no total known)
            self._progress_callback(0.0, self.frame_count)
    
    def write_frames(self, frames: np.ndarray) -> None:
        """
        Write multiple frames to the video.
        
        Args:
            frames: 4D numpy array with shape (num_frames, height, width, channels)
        """
        if len(frames.shape) != 4:
            raise VideoIOError(f"Expected 4D array, got shape {frames.shape}")
        
        total_frames = frames.shape[0]
        
        for i, frame in enumerate(frames):
            self.write_frame(frame)
            
            if self._progress_callback:
                progress = (i + 1) / total_frames
                self._progress_callback(progress, i + 1)


def load_video(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load an entire video into a 4D numpy array.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        numpy array with shape (frames, height, width, channels) in RGB format
        
    Warning:
        This loads all frames into memory and may consume large amounts of RAM
    """
    with VideoReader(file_path) as reader:
        return reader.read_all_frames()


def save_video(frames: np.ndarray, file_path: Union[str, Path], 
               fps: float = 30.0, codec: Optional[str] = None) -> None:
    """
    Save a 4D numpy array as a video file.
    
    Args:
        frames: 4D numpy array with shape (num_frames, height, width, channels)
        file_path: Path where to save the video
        fps: Frames per second
        codec: Video codec (auto-detected from extension if None)
    """
    if len(frames.shape) != 4:
        raise VideoIOError(f"Expected 4D array, got shape {frames.shape}")
    
    num_frames, height, width, channels = frames.shape
    
    with VideoWriter(file_path, width, height, fps, codec) as writer:
        writer.write_frames(frames)


def get_video_info(file_path: Union[str, Path]) -> VideoInfo:
    """
    Get information about a video file without loading frames.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        VideoInfo object containing video metadata
    """
    with VideoReader(file_path) as reader:
        return reader.get_info()


def extract_frames(file_path: Union[str, Path], output_dir: Union[str, Path],
                   start_frame: int = 0, end_frame: Optional[int] = None,
                   frame_format: str = 'png') -> List[str]:
    """
    Extract frames from a video and save them as individual images.
    
    Args:
        file_path: Path to the video file
        output_dir: Directory where to save the extracted frames
        start_frame: Starting frame number (0-based)
        end_frame: Ending frame number (exclusive), None for all frames
        frame_format: Image format for saved frames ('png', 'jpg', etc.)
        
    Returns:
        List of paths to the saved frame images
    """
    from .image_io import save_image_pil
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    with VideoReader(file_path) as reader:
        for i, frame in enumerate(reader.read_frames(start_frame, end_frame)):
            frame_number = start_frame + i
            frame_path = output_dir / f"frame_{frame_number:06d}.{frame_format}"
            
            save_image_pil(frame, frame_path)
            saved_paths.append(str(frame_path))
    
    return saved_paths


def process_video_frames(input_path: Union[str, Path], output_path: Union[str, Path],
                        frame_processor: Callable[[np.ndarray], np.ndarray],
                        progress_callback: Optional[Callable[[float], None]] = None) -> None:
    """
    Process video frames using a custom function and save the result.
    
    Args:
        input_path: Path to the input video
        output_path: Path for the output video
        frame_processor: Function that takes a frame (RGB numpy array) and returns processed frame
        progress_callback: Optional callback for progress updates
    """
    with VideoReader(input_path) as reader:
        # Get video properties
        info = reader.get_info()
        
        with VideoWriter(output_path, info.width, info.height, info.fps) as writer:
            if progress_callback:
                reader.set_progress_callback(progress_callback)
            
            for frame in reader.read_frames():
                # Process the frame
                processed_frame = frame_processor(frame)
                
                # Write the processed frame
                writer.write_frame(processed_frame)


def create_video_from_images(image_paths: List[Union[str, Path]], 
                           output_path: Union[str, Path],
                           fps: float = 30.0, codec: Optional[str] = None) -> None:
    """
    Create a video from a sequence of images.
    
    Args:
        image_paths: List of paths to image files
        output_path: Path for the output video
        fps: Frames per second
        codec: Video codec (auto-detected from extension if None)
    """
    from .image_io import load_image_pil
    
    if not image_paths:
        raise VideoIOError("No image paths provided")
    
    # Load first image to get dimensions
    first_image = load_image_pil(image_paths[0])
    height, width = first_image.shape[:2]
    
    with VideoWriter(output_path, width, height, fps, codec) as writer:
        for i, img_path in enumerate(image_paths):
            frame = load_image_pil(img_path)
            
            # Ensure all images have the same dimensions
            if frame.shape[:2] != (height, width):
                raise VideoIOError(
                    f"Image {img_path} has different dimensions {frame.shape[:2]} "
                    f"than expected ({height}, {width})"
                )
            
            writer.write_frame(frame)
            
            # Update progress
            if writer._progress_callback:
                progress = (i + 1) / len(image_paths)
                writer._progress_callback(progress, i + 1)