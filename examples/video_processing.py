#!/usr/bin/env python3
"""
Video Processing Example

This example demonstrates how to process video files using the library.
"""

import numpy as np
from image_processing_library.media_io import VideoReader, VideoWriter, create_video_from_images
from image_processing_library.filters.artistic import GlitchFilter
from image_processing_library.filters.artistic import PrintSimulationFilter

def main():
    """Demonstrate video processing capabilities."""
    
    # Create a sample video for demonstration
    print("1. Creating sample video...")
    sample_video_path = "examples/sample_video.mp4"
    create_sample_video(sample_video_path)
    
    # Example 1: Process video frame by frame
    print("\n2. Processing video frame by frame...")
    process_video_frames(sample_video_path, "examples/output_glitch_video.mp4")
    
    # Example 2: Apply multiple filters to video
    print("\n3. Applying multiple filters to video...")
    apply_multiple_filters_to_video(sample_video_path, "examples/output_multi_filter_video.mp4")
    
    print("\nVideo processing examples completed!")

def create_sample_video(output_path, duration_seconds=3, fps=10):
    """Create a sample video with animated content."""
    
    frames = []
    total_frames = duration_seconds * fps
    
    for frame_num in range(total_frames):
        # Create animated frame
        frame = create_animated_frame(frame_num, total_frames)
        frames.append(frame)
    
    # Convert to numpy array
    video_array = np.array(frames)
    
    # Save as video
    writer = VideoWriter(output_path, fps=fps, codec='mp4v')
    
    try:
        for frame in video_array:
            writer.write_frame(frame)
        print(f"   Created sample video: {output_path}")
    finally:
        writer.release()

def create_animated_frame(frame_num, total_frames):
    """Create a single animated frame."""
    height, width = 200, 300
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Animated circle
    progress = frame_num / total_frames
    center_x = int(50 + (width - 100) * progress)
    center_y = height // 2
    radius = 20 + int(10 * np.sin(progress * 4 * np.pi))
    
    # Draw animated circle
    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if distance < radius:
                # Color changes over time
                r = int(255 * (0.5 + 0.5 * np.sin(progress * 2 * np.pi)))
                g = int(255 * (0.5 + 0.5 * np.cos(progress * 2 * np.pi)))
                b = int(255 * (0.5 + 0.5 * np.sin(progress * 3 * np.pi)))
                frame[y, x] = [r, g, b]
            else:
                # Background gradient
                frame[y, x] = [
                    int(50 + 50 * progress),
                    int(100 * (1 - progress)),
                    int(150)
                ]
    
    return frame

def process_video_frames(input_path, output_path):
    """Process video frames with a single filter."""
    
    reader = VideoReader(input_path)
    writer = VideoWriter(output_path, fps=reader.fps, codec='mp4v')
    
    # Create filter
    glitch_filter = GlitchFilter(shift_intensity=8, line_width=3)
    
    def progress_callback(progress):
        print(f"   Frame progress: {progress:.1%}")
    
    glitch_filter.set_progress_callback(progress_callback)
    
    try:
        frame_count = 0
        total_frames = reader.frame_count
        
        for frame in reader:
            # Apply filter to frame
            processed_frame = glitch_filter.apply(frame)
            writer.write_frame(processed_frame)
            
            frame_count += 1
            overall_progress = frame_count / total_frames
            print(f"   Overall progress: {overall_progress:.1%}")
        
        print(f"   Processed {frame_count} frames")
        
    finally:
        reader.release()
        writer.release()

def apply_multiple_filters_to_video(input_path, output_path):
    """Apply multiple filters to video using frame-by-frame processing."""
    
    reader = VideoReader(input_path)
    writer = VideoWriter(output_path, fps=reader.fps, codec='mp4v')
    
    # Create filters
    glitch_filter = GlitchFilter(shift_intensity=5, line_width=2)
    print_filter = PrintSimulationFilter(band_intensity=15, band_frequency=30)
    
    try:
        frame_count = 0
        total_frames = reader.frame_count
        
        for frame in reader:
            # Apply filters in sequence
            temp_frame = glitch_filter.apply(frame)
            processed_frame = print_filter.apply(temp_frame)
            
            writer.write_frame(processed_frame)
            
            frame_count += 1
            progress = frame_count / total_frames
            print(f"   Processing frame {frame_count}/{total_frames} ({progress:.1%})")
        
        print(f"   Applied multiple filters to {frame_count} frames")
        
    finally:
        reader.release()
        writer.release()

if __name__ == "__main__":
    main()