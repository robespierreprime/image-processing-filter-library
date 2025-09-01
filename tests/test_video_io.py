"""
Unit tests for video I/O functionality.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import cv2

from image_processing_library.media_io.video_io import (
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
    VIDEO_CODECS,
)


class TestVideoIO(unittest.TestCase):
    """Test cases for video I/O operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a simple test video
        self.test_video_path = Path(self.temp_dir) / "test_video.mp4"
        self.create_test_video(self.test_video_path)

        # Test video properties
        self.video_width = 100
        self.video_height = 100
        self.video_fps = 10.0
        self.video_frames = 30

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_test_video(
        self,
        output_path: Path,
        width: int = 100,
        height: int = 100,
        fps: float = 10.0,
        num_frames: int = 30,
    ):
        """Create a test video with colored frames."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for i in range(num_frames):
            # Create a frame with changing colors
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 8) % 256  # Red channel changes
            frame[:, :, 1] = 128  # Green constant
            frame[:, :, 2] = 255 - (i * 8) % 256  # Blue channel changes

            writer.write(frame)

        writer.release()

    def test_video_reader_initialization(self):
        """Test VideoReader initialization."""
        reader = VideoReader(self.test_video_path)

        self.assertEqual(reader.width, self.video_width)
        self.assertEqual(reader.height, self.video_height)
        self.assertAlmostEqual(reader.fps, self.video_fps, places=1)
        self.assertEqual(reader.frame_count, self.video_frames)

        reader.close()

    def test_video_reader_context_manager(self):
        """Test VideoReader as context manager."""
        with VideoReader(self.test_video_path) as reader:
            self.assertEqual(reader.width, self.video_width)
            self.assertEqual(reader.height, self.video_height)

    def test_video_reader_nonexistent_file(self):
        """Test VideoReader with non-existent file."""
        with self.assertRaises(VideoIOError):
            VideoReader("nonexistent_video.mp4")

    def test_video_reader_read_frame(self):
        """Test reading individual frames."""
        with VideoReader(self.test_video_path) as reader:
            frame = reader.read_frame()

            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (self.video_height, self.video_width, 3))
            self.assertEqual(frame.dtype, np.uint8)

    def test_video_reader_read_all_frames(self):
        """Test reading all frames at once."""
        with VideoReader(self.test_video_path) as reader:
            frames = reader.read_all_frames()

            self.assertEqual(
                frames.shape,
                (self.video_frames, self.video_height, self.video_width, 3),
            )
            self.assertEqual(frames.dtype, np.uint8)

    def test_video_reader_get_frame_at(self):
        """Test getting specific frame by number."""
        with VideoReader(self.test_video_path) as reader:
            # Get middle frame
            frame = reader.get_frame_at(15)

            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (self.video_height, self.video_width, 3))

            # Test out of bounds
            frame_oob = reader.get_frame_at(1000)
            self.assertIsNone(frame_oob)

    def test_video_reader_read_frames_iterator(self):
        """Test reading frames as iterator."""
        with VideoReader(self.test_video_path) as reader:
            frames = list(reader.read_frames(0, 10))  # Read first 10 frames

            self.assertEqual(len(frames), 10)
            for frame in frames:
                self.assertEqual(frame.shape, (self.video_height, self.video_width, 3))

    def test_video_reader_progress_callback(self):
        """Test progress callback functionality."""
        progress_values = []

        def progress_callback(progress):
            progress_values.append(progress)

        with VideoReader(self.test_video_path) as reader:
            reader.set_progress_callback(progress_callback)

            # Read a few frames
            for _ in range(5):
                reader.read_frame()

        self.assertTrue(len(progress_values) > 0)
        self.assertTrue(all(0 <= p <= 1 for p in progress_values))

    def test_video_writer_initialization(self):
        """Test VideoWriter initialization."""
        output_path = Path(self.temp_dir) / "output_test.mp4"

        writer = VideoWriter(
            output_path, self.video_width, self.video_height, self.video_fps
        )

        self.assertEqual(writer.width, self.video_width)
        self.assertEqual(writer.height, self.video_height)
        self.assertEqual(writer.fps, self.video_fps)

        writer.close()

    def test_video_writer_context_manager(self):
        """Test VideoWriter as context manager."""
        output_path = Path(self.temp_dir) / "output_context.mp4"

        with VideoWriter(
            output_path, self.video_width, self.video_height, self.video_fps
        ) as writer:
            self.assertEqual(writer.width, self.video_width)

    def test_video_writer_write_frame(self):
        """Test writing individual frames."""
        output_path = Path(self.temp_dir) / "output_frames.mp4"

        with VideoWriter(
            output_path, self.video_width, self.video_height, self.video_fps
        ) as writer:
            # Create and write test frames
            for i in range(10):
                frame = np.random.randint(
                    0, 255, (self.video_height, self.video_width, 3), dtype=np.uint8
                )
                writer.write_frame(frame)

        # Verify the video was created
        self.assertTrue(output_path.exists())

        # Verify we can read it back
        with VideoReader(output_path) as reader:
            self.assertEqual(reader.width, self.video_width)
            self.assertEqual(reader.height, self.video_height)

    def test_video_writer_write_frames_batch(self):
        """Test writing multiple frames at once."""
        output_path = Path(self.temp_dir) / "output_batch.mp4"

        # Create test frames
        frames = np.random.randint(
            0, 255, (10, self.video_height, self.video_width, 3), dtype=np.uint8
        )

        with VideoWriter(
            output_path, self.video_width, self.video_height, self.video_fps
        ) as writer:
            writer.write_frames(frames)

        # Verify the video was created
        self.assertTrue(output_path.exists())

    def test_video_writer_invalid_frame_size(self):
        """Test writing frame with wrong dimensions."""
        output_path = Path(self.temp_dir) / "output_invalid.mp4"

        with VideoWriter(
            output_path, self.video_width, self.video_height, self.video_fps
        ) as writer:
            # Try to write frame with wrong size
            wrong_frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

            with self.assertRaises(VideoIOError):
                writer.write_frame(wrong_frame)

    def test_load_video_function(self):
        """Test load_video convenience function."""
        frames = load_video(self.test_video_path)

        self.assertEqual(
            frames.shape, (self.video_frames, self.video_height, self.video_width, 3)
        )
        self.assertEqual(frames.dtype, np.uint8)

    def test_save_video_function(self):
        """Test save_video convenience function."""
        output_path = Path(self.temp_dir) / "saved_video.mp4"

        # Create test frames
        frames = np.random.randint(
            0, 255, (15, self.video_height, self.video_width, 3), dtype=np.uint8
        )

        save_video(frames, output_path, fps=self.video_fps)

        # Verify the video was created and can be read
        self.assertTrue(output_path.exists())

        loaded_frames = load_video(output_path)
        self.assertEqual(loaded_frames.shape[0], 15)  # Number of frames

    def test_get_video_info_function(self):
        """Test get_video_info convenience function."""
        info = get_video_info(self.test_video_path)

        self.assertIsInstance(info, VideoInfo)
        self.assertEqual(info.width, self.video_width)
        self.assertEqual(info.height, self.video_height)
        self.assertAlmostEqual(info.fps, self.video_fps, places=1)
        self.assertEqual(info.frame_count, self.video_frames)

    def test_extract_frames_function(self):
        """Test extract_frames function."""
        output_dir = Path(self.temp_dir) / "extracted_frames"

        saved_paths = extract_frames(self.test_video_path, output_dir, 0, 5, "png")

        self.assertEqual(len(saved_paths), 5)

        # Verify files were created
        for path in saved_paths:
            self.assertTrue(Path(path).exists())

    def test_process_video_frames_function(self):
        """Test process_video_frames function."""
        output_path = Path(self.temp_dir) / "processed_video.mp4"

        def simple_processor(frame):
            # Simple processing: convert to grayscale and back to RGB
            gray = np.mean(frame, axis=2, keepdims=True)
            return np.repeat(gray, 3, axis=2).astype(np.uint8)

        process_video_frames(self.test_video_path, output_path, simple_processor)

        # Verify the processed video was created
        self.assertTrue(output_path.exists())

        # Verify it can be read
        with VideoReader(output_path) as reader:
            self.assertEqual(reader.width, self.video_width)
            self.assertEqual(reader.height, self.video_height)

    def test_create_video_from_images_function(self):
        """Test create_video_from_images function."""
        # Create test images
        image_dir = Path(self.temp_dir) / "test_images"
        image_dir.mkdir()

        image_paths = []
        for i in range(5):
            img_path = image_dir / f"image_{i:03d}.png"

            # Create a simple test image
            image = np.random.randint(
                0, 255, (self.video_height, self.video_width, 3), dtype=np.uint8
            )

            # Save using PIL
            from image_processing_library.media_io.image_io import save_image_pil

            save_image_pil(image, img_path)

            image_paths.append(img_path)

        # Create video from images
        output_path = Path(self.temp_dir) / "from_images.mp4"
        create_video_from_images(image_paths, output_path, fps=5.0)

        # Verify the video was created
        self.assertTrue(output_path.exists())

        # Verify properties
        with VideoReader(output_path) as reader:
            self.assertEqual(reader.frame_count, 5)
            self.assertEqual(reader.width, self.video_width)
            self.assertEqual(reader.height, self.video_height)

    def test_supported_formats_constants(self):
        """Test that format constants are properly defined."""
        self.assertIsInstance(SUPPORTED_VIDEO_FORMATS, set)
        self.assertIn(".mp4", SUPPORTED_VIDEO_FORMATS)
        self.assertIn(".avi", SUPPORTED_VIDEO_FORMATS)

        self.assertIsInstance(VIDEO_CODECS, dict)
        self.assertIn(".mp4", VIDEO_CODECS)

    def test_video_round_trip(self):
        """Test complete round-trip: load -> process -> save -> load."""
        # Load original video
        original_frames = load_video(self.test_video_path)

        # Save to new location
        output_path = Path(self.temp_dir) / "round_trip.mp4"
        save_video(original_frames, output_path, fps=self.video_fps)

        # Load the saved video
        loaded_frames = load_video(output_path)

        # Check basic properties (allowing for codec differences)
        self.assertEqual(
            loaded_frames.shape[0], original_frames.shape[0]
        )  # Same number of frames
        self.assertEqual(
            loaded_frames.shape[1:3], original_frames.shape[1:3]
        )  # Same dimensions


if __name__ == "__main__":
    unittest.main()
