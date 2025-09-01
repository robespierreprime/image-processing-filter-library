"""
Tests for the preset management system.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from image_processing_library.core import (
    PresetManager, 
    PresetMetadata, 
    ExecutionQueue, 
    FilterStep,
    BaseFilter,
    DataType,
    ColorFormat,
    PresetError
)
from image_processing_library.core.preset_manager import (
    PresetNotFoundError,
    PresetValidationError,
    FilterClassNotFoundError
)


class MockFilter(BaseFilter):
    """Mock filter for testing."""
    
    def __init__(self, intensity=1.0, **kwargs):
        super().__init__(
            name="mock_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test",
            intensity=intensity,
            **kwargs
        )
    
    def apply(self, data, **kwargs):
        """Mock apply method."""
        return data * self.parameters.get('intensity', 1.0)


class AnotherMockFilter(BaseFilter):
    """Another mock filter for testing."""
    
    def __init__(self, blur_radius=5, **kwargs):
        super().__init__(
            name="another_mock_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test",
            blur_radius=blur_radius,
            **kwargs
        )
    
    def apply(self, data, **kwargs):
        """Mock apply method."""
        return data


class TestPresetMetadata(unittest.TestCase):
    """Test cases for PresetMetadata."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)
        metadata = PresetMetadata(
            name="test_preset",
            description="Test description",
            created_at=created_at,
            version="2.0",
            author="Test Author"
        )
        
        expected = {
            "name": "test_preset",
            "description": "Test description",
            "created_at": "2023-01-01T12:00:00",
            "version": "2.0",
            "author": "Test Author"
        }
        
        self.assertEqual(metadata.to_dict(), expected)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "test_preset",
            "description": "Test description",
            "created_at": "2023-01-01T12:00:00",
            "version": "2.0",
            "author": "Test Author"
        }
        
        metadata = PresetMetadata.from_dict(data)
        
        self.assertEqual(metadata.name, "test_preset")
        self.assertEqual(metadata.description, "Test description")
        self.assertEqual(metadata.created_at, datetime(2023, 1, 1, 12, 0, 0))
        self.assertEqual(metadata.version, "2.0")
        self.assertEqual(metadata.author, "Test Author")
    
    def test_from_dict_with_defaults(self):
        """Test creation from dictionary with default values."""
        data = {
            "name": "test_preset",
            "description": "Test description",
            "created_at": "2023-01-01T12:00:00"
        }
        
        metadata = PresetMetadata.from_dict(data)
        
        self.assertEqual(metadata.version, "1.0")
        self.assertIsNone(metadata.author)


class TestPresetManager(unittest.TestCase):
    """Test cases for PresetManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.preset_manager = PresetManager(self.temp_dir)
        
        # Create a mock execution queue
        self.execution_queue = ExecutionQueue()
        
        # Add mock filter steps
        step1 = FilterStep(
            filter_class=MockFilter,
            parameters={"intensity": 2.0},
            save_intermediate=False,
            save_path=None
        )
        
        step2 = FilterStep(
            filter_class=AnotherMockFilter,
            parameters={"blur_radius": 10},
            save_intermediate=True,
            save_path="/tmp/intermediate.png"
        )
        
        self.execution_queue.steps = [step1, step2]
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_preset(self):
        """Test saving a preset."""
        preset_path = self.preset_manager.save_preset(
            name="test_preset",
            execution_queue=self.execution_queue,
            description="Test preset",
            author="Test Author"
        )
        
        # Check that file was created
        self.assertTrue(Path(preset_path).exists())
        
        # Check file content
        with open(preset_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn("metadata", data)
        self.assertIn("steps", data)
        
        # Check metadata
        metadata = data["metadata"]
        self.assertEqual(metadata["name"], "test_preset")
        self.assertEqual(metadata["description"], "Test preset")
        self.assertEqual(metadata["author"], "Test Author")
        self.assertEqual(metadata["version"], "1.0")
        
        # Check steps
        steps = data["steps"]
        self.assertEqual(len(steps), 2)
        
        # Check first step
        step1 = steps[0]
        self.assertEqual(step1["filter_class"], f"{MockFilter.__module__}.{MockFilter.__name__}")
        self.assertEqual(step1["parameters"], {"intensity": 2.0})
        self.assertFalse(step1["save_intermediate"])
        self.assertIsNone(step1["save_path"])
        
        # Check second step
        step2 = steps[1]
        self.assertEqual(step2["filter_class"], f"{AnotherMockFilter.__module__}.{AnotherMockFilter.__name__}")
        self.assertEqual(step2["parameters"], {"blur_radius": 10})
        self.assertTrue(step2["save_intermediate"])
        self.assertEqual(step2["save_path"], "/tmp/intermediate.png")
    
    def test_load_preset(self):
        """Test loading a preset."""
        # First save a preset
        self.preset_manager.save_preset(
            name="test_preset",
            execution_queue=self.execution_queue,
            description="Test preset"
        )
        
        # Then load it
        loaded_queue = self.preset_manager.load_preset("test_preset")
        
        # Check that queue was loaded correctly
        self.assertEqual(len(loaded_queue.steps), 2)
        
        # Check first step
        step1 = loaded_queue.steps[0]
        self.assertEqual(step1.filter_class, MockFilter)
        self.assertEqual(step1.parameters, {"intensity": 2.0})
        self.assertFalse(step1.save_intermediate)
        self.assertIsNone(step1.save_path)
        
        # Check second step
        step2 = loaded_queue.steps[1]
        self.assertEqual(step2.filter_class, AnotherMockFilter)
        self.assertEqual(step2.parameters, {"blur_radius": 10})
        self.assertTrue(step2.save_intermediate)
        self.assertEqual(step2.save_path, "/tmp/intermediate.png")
    
    def test_load_nonexistent_preset(self):
        """Test loading a preset that doesn't exist."""
        with self.assertRaises(PresetNotFoundError):
            self.preset_manager.load_preset("nonexistent_preset")
    
    def test_load_invalid_json(self):
        """Test loading a preset with invalid JSON."""
        # Create invalid JSON file
        preset_path = Path(self.temp_dir) / "invalid.json"
        with open(preset_path, 'w') as f:
            f.write("invalid json content")
        
        with self.assertRaises(PresetValidationError):
            self.preset_manager.load_preset("invalid")
    
    def test_load_preset_missing_metadata(self):
        """Test loading a preset with missing metadata."""
        # Create preset file without metadata
        preset_data = {"steps": []}
        preset_path = Path(self.temp_dir) / "no_metadata.json"
        with open(preset_path, 'w') as f:
            json.dump(preset_data, f)
        
        with self.assertRaises(PresetValidationError):
            self.preset_manager.load_preset("no_metadata")
    
    def test_load_preset_missing_steps(self):
        """Test loading a preset with missing steps."""
        # Create preset file without steps
        preset_data = {
            "metadata": {
                "name": "test",
                "description": "test",
                "created_at": "2023-01-01T12:00:00"
            }
        }
        preset_path = Path(self.temp_dir) / "no_steps.json"
        with open(preset_path, 'w') as f:
            json.dump(preset_data, f)
        
        with self.assertRaises(PresetValidationError):
            self.preset_manager.load_preset("no_steps")
    
    def test_import_filter_class_invalid_path(self):
        """Test importing a filter class with invalid path."""
        with self.assertRaises(FilterClassNotFoundError):
            self.preset_manager._import_filter_class("invalid.module.path")
    
    def test_import_filter_class_nonexistent_module(self):
        """Test importing a filter class from nonexistent module."""
        with self.assertRaises(FilterClassNotFoundError):
            self.preset_manager._import_filter_class("nonexistent_module.SomeClass")
    
    def test_import_filter_class_nonexistent_class(self):
        """Test importing a nonexistent class from valid module."""
        with self.assertRaises(FilterClassNotFoundError):
            self.preset_manager._import_filter_class("unittest.NonexistentClass")
    
    def test_list_presets(self):
        """Test listing all presets."""
        # Save multiple presets
        self.preset_manager.save_preset(
            name="preset1",
            execution_queue=self.execution_queue,
            description="First preset",
            author="Author 1"
        )
        
        self.preset_manager.save_preset(
            name="preset2",
            execution_queue=self.execution_queue,
            description="Second preset",
            author="Author 2"
        )
        
        # List presets
        presets = self.preset_manager.list_presets()
        
        self.assertEqual(len(presets), 2)
        
        # Check that both presets are in the list
        preset_names = [p.name for p in presets]
        self.assertIn("preset1", preset_names)
        self.assertIn("preset2", preset_names)
    
    def test_list_presets_with_invalid_files(self):
        """Test listing presets with some invalid files in directory."""
        # Save a valid preset
        self.preset_manager.save_preset(
            name="valid_preset",
            execution_queue=self.execution_queue,
            description="Valid preset"
        )
        
        # Create an invalid JSON file
        invalid_path = Path(self.temp_dir) / "invalid.json"
        with open(invalid_path, 'w') as f:
            f.write("invalid json")
        
        # Create a file without metadata
        no_metadata_path = Path(self.temp_dir) / "no_metadata.json"
        with open(no_metadata_path, 'w') as f:
            json.dump({"steps": []}, f)
        
        # List presets should only return the valid one
        presets = self.preset_manager.list_presets()
        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0].name, "valid_preset")
    
    def test_delete_preset(self):
        """Test deleting a preset."""
        # Save a preset
        self.preset_manager.save_preset(
            name="to_delete",
            execution_queue=self.execution_queue,
            description="Preset to delete"
        )
        
        # Check it exists
        self.assertTrue(self.preset_manager.preset_exists("to_delete"))
        
        # Delete it
        result = self.preset_manager.delete_preset("to_delete")
        self.assertTrue(result)
        
        # Check it's gone
        self.assertFalse(self.preset_manager.preset_exists("to_delete"))
    
    def test_delete_nonexistent_preset(self):
        """Test deleting a preset that doesn't exist."""
        result = self.preset_manager.delete_preset("nonexistent")
        self.assertFalse(result)
    
    def test_preset_exists(self):
        """Test checking if a preset exists."""
        # Check nonexistent preset
        self.assertFalse(self.preset_manager.preset_exists("nonexistent"))
        
        # Save a preset
        self.preset_manager.save_preset(
            name="existing",
            execution_queue=self.execution_queue,
            description="Existing preset"
        )
        
        # Check existing preset
        self.assertTrue(self.preset_manager.preset_exists("existing"))
    
    def test_save_preset_error_handling(self):
        """Test error handling during preset saving."""
        # Create a mock execution queue with invalid filter class
        invalid_queue = ExecutionQueue()
        
        # Create a step with a filter class that can't be serialized properly
        class UnserializableFilter:
            pass
        
        step = FilterStep(
            filter_class=UnserializableFilter,
            parameters={},
            save_intermediate=False,
            save_path=None
        )
        invalid_queue.steps = [step]
        
        # This should work fine as we just serialize the class path
        try:
            self.preset_manager.save_preset(
                name="test_error",
                execution_queue=invalid_queue,
                description="Test error handling"
            )
        except PresetError:
            self.fail("save_preset raised PresetError unexpectedly")


class TestPresetManagerIntegration(unittest.TestCase):
    """Integration tests for preset manager with real filter workflows."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.preset_manager = PresetManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_preset_workflow(self):
        """Test complete workflow: create queue, save preset, load preset, execute."""
        import numpy as np
        
        # Create execution queue with multiple filters
        queue = ExecutionQueue()
        
        # Add filters to queue
        queue.add_filter(
            filter_class=MockFilter,
            parameters={"intensity": 1.5}
        )
        
        queue.add_filter(
            filter_class=AnotherMockFilter,
            parameters={"blur_radius": 3}
        )
        
        # Save as preset
        preset_path = self.preset_manager.save_preset(
            name="workflow_test",
            execution_queue=queue,
            description="Complete workflow test",
            author="Test Suite"
        )
        
        # Load preset
        loaded_queue = self.preset_manager.load_preset("workflow_test")
        
        # Execute loaded queue
        test_data = np.ones((10, 10, 3))
        result = loaded_queue.execute(test_data)
        
        # Verify result (MockFilter multiplies by intensity)
        expected = test_data * 1.5
        np.testing.assert_array_equal(result, expected)
    
    def test_preset_with_different_filter_types(self):
        """Test preset save/load with various filter types and parameters."""
        # Create queue with filters having different parameter types
        queue = ExecutionQueue()
        
        # Filter with numeric parameters
        queue.add_filter(
            filter_class=MockFilter,
            parameters={
                "intensity": 2.5,
                "threshold": 128,
                "enabled": True,
                "mode": "normal"
            }
        )
        
        # Filter with complex parameters
        queue.add_filter(
            filter_class=AnotherMockFilter,
            parameters={
                "blur_radius": 7,
                "kernel_size": [3, 3],
                "weights": [0.1, 0.8, 0.1],
                "config": {"adaptive": True, "iterations": 5}
            }
        )
        
        # Save and load preset
        self.preset_manager.save_preset(
            name="complex_params",
            execution_queue=queue,
            description="Test with complex parameters"
        )
        
        loaded_queue = self.preset_manager.load_preset("complex_params")
        
        # Verify parameters were preserved
        self.assertEqual(len(loaded_queue.steps), 2)
        
        step1 = loaded_queue.steps[0]
        self.assertEqual(step1.parameters["intensity"], 2.5)
        self.assertEqual(step1.parameters["threshold"], 128)
        self.assertTrue(step1.parameters["enabled"])
        self.assertEqual(step1.parameters["mode"], "normal")
        
        step2 = loaded_queue.steps[1]
        self.assertEqual(step2.parameters["blur_radius"], 7)
        self.assertEqual(step2.parameters["kernel_size"], [3, 3])
        self.assertEqual(step2.parameters["weights"], [0.1, 0.8, 0.1])
        self.assertEqual(step2.parameters["config"], {"adaptive": True, "iterations": 5})


class TestPresetManagerAdvanced(unittest.TestCase):
    """Advanced test cases for PresetManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.preset_manager = PresetManager(self.temp_dir)
        
        # Create a complex execution queue for testing
        self.complex_queue = ExecutionQueue()
        
        step1 = FilterStep(
            filter_class=MockFilter,
            parameters={"intensity": 1.5, "mode": "enhance"},
            save_intermediate=True,
            save_path="/tmp/step1.png"
        )
        
        step2 = FilterStep(
            filter_class=AnotherMockFilter,
            parameters={"blur_radius": 5, "iterations": 3},
            save_intermediate=False,
            save_path=None
        )
        
        step3 = FilterStep(
            filter_class=MockFilter,
            parameters={"intensity": 0.8, "threshold": 128},
            save_intermediate=True,
            save_path="/tmp/final.jpg"
        )
        
        self.complex_queue.steps = [step1, step2, step3]
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_preset_metadata_comprehensive(self):
        """Test comprehensive preset metadata handling."""
        created_time = datetime.now()
        
        preset_path = self.preset_manager.save_preset(
            name="metadata_test",
            execution_queue=self.complex_queue,
            description="Comprehensive metadata test preset",
            author="Test Suite v2.0"
        )
        
        # Load and verify metadata
        with open(preset_path, 'r') as f:
            data = json.load(f)
        
        metadata = data["metadata"]
        self.assertEqual(metadata["name"], "metadata_test")
        self.assertEqual(metadata["description"], "Comprehensive metadata test preset")
        self.assertEqual(metadata["author"], "Test Suite v2.0")
        self.assertEqual(metadata["version"], "1.0")
        
        # Check timestamp is reasonable (within last minute)
        saved_time = datetime.fromisoformat(metadata["created_at"])
        time_diff = abs((saved_time - created_time).total_seconds())
        self.assertLess(time_diff, 60)  # Should be within 1 minute
    
    def test_preset_with_complex_parameters(self):
        """Test preset save/load with complex parameter structures."""
        # Create queue with nested and complex parameters
        queue = ExecutionQueue()
        
        complex_params = {
            "nested_config": {
                "processing": {
                    "method": "adaptive",
                    "parameters": [1.0, 2.5, 3.7],
                    "enabled": True
                },
                "output": {
                    "format": "RGB",
                    "quality": 95
                }
            },
            "simple_params": {
                "intensity": 2.0,
                "threshold": 128
            },
            "list_params": [1, 2, 3, "test", True],
            "tuple_params": (10, 20, 30)  # Will be converted to list in JSON
        }
        
        queue.add_filter(MockFilter, complex_params)
        
        # Save and load preset
        self.preset_manager.save_preset(
            name="complex_params_test",
            execution_queue=queue,
            description="Test complex parameter structures"
        )
        
        loaded_queue = self.preset_manager.load_preset("complex_params_test")
        
        # Verify complex parameters were preserved
        loaded_params = loaded_queue.steps[0].parameters
        
        # Check nested structure
        self.assertEqual(
            loaded_params["nested_config"]["processing"]["method"], 
            "adaptive"
        )
        self.assertEqual(
            loaded_params["nested_config"]["processing"]["parameters"], 
            [1.0, 2.5, 3.7]
        )
        self.assertTrue(loaded_params["nested_config"]["processing"]["enabled"])
        
        # Check simple parameters
        self.assertEqual(loaded_params["simple_params"]["intensity"], 2.0)
        self.assertEqual(loaded_params["simple_params"]["threshold"], 128)
        
        # Check list parameters
        self.assertEqual(loaded_params["list_params"], [1, 2, 3, "test", True])
        
        # Tuple becomes list in JSON
        self.assertEqual(loaded_params["tuple_params"], [10, 20, 30])
    
    def test_preset_versioning_and_compatibility(self):
        """Test preset versioning and backward compatibility."""
        # Create preset with specific version
        queue = ExecutionQueue()
        queue.add_filter(MockFilter, {"intensity": 1.0})
        
        preset_path = self.preset_manager.save_preset(
            name="versioned_preset",
            execution_queue=queue,
            description="Version test"
        )
        
        # Manually modify the preset file to simulate older version
        with open(preset_path, 'r') as f:
            data = json.load(f)
        
        # Simulate older version format
        data["metadata"]["version"] = "0.9"
        data["metadata"]["legacy_field"] = "old_value"
        
        with open(preset_path, 'w') as f:
            json.dump(data, f)
        
        # Should still load successfully
        loaded_queue = self.preset_manager.load_preset("versioned_preset")
        self.assertEqual(len(loaded_queue.steps), 1)
    
    def test_preset_with_unicode_and_special_characters(self):
        """Test preset handling with unicode and special characters."""
        queue = ExecutionQueue()
        
        # Parameters with unicode and special characters
        unicode_params = {
            "description": "Тест с русскими символами",
            "emoji": "🎨🖼️📸",
            "special_chars": "!@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        }
        
        queue.add_filter(MockFilter, unicode_params)
        
        # Save preset with unicode name and description
        preset_path = self.preset_manager.save_preset(
            name="unicode_test_🎨",
            execution_queue=queue,
            description="Preset with unicode: 测试 🚀",
            author="Автор Тест"
        )
        
        # Load and verify
        loaded_queue = self.preset_manager.load_preset("unicode_test_🎨")
        loaded_params = loaded_queue.steps[0].parameters
        
        self.assertEqual(loaded_params["description"], "Тест с русскими символами")
        self.assertEqual(loaded_params["emoji"], "🎨🖼️📸")
        self.assertEqual(loaded_params["special_chars"], "!@#$%^&*()_+-={}[]|\\:;\"'<>?,./")
    
    def test_preset_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios."""
        # Test with corrupted JSON
        corrupted_path = Path(self.temp_dir) / "corrupted.json"
        with open(corrupted_path, 'w') as f:
            f.write('{"metadata": {"name": "test"}, "steps": [invalid json')
        
        with self.assertRaises(PresetValidationError):
            self.preset_manager.load_preset("corrupted")
        
        # Test with missing required fields
        incomplete_data = {
            "metadata": {"name": "incomplete"},
            # Missing steps field
        }
        incomplete_path = Path(self.temp_dir) / "incomplete.json"
        with open(incomplete_path, 'w') as f:
            json.dump(incomplete_data, f)
        
        with self.assertRaises(PresetValidationError):
            self.preset_manager.load_preset("incomplete")
        
        # Test with invalid filter class path
        invalid_filter_data = {
            "metadata": {
                "name": "invalid_filter",
                "description": "Test",
                "created_at": datetime.now().isoformat()
            },
            "steps": [{
                "filter_class": "nonexistent.module.NonexistentFilter",
                "parameters": {},
                "save_intermediate": False,
                "save_path": None
            }]
        }
        invalid_path = Path(self.temp_dir) / "invalid_filter.json"
        with open(invalid_path, 'w') as f:
            json.dump(invalid_filter_data, f)
        
        with self.assertRaises(FilterClassNotFoundError):
            self.preset_manager.load_preset("invalid_filter")
    
    def test_preset_directory_management(self):
        """Test preset directory creation and management."""
        # Test with non-existent directory
        non_existent_dir = Path(self.temp_dir) / "nested" / "presets"
        preset_manager = PresetManager(str(non_existent_dir))
        
        # Directory should be created automatically
        self.assertTrue(non_existent_dir.exists())
        
        # Should be able to save presets
        queue = ExecutionQueue()
        queue.add_filter(MockFilter, {"intensity": 1.0})
        
        preset_path = preset_manager.save_preset(
            name="nested_test",
            execution_queue=queue,
            description="Test in nested directory"
        )
        
        self.assertTrue(Path(preset_path).exists())
    
    def test_preset_list_with_mixed_files(self):
        """Test listing presets with various file types in directory."""
        # Create valid presets
        queue = ExecutionQueue()
        queue.add_filter(MockFilter, {"intensity": 1.0})
        
        self.preset_manager.save_preset("valid1", queue, "First valid preset")
        self.preset_manager.save_preset("valid2", queue, "Second valid preset")
        
        # Create non-JSON files
        (Path(self.temp_dir) / "readme.txt").write_text("This is not a preset")
        (Path(self.temp_dir) / "config.ini").write_text("[section]\nkey=value")
        
        # Create invalid JSON files
        (Path(self.temp_dir) / "invalid.json").write_text("not valid json")
        
        # Create JSON file without proper preset structure
        invalid_preset = {"not": "a preset"}
        with open(Path(self.temp_dir) / "not_preset.json", 'w') as f:
            json.dump(invalid_preset, f)
        
        # List presets should only return valid ones
        presets = self.preset_manager.list_presets()
        preset_names = [p.name for p in presets]
        
        self.assertEqual(len(presets), 2)
        self.assertIn("valid1", preset_names)
        self.assertIn("valid2", preset_names)
    
    def test_preset_concurrent_access_simulation(self):
        """Test preset operations that might happen concurrently."""
        queue = ExecutionQueue()
        queue.add_filter(MockFilter, {"intensity": 1.0})
        
        # Simulate multiple save operations
        for i in range(5):
            self.preset_manager.save_preset(
                f"concurrent_test_{i}",
                queue,
                f"Concurrent test preset {i}"
            )
        
        # Verify all presets were saved
        presets = self.preset_manager.list_presets()
        self.assertEqual(len(presets), 5)
        
        # Test loading all presets
        for i in range(5):
            loaded_queue = self.preset_manager.load_preset(f"concurrent_test_{i}")
            self.assertEqual(len(loaded_queue.steps), 1)
        
        # Test deleting some presets
        self.assertTrue(self.preset_manager.delete_preset("concurrent_test_2"))
        self.assertFalse(self.preset_manager.preset_exists("concurrent_test_2"))
        
        # Verify others still exist
        self.assertTrue(self.preset_manager.preset_exists("concurrent_test_0"))
        self.assertTrue(self.preset_manager.preset_exists("concurrent_test_4"))


if __name__ == '__main__':
    unittest.main()