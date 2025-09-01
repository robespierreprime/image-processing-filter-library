"""
Preset management system for saving and loading filter configurations.
"""

import json
import importlib
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from .utils import PresetError

if TYPE_CHECKING:
    from .execution_queue import ExecutionQueue


@dataclass
class PresetMetadata:
    """Metadata for filter presets."""
    name: str
    description: str
    created_at: datetime
    version: str = "1.0"
    author: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "author": self.author
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PresetMetadata':
        """Create PresetMetadata from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            version=data.get("version", "1.0"),
            author=data.get("author")
        )


class PresetNotFoundError(PresetError):
    """Raised when a preset file is not found."""
    pass


class PresetValidationError(PresetError):
    """Raised when preset data is invalid."""
    pass


class FilterClassNotFoundError(PresetError):
    """Raised when a filter class cannot be imported."""
    pass


class PresetManager:
    """Manages saving and loading of filter presets."""
    
    def __init__(self, presets_dir: str = "presets"):
        """
        Initialize PresetManager.
        
        Args:
            presets_dir: Directory to store preset files
        """
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(exist_ok=True)
    
    def save_preset(self, name: str, execution_queue: 'ExecutionQueue', 
                   description: str = "", author: Optional[str] = None) -> str:
        """
        Save an execution queue as a preset.
        
        Args:
            name: Name of the preset
            execution_queue: ExecutionQueue to save
            description: Optional description
            author: Optional author name
            
        Returns:
            Path to the saved preset file
            
        Raises:
            PresetError: If saving fails
        """
        try:
            metadata = PresetMetadata(
                name=name,
                description=description,
                created_at=datetime.now(),
                author=author
            )
            
            preset_data = {
                "metadata": metadata.to_dict(),
                "steps": []
            }
            
            # Serialize each step in the execution queue
            for step in execution_queue.steps:
                step_data = {
                    "filter_class": f"{step.filter_class.__module__}.{step.filter_class.__name__}",
                    "parameters": step.parameters,
                    "save_intermediate": step.save_intermediate,
                    "save_path": step.save_path
                }
                preset_data["steps"].append(step_data)
            
            # Save to JSON file
            preset_path = self.presets_dir / f"{name}.json"
            with open(preset_path, 'w') as f:
                json.dump(preset_data, f, indent=2)
            
            return str(preset_path)
            
        except Exception as e:
            raise PresetError(f"Failed to save preset '{name}': {str(e)}") from e
    
    def load_preset(self, name: str) -> 'ExecutionQueue':
        """
        Load a preset and return an execution queue.
        
        Args:
            name: Name of the preset to load
            
        Returns:
            ExecutionQueue configured from the preset
            
        Raises:
            PresetNotFoundError: If preset file doesn't exist
            PresetValidationError: If preset data is invalid
            FilterClassNotFoundError: If filter class cannot be imported
        """
        preset_path = self.presets_dir / f"{name}.json"
        
        if not preset_path.exists():
            raise PresetNotFoundError(f"Preset '{name}' not found at {preset_path}")
        
        try:
            with open(preset_path, 'r') as f:
                preset_data = json.load(f)
            
            # Validate preset structure
            if "metadata" not in preset_data or "steps" not in preset_data:
                raise PresetValidationError("Invalid preset format: missing metadata or steps")
            
            # Import ExecutionQueue here to avoid circular imports
            from .execution_queue import ExecutionQueue, FilterStep
            
            queue = ExecutionQueue()
            
            # Load each step
            for step_data in preset_data["steps"]:
                try:
                    # Dynamically import filter class
                    filter_class = self._import_filter_class(step_data["filter_class"])
                    
                    # Create FilterStep
                    step = FilterStep(
                        filter_class=filter_class,
                        parameters=step_data["parameters"],
                        save_intermediate=step_data.get("save_intermediate", False),
                        save_path=step_data.get("save_path")
                    )
                    
                    queue.steps.append(step)
                    
                except KeyError as e:
                    raise PresetValidationError(f"Missing required field in step: {e}")
            
            return queue
            
        except json.JSONDecodeError as e:
            raise PresetValidationError(f"Invalid JSON in preset '{name}': {str(e)}") from e
        except Exception as e:
            if isinstance(e, (PresetNotFoundError, PresetValidationError, FilterClassNotFoundError)):
                raise
            raise PresetError(f"Failed to load preset '{name}': {str(e)}") from e
    
    def _import_filter_class(self, class_path: str):
        """
        Dynamically import a filter class from its module path.
        
        Args:
            class_path: Full module path to the class (e.g., 'module.submodule.ClassName')
            
        Returns:
            The imported class
            
        Raises:
            FilterClassNotFoundError: If the class cannot be imported
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            filter_class = getattr(module, class_name)
            return filter_class
            
        except (ValueError, ImportError, AttributeError) as e:
            raise FilterClassNotFoundError(f"Cannot import filter class '{class_path}': {str(e)}") from e
    
    def list_presets(self) -> List[PresetMetadata]:
        """
        List all available presets.
        
        Returns:
            List of PresetMetadata for all presets
        """
        presets = []
        
        for preset_file in self.presets_dir.glob("*.json"):
            try:
                with open(preset_file, 'r') as f:
                    preset_data = json.load(f)
                
                if "metadata" in preset_data:
                    metadata = PresetMetadata.from_dict(preset_data["metadata"])
                    presets.append(metadata)
                    
            except (json.JSONDecodeError, KeyError, ValueError):
                # Skip invalid preset files
                continue
        
        return presets
    
    def delete_preset(self, name: str) -> bool:
        """
        Delete a preset file.
        
        Args:
            name: Name of the preset to delete
            
        Returns:
            True if deleted successfully, False if preset didn't exist
        """
        preset_path = self.presets_dir / f"{name}.json"
        
        if preset_path.exists():
            preset_path.unlink()
            return True
        
        return False
    
    def preset_exists(self, name: str) -> bool:
        """
        Check if a preset exists.
        
        Args:
            name: Name of the preset to check
            
        Returns:
            True if preset exists, False otherwise
        """
        preset_path = self.presets_dir / f"{name}.json"
        return preset_path.exists()