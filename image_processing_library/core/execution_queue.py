"""
Execution queue system for filter chaining and sequential processing.

This module provides the ExecutionQueue class for managing and executing
sequences of filters with progress tracking and intermediate saving capabilities.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable, Union, Type
import numpy as np
from .protocols import FilterProtocol
from .utils import FilterExecutionError


@dataclass
class FilterStep:
    """
    Represents a single step in the execution queue.
    
    This dataclass encapsulates all information needed to execute
    a filter as part of a processing pipeline, including the filter
    class/instance, parameters, and intermediate saving options.
    """
    filter_class: Union[Type[FilterProtocol], FilterProtocol]
    parameters: Dict[str, Any]
    save_intermediate: bool = False
    save_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate FilterStep after initialization."""
        if self.save_intermediate and not self.save_path:
            raise ValueError("save_path must be provided when save_intermediate is True")


class ExecutionQueue:
    """
    Manages filter execution queue with chaining and intermediate saving.
    
    The ExecutionQueue allows building complex processing pipelines by
    chaining multiple filters together. It provides progress tracking,
    error handling, and the ability to save intermediate results at
    specified steps in the pipeline.
    """
    
    def __init__(self):
        """Initialize an empty execution queue."""
        self.steps: List[FilterStep] = []
        self._progress_callback: Optional[Callable[[float, str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """
        Set callback function for overall progress updates.
        
        Args:
            callback: Function that accepts (progress, current_filter_name)
                     where progress is between 0.0 and 1.0
        """
        self._progress_callback = callback
    
    def add_filter(self, filter_class: Union[Type[FilterProtocol], FilterProtocol], 
                   parameters: Optional[Dict[str, Any]] = None,
                   save_intermediate: bool = False, 
                   save_path: Optional[str] = None) -> None:
        """
        Add a filter to the execution queue.
        
        Args:
            filter_class: Filter class or instance to add to the queue
            parameters: Parameters to pass to the filter (if filter_class is a class)
            save_intermediate: Whether to save the result of this filter step
            save_path: Path where to save intermediate result (required if save_intermediate=True)
            
        Raises:
            ValueError: If save_intermediate is True but save_path is not provided
        """
        if parameters is None:
            parameters = {}
            
        step = FilterStep(
            filter_class=filter_class,
            parameters=parameters,
            save_intermediate=save_intermediate,
            save_path=save_path
        )
        self.steps.append(step)
    
    def execute(self, input_data: np.ndarray) -> np.ndarray:
        """
        Execute all filters in the queue sequentially.
        
        Processes the input data through each filter in the queue in order,
        with progress tracking and optional intermediate saving.
        
        Args:
            input_data: Input numpy array to process
            
        Returns:
            Final processed numpy array after all filters
            
        Raises:
            FilterExecutionError: If any filter in the queue fails
            ValueError: If the queue is empty
        """
        if not self.steps:
            raise ValueError("Cannot execute empty queue")
        
        current_data = input_data.copy()
        total_steps = len(self.steps)
        
        for i, step in enumerate(self.steps):
            filter_instance = None
            try:
                # Create or use filter instance
                filter_instance = self._get_filter_instance(step)
                
                # Set up progress tracking for this step
                def step_progress(progress: float):
                    # Calculate overall progress: completed steps + current step progress
                    overall_progress = (i + progress) / total_steps
                    if self._progress_callback:
                        self._progress_callback(overall_progress, filter_instance.name)
                
                # Set progress callback if filter supports it
                if hasattr(filter_instance, 'set_progress_callback'):
                    filter_instance.set_progress_callback(step_progress)
                
                # Apply the filter
                current_data = filter_instance.apply(current_data)
                
                # Update progress to show step completion
                if self._progress_callback:
                    overall_progress = (i + 1) / total_steps
                    self._progress_callback(overall_progress, filter_instance.name)
                
                # Save intermediate result if requested
                if step.save_intermediate and step.save_path:
                    self._save_intermediate(current_data, step.save_path, filter_instance)
                    
            except Exception as e:
                filter_name = filter_instance.name if filter_instance else "unknown"
                raise FilterExecutionError(
                    f"Filter '{filter_name}' failed at step {i + 1}: {str(e)}"
                ) from e
        
        return current_data
    
    def _get_filter_instance(self, step: FilterStep) -> FilterProtocol:
        """
        Get a filter instance from a FilterStep.
        
        Args:
            step: FilterStep containing filter class or instance
            
        Returns:
            Filter instance ready for execution
            
        Raises:
            FilterExecutionError: If filter instance cannot be created
        """
        try:
            # Check if it's already an instance (has apply method and name attribute)
            if hasattr(step.filter_class, 'apply') and hasattr(step.filter_class, 'name'):
                filter_instance = step.filter_class
                # Update parameters if provided
                if step.parameters and hasattr(filter_instance, 'set_parameters'):
                    filter_instance.set_parameters(**step.parameters)
            else:
                # It's a class, instantiate it with parameters
                filter_instance = step.filter_class(**step.parameters)
            
            return filter_instance
            
        except Exception as e:
            filter_name = getattr(step.filter_class, 'name', str(step.filter_class))
            raise FilterExecutionError(
                f"Failed to create filter instance '{filter_name}': {str(e)}"
            ) from e
    
    def _save_intermediate(self, data: np.ndarray, path: str, filter_instance: FilterProtocol) -> None:
        """
        Save intermediate processing result.
        
        Provides basic saving functionality for common image formats.
        This implementation will be enhanced when full I/O modules are created in task 5.
        
        Args:
            data: Numpy array to save
            path: File path where to save the data
            filter_instance: Filter instance that produced this data
            
        Note:
            This method provides basic functionality and will be enhanced
            in task 5 when comprehensive I/O modules are implemented.
        """
        try:
            # Basic validation
            if not isinstance(data, np.ndarray):
                raise ValueError("Data must be a numpy array")
            
            if not path:
                raise ValueError("Save path cannot be empty")
            
            # Import required libraries for basic saving
            import os
            from pathlib import Path
            
            # Create directory if it doesn't exist
            save_dir = Path(path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine file format from extension
            file_ext = Path(path).suffix.lower()
            
            if file_ext in ['.npy']:
                # Save as numpy array
                np.save(path, data)
                
            elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                # Try to save as image using PIL if available
                try:
                    from PIL import Image
                    
                    # Prepare data for PIL
                    if data.dtype != np.uint8:
                        # Convert to uint8 if needed
                        if data.max() <= 1.0:
                            # Assume normalized data [0, 1]
                            save_data = (data * 255).astype(np.uint8)
                        else:
                            # Clip and convert
                            save_data = np.clip(data, 0, 255).astype(np.uint8)
                    else:
                        save_data = data
                    
                    # Handle different array shapes
                    if save_data.ndim == 2:
                        # Grayscale
                        image = Image.fromarray(save_data, mode='L')
                    elif save_data.ndim == 3:
                        if save_data.shape[2] == 3:
                            # RGB
                            image = Image.fromarray(save_data, mode='RGB')
                        elif save_data.shape[2] == 4:
                            # RGBA
                            image = Image.fromarray(save_data, mode='RGBA')
                        else:
                            raise ValueError(f"Unsupported number of channels: {save_data.shape[2]}")
                    else:
                        raise ValueError(f"Unsupported array dimensions: {save_data.ndim}")
                    
                    image.save(path)
                    
                except ImportError:
                    # PIL not available, fall back to numpy save
                    np_path = str(Path(path).with_suffix('.npy'))
                    np.save(np_path, data)
                    print(f"Warning: PIL not available, saved as numpy array: {np_path}")
                    
            else:
                # Default to numpy save for unknown formats
                np_path = str(Path(path).with_suffix('.npy'))
                np.save(np_path, data)
                print(f"Warning: Unknown format {file_ext}, saved as numpy array: {np_path}")
            
            print(f"Intermediate result saved: {path} (shape: {data.shape}, filter: {filter_instance.name})")
            
        except Exception as e:
            # Don't fail the entire pipeline for intermediate save errors
            print(f"Warning: Failed to save intermediate result to {path}: {str(e)}")
    
    def clear(self) -> None:
        """Clear all steps from the execution queue."""
        self.steps.clear()
    
    def get_step_count(self) -> int:
        """
        Get the number of steps in the queue.
        
        Returns:
            Number of filter steps in the queue
        """
        return len(self.steps)
    
    def get_step_info(self, index: int) -> Dict[str, Any]:
        """
        Get information about a specific step in the queue.
        
        Args:
            index: Index of the step to get information about
            
        Returns:
            Dictionary containing step information
            
        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.steps):
            raise IndexError(f"Step index {index} out of range (0-{len(self.steps)-1})")
        
        step = self.steps[index]
        
        # Try to get filter name from instance or class
        if hasattr(step.filter_class, 'name'):
            filter_name = step.filter_class.name
        elif hasattr(step.filter_class, '__name__'):
            filter_name = step.filter_class.__name__
        else:
            filter_name = str(step.filter_class)
        
        return {
            'index': index,
            'filter_name': filter_name,
            'parameters': step.parameters.copy(),
            'save_intermediate': step.save_intermediate,
            'save_path': step.save_path
        }
    
    def remove_step(self, index: int) -> None:
        """
        Remove a step from the execution queue.
        
        Args:
            index: Index of the step to remove
            
        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.steps):
            raise IndexError(f"Step index {index} out of range (0-{len(self.steps)-1})")
        
        del self.steps[index]