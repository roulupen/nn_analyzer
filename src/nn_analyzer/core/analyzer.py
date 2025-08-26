"""Neural network architecture analyzer."""

from typing import List, Tuple, Dict, Any
from .layers import Layer


class NetworkAnalyzer:
    """Analyzes neural network architectures."""
    
    def __init__(self, input_shape: Tuple[int, int, int]):
        """
        Initialize with input shape.
        
        Args:
            input_shape: Input shape as (height, width, channels)
        """
        self.input_shape = input_shape
        self.layers: List[Layer] = []
        self.analysis_results: List[Dict[str, Any]] = []
    
    def add_layer(self, layer: Layer) -> None:
        """Add a layer to the network."""
        self.layers.append(layer)
        self._update_analysis()
    
    def remove_layer(self, index: int) -> None:
        """Remove a layer at the given index."""
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
            self._update_analysis()
    
    def clear_layers(self) -> None:
        """Clear all layers."""
        self.layers.clear()
        self.analysis_results.clear()
    
    def _update_analysis(self) -> None:
        """Update the analysis results for all layers."""
        self.analysis_results.clear()
        
        if not self.layers:
            return
        
        current_shape = self.input_shape
        current_rf = 1
        current_stride = 1
        
        for i, layer in enumerate(self.layers):
            # Calculate output shape
            output_shape = layer.calculate_output_shape(current_shape)
            
            # Calculate parameters
            parameters = layer.calculate_parameters(current_shape)
            
            # Calculate receptive field
            rf, stride = layer.calculate_receptive_field(current_rf, current_stride)
            
            # Store results
            result = {
                'layer_index': i,
                'layer_name': layer.name,
                'layer_info': layer.get_layer_info(),
                'input_shape': current_shape,
                'output_shape': output_shape,
                'parameters': parameters,
                'receptive_field': rf if rf != float('inf') else 'Global',
                'effective_stride': stride
            }
            
            self.analysis_results.append(result)
            
            # Update for next iteration
            current_shape = output_shape
            current_rf = rf if rf != float('inf') else current_rf
            current_stride = stride
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters in the network."""
        return sum(result['parameters'] for result in self.analysis_results)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the network analysis."""
        if not self.analysis_results:
            return {
                'input_shape': self.input_shape,
                'output_shape': self.input_shape,
                'total_parameters': 0,
                'total_layers': 0,
                'final_receptive_field': 1
            }
        
        final_result = self.analysis_results[-1]
        
        return {
            'input_shape': self.input_shape,
            'output_shape': final_result['output_shape'],
            'total_parameters': self.get_total_parameters(),
            'total_layers': len(self.layers),
            'final_receptive_field': final_result['receptive_field']
        }
    
    def get_layer_details(self) -> List[Dict[str, Any]]:
        """Get detailed analysis for each layer."""
        return self.analysis_results.copy()
    
    def export_architecture(self) -> Dict[str, Any]:
        """Export the complete architecture configuration."""
        return {
            'input_shape': self.input_shape,
            'layers': [
                {
                    'name': layer.name,
                    'config': layer.get_layer_info()
                }
                for layer in self.layers
            ],
            'analysis': self.get_analysis_summary(),
            'layer_details': self.get_layer_details()
        }
