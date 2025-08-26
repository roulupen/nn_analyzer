"""Neural network layer definitions and calculations."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import math


class Layer(ABC):
    """Abstract base class for neural network layers."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate output shape given input shape (H, W, C)."""
        pass
    
    @abstractmethod
    def calculate_parameters(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate number of parameters."""
        pass
    
    @abstractmethod
    def calculate_receptive_field(self, input_rf: int, input_stride: int) -> Tuple[int, int]:
        """Calculate receptive field and effective stride."""
        pass
    
    @abstractmethod
    def get_layer_info(self) -> Dict[str, Any]:
        """Get layer configuration information."""
        pass


class Conv2DLayer(Layer):
    """Convolutional 2D layer."""
    
    def __init__(self, filters: int, kernel_size: int, stride: int = 1, 
                 padding: str = 'valid', activation: str = 'relu', name: str = None):
        super().__init__(name or f"Conv2D_{filters}_{kernel_size}x{kernel_size}")
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
    
    def calculate_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        h, w, c = input_shape
        
        if self.padding == 'same':
            out_h = math.ceil(h / self.stride)
            out_w = math.ceil(w / self.stride)
        else:  # valid padding
            out_h = math.ceil((h - self.kernel_size + 1) / self.stride)
            out_w = math.ceil((w - self.kernel_size + 1) / self.stride)
        
        return (out_h, out_w, self.filters)
    
    def calculate_parameters(self, input_shape: Tuple[int, int, int]) -> int:
        _, _, input_channels = input_shape
        # (kernel_size * kernel_size * input_channels + 1) * filters
        return (self.kernel_size * self.kernel_size * input_channels + 1) * self.filters
    
    def calculate_receptive_field(self, input_rf: int, input_stride: int) -> Tuple[int, int]:
        rf = input_rf + (self.kernel_size - 1) * input_stride
        stride = input_stride * self.stride
        return rf, stride
    
    def get_layer_info(self) -> Dict[str, Any]:
        return {
            'type': 'Conv2D',
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'activation': self.activation
        }


class MaxPool2DLayer(Layer):
    """Max pooling 2D layer."""
    
    def __init__(self, pool_size: int, stride: int = None, padding: str = 'valid', name: str = None):
        super().__init__(name or f"MaxPool2D_{pool_size}x{pool_size}")
        self.pool_size = pool_size
        self.stride = stride or pool_size
        self.padding = padding
    
    def calculate_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        h, w, c = input_shape
        
        if self.padding == 'same':
            out_h = math.ceil(h / self.stride)
            out_w = math.ceil(w / self.stride)
        else:  # valid padding
            out_h = math.ceil((h - self.pool_size + 1) / self.stride)
            out_w = math.ceil((w - self.pool_size + 1) / self.stride)
        
        return (out_h, out_w, c)
    
    def calculate_parameters(self, input_shape: Tuple[int, int, int]) -> int:
        return 0  # No parameters in pooling layers
    
    def calculate_receptive_field(self, input_rf: int, input_stride: int) -> Tuple[int, int]:
        rf = input_rf + (self.pool_size - 1) * input_stride
        stride = input_stride * self.stride
        return rf, stride
    
    def get_layer_info(self) -> Dict[str, Any]:
        return {
            'type': 'MaxPool2D',
            'pool_size': self.pool_size,
            'stride': self.stride,
            'padding': self.padding
        }


class AvgPool2DLayer(Layer):
    """Average pooling 2D layer."""
    
    def __init__(self, pool_size: int, stride: int = None, padding: str = 'valid', name: str = None):
        super().__init__(name or f"AvgPool2D_{pool_size}x{pool_size}")
        self.pool_size = pool_size
        self.stride = stride or pool_size
        self.padding = padding
    
    def calculate_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        h, w, c = input_shape
        
        if self.padding == 'same':
            out_h = math.ceil(h / self.stride)
            out_w = math.ceil(w / self.stride)
        else:  # valid padding
            out_h = math.ceil((h - self.pool_size + 1) / self.stride)
            out_w = math.ceil((w - self.pool_size + 1) / self.stride)
        
        return (out_h, out_w, c)
    
    def calculate_parameters(self, input_shape: Tuple[int, int, int]) -> int:
        return 0  # No parameters in pooling layers
    
    def calculate_receptive_field(self, input_rf: int, input_stride: int) -> Tuple[int, int]:
        rf = input_rf + (self.pool_size - 1) * input_stride
        stride = input_stride * self.stride
        return rf, stride
    
    def get_layer_info(self) -> Dict[str, Any]:
        return {
            'type': 'AvgPool2D',
            'pool_size': self.pool_size,
            'stride': self.stride,
            'padding': self.padding
        }


class BatchNormLayer(Layer):
    """Batch normalization layer."""
    
    def __init__(self, name: str = None):
        super().__init__(name or "BatchNorm2D")
    
    def calculate_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return input_shape  # Same shape as input
    
    def calculate_parameters(self, input_shape: Tuple[int, int, int]) -> int:
        _, _, channels = input_shape
        # 2 parameters per channel (scale and shift)
        return 2 * channels
    
    def calculate_receptive_field(self, input_rf: int, input_stride: int) -> Tuple[int, int]:
        return input_rf, input_stride  # No change in RF
    
    def get_layer_info(self) -> Dict[str, Any]:
        return {
            'type': 'BatchNorm2D'
        }


class DropoutLayer(Layer):
    """Dropout layer."""
    
    def __init__(self, rate: float = 0.5, name: str = None):
        super().__init__(name or f"Dropout_{rate}")
        self.rate = rate
    
    def calculate_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return input_shape  # Same shape as input
    
    def calculate_parameters(self, input_shape: Tuple[int, int, int]) -> int:
        return 0  # No parameters
    
    def calculate_receptive_field(self, input_rf: int, input_stride: int) -> Tuple[int, int]:
        return input_rf, input_stride  # No change in RF
    
    def get_layer_info(self) -> Dict[str, Any]:
        return {
            'type': 'Dropout',
            'rate': self.rate
        }


class GlobalAvgPool2DLayer(Layer):
    """Global average pooling 2D layer."""
    
    def __init__(self, name: str = None):
        super().__init__(name or "GlobalAvgPool2D")
    
    def calculate_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        _, _, c = input_shape
        return (1, 1, c)
    
    def calculate_parameters(self, input_shape: Tuple[int, int, int]) -> int:
        return 0  # No parameters
    
    def calculate_receptive_field(self, input_rf: int, input_stride: int) -> Tuple[int, int]:
        # Global pooling sees the entire feature map
        return float('inf'), input_stride
    
    def get_layer_info(self) -> Dict[str, Any]:
        return {
            'type': 'GlobalAvgPool2D'
        }


class FlattenLayer(Layer):
    """Flatten layer to convert 2D feature maps to 1D."""
    
    def __init__(self, name: str = None):
        super().__init__(name or "Flatten")
    
    def calculate_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        h, w, c = input_shape
        flattened_size = h * w * c
        return (1, 1, flattened_size)
    
    def calculate_parameters(self, input_shape: Tuple[int, int, int]) -> int:
        return 0  # No parameters
    
    def calculate_receptive_field(self, input_rf: int, input_stride: int) -> Tuple[int, int]:
        return input_rf, input_stride  # No change in RF
    
    def get_layer_info(self) -> Dict[str, Any]:
        return {
            'type': 'Flatten'
        }


class DenseLayer(Layer):
    """Dense/Linear layer for classification and regression."""
    
    def __init__(self, units: int, activation: str = 'relu', use_bias: bool = True, name: str = None):
        super().__init__(name or f"Dense_{units}")
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
    
    def calculate_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (1, 1, self.units)
    
    def calculate_parameters(self, input_shape: Tuple[int, int, int]) -> int:
        h, w, c = input_shape
        input_size = h * w * c
        # weights + bias (if used)
        params = input_size * self.units
        if self.use_bias:
            params += self.units
        return params
    
    def calculate_receptive_field(self, input_rf: int, input_stride: int) -> Tuple[int, int]:
        return input_rf, input_stride  # No change in RF
    
    def get_layer_info(self) -> Dict[str, Any]:
        return {
            'type': 'Dense',
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias
        }


class ActivationLayer(Layer):
    """Standalone activation layer."""
    
    def __init__(self, activation: str = 'relu', name: str = None):
        super().__init__(name or f"Activation_{activation}")
        self.activation = activation
    
    def calculate_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return input_shape  # Same shape as input
    
    def calculate_parameters(self, input_shape: Tuple[int, int, int]) -> int:
        return 0  # No parameters
    
    def calculate_receptive_field(self, input_rf: int, input_stride: int) -> Tuple[int, int]:
        return input_rf, input_stride  # No change in RF
    
    def get_layer_info(self) -> Dict[str, Any]:
        return {
            'type': 'Activation',
            'activation': self.activation
        }


# Layer factory for creating layers from configuration
LAYER_TYPES = {
    'conv2d': Conv2DLayer,
    'maxpool2d': MaxPool2DLayer,
    'avgpool2d': AvgPool2DLayer,
    'batchnorm': BatchNormLayer,
    'dropout': DropoutLayer,
    'globalavgpool2d': GlobalAvgPool2DLayer,
    'flatten': FlattenLayer,
    'dense': DenseLayer,
    'activation': ActivationLayer,
}


def create_layer(layer_type: str, **kwargs) -> Layer:
    """Create a layer instance from type and parameters."""
    if layer_type.lower() not in LAYER_TYPES:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    layer_class = LAYER_TYPES[layer_type.lower()]
    return layer_class(**kwargs)
