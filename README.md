# Neural Network Analyzer

A powerful web-based tool for analyzing Convolutional Neural Network (CNN) architectures. This application helps you understand the structure of your neural networks by calculating receptive fields, parameter counts, and visualizing the architecture.

## Features

- 🧠 **Layer Analysis**: Support for Conv2D, MaxPool2D, AvgPool2D, BatchNorm, Dropout, GlobalAvgPool2D, Flatten, Dense, and Activation layers
- 📊 **Parameter Calculation**: Automatic calculation of trainable parameters for each layer
- 🔍 **Receptive Field Calculation**: Accurate receptive field computation throughout the network
- 📐 **Shape Tracking**: Track input/output shapes through the entire network
- 🎨 **Architecture Visualization**: Generate beautiful architecture diagrams
- 📋 **Detailed Analysis**: Comprehensive layer-by-layer analysis tables
- 💾 **Export Functionality**: Export architecture configurations as JSON
- 🎯 **Problem Type Support**: One-click configuration for classification, regression, and binary classification
- 🧠 **Professional Branding**: Custom brain favicon and app manifest for web app experience
- 📱 **Compact Interface**: Optimized left panel with collapsible sections that fits without scrolling

## Installation

This project uses `uv` for dependency management. Make sure you have `uv` installed.

```bash
# Clone or navigate to the project directory
cd nn_analyzer

# Install dependencies (uv will create a virtual environment automatically)
uv sync

# Run the application
python main.py
```

## Usage

### Starting the Application

```bash
python main.py
```

The application will start on `http://localhost:8000`. Open this URL in your web browser.

### Using the Web Interface

1. **Set Input Shape**: 
   - Enter the height, width, and number of channels for your input images
   - Default is 224×224×3 (typical for ImageNet-style inputs)

2. **Add Layers**:
   - Select a layer type from the dropdown
   - Configure layer parameters as needed
   - Click "Add Layer" to add it to your network

3. **Layer Types Available**:
   
   **Feature Extraction:**
   - **Convolutional 2D**: Specify filters, kernel size, stride, padding, and activation
   - **Max Pooling 2D**: Specify pool size, stride, and padding
   - **Average Pooling 2D**: Specify pool size, stride, and padding
   - **Global Average Pooling 2D**: No additional parameters needed
   
   **Regularization:**
   - **Batch Normalization**: No additional parameters needed
   - **Dropout**: Specify dropout rate (0.0 to 1.0)
   
   **Transition:**
   - **Flatten**: Convert 2D feature maps to 1D vector
   
   **Output:**
   - **Dense/Linear**: Specify units, activation function, and bias usage
   
   **Activation:**
   - **Activation**: Standalone activation functions (ReLU, Sigmoid, Tanh, Softmax, etc.)

4. **Configure Problem Type** (NEW 🎯):
   - **Multi-class Classification**: Specify number of classes (e.g., 10 for CIFAR-10)
   - **Binary Classification**: Automatic single sigmoid output
   - **Regression**: Single or multiple continuous outputs
   - Click "Configure Output" to automatically add appropriate final layers

5. **View Analysis**:
   - Real-time updates of network summary (layers, parameters, shapes, receptive field)
   - Detailed layer-by-layer breakdown
   - Remove individual layers or clear all layers

6. **Generate Architecture Diagram**:
   - Click "Generate Architecture Diagram" to create a visual representation
   - View detailed analysis table with all layer information

7. **Export Configuration**:
   - Click "Export" to download the architecture configuration as JSON

### Example Networks

#### Classification Network (CIFAR-10)

1. Set input shape: 32×32×3
2. Add Conv2D: 32 filters, 3×3 kernel, stride=1, padding=same
3. Add MaxPool2D: 2×2 pool size, stride=2
4. Add Conv2D: 64 filters, 3×3 kernel, stride=1, padding=same
5. Add BatchNorm2D
6. Add MaxPool2D: 2×2 pool size, stride=2
7. Add Flatten
8. Add Dense: 128 units, ReLU activation
9. Add Dropout: rate=0.5
10. **Configure Problem Type**: Select "Multi-class Classification", set classes=10
    - This automatically adds: Dense layer with 10 units and softmax activation

Final network: ~25M parameters, receptive field of 6

#### Regression Network

1. Set input shape: 64×64×1
2. Add Conv2D: 16 filters, 5×5 kernel, stride=2, padding=same
3. Add Conv2D: 32 filters, 3×3 kernel, stride=2, padding=same
4. Add GlobalAvgPool2D
5. Add Flatten
6. **Configure Problem Type**: Select "Regression"
    - This automatically adds: Dense layer with 1 unit and linear activation

Final network: ~5K parameters for continuous value prediction

## Architecture Analysis

The application calculates several important metrics:

### Receptive Field
The receptive field shows how much of the input image each neuron can "see". This is crucial for understanding what level of features your network can detect.

### Parameter Count
Accurate parameter counting helps you understand model complexity and memory requirements:
- **Conv2D**: `(kernel_size × kernel_size × input_channels + 1) × filters`
- **BatchNorm**: `2 × channels` (scale and shift parameters)
- **Other layers**: 0 parameters

### Shape Calculation
Tracks how tensor dimensions change through the network, accounting for:
- Padding (same vs valid)
- Stride effects
- Pooling operations

## Project Structure

```
nn_analyzer/
├── src/nn_analyzer/
│   ├── core/
│   │   ├── analyzer.py      # Main network analyzer
│   │   ├── layers.py        # Layer definitions and calculations
│   │   └── diagram_generator.py  # Visualization generation
│   └── web/
│       └── app.py           # FastAPI web application
├── templates/
│   └── index.html           # Main web interface
├── static/
│   ├── css/style.css        # Styling
│   └── js/app.js            # Frontend JavaScript
├── main.py                  # Application entry point
├── test_analyzer.py         # Test script
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Development

### Running Tests

```bash
python test_analyzer.py
```

### Adding New Layer Types

1. Create a new layer class in `src/nn_analyzer/core/layers.py` that inherits from `Layer`
2. Implement the required methods: `calculate_output_shape`, `calculate_parameters`, `calculate_receptive_field`, `get_layer_info`
3. Add the layer to the `LAYER_TYPES` dictionary
4. Update the web interface in `templates/index.html` and `static/js/app.js` to include the new layer type

## Technologies Used

- **Backend**: FastAPI, Python 3.13+
- **Frontend**: HTML5, Bootstrap 5, JavaScript
- **Visualization**: Matplotlib
- **Package Management**: uv
- **Image Processing**: Pillow

## License

This project is open source. Feel free to use, modify, and distribute as needed.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
