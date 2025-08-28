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

## Running on AWS EC2 (Ubuntu)

This section provides step-by-step instructions for deploying the Neural Network Analyzer on an AWS EC2 instance running Ubuntu.

### 1. Create an EC2 Instance

1. Log into your AWS Console
2. Navigate to EC2 service
3. Click "Launch Instance"
4. Choose "Ubuntu Server 22.04 LTS" (or latest LTS version)
5. Select an appropriate instance type (t2.micro for testing, t2.small or larger for production)
6. Configure security groups (we'll add inbound rules later)
7. Launch the instance and download your `.pem` key file

### 2. Connect to Your EC2 Instance

```bash
# Save the .pem file in your working directory
# Make sure the key file has the correct permissions
chmod 400 aws-key.pem

# Connect to your EC2 instance
ssh -i /path/to/your-key.pem ubuntu@<EC2-Public-IP-or-DNS>
```

**Note:** Replace `/path/to/your-key.pem` with the actual path to your key file and `<EC2-Public-IP-or-DNS>` with your instance's public IP address or DNS name.

### 3. Update Ubuntu System

```bash
# Update package lists
sudo apt update

# Upgrade all installed packages
sudo apt upgrade -y
```

### 4. Install Python and uv

```bash
# Install Python 3+ (required by this project)
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3 python3-venv python3-dev

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart shell or source profile
source ~/.bashrc

# Verify uv installation
uv --version
```

### 5. Clone and Setup Your Project

```bash
# Clone your repository (if using git)
git clone <your-repository-url>
cd nn_analyzer

# Or upload your project files using scp
# From your local machine:
# scp -i /path/to/your-key.pem -r /path/to/your/project ubuntu@<EC2-Public-IP>:/home/ubuntu/
```

### 6. Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies using uv
uv sync
```

### 7. Run the Application

```bash
# Use uv run for direct execution
uv run python main.py
```

The application will start and listen on port 8000.

### 8. Configure EC2 Security Group

1. Go to EC2 Dashboard → Security Groups
2. Select the security group associated with your instance
3. Click "Edit inbound rules"
4. Add a new rule:
   - **Type**: Custom TCP
   - **Port**: 8000
   - **Source**: 0.0.0.0/0 (for public access) or your IP address for restricted access
   - **Description**: Neural Network Analyzer Web App
5. Click "Save rules"

### 9. Access Your Application

Open your web browser and navigate to:
```
http://<EC2-Public-IP>:8000
```

**Note:** Replace `<EC2-Public-IP>` with your actual EC2 instance's public IP address.

### 10. Running in Background (Optional)

To keep the application running after you disconnect from SSH:

```bash
# Use nohup to run in background
nohup uv run python main.py > app.log 2>&1 &

# Or use screen/tmux for better process management
sudo apt install screen
screen -S nn-analyzer
uv run python main.py
# Press Ctrl+A, then D to detach
# Use 'screen -r nn-analyzer' to reattach
```

### Security Considerations

- **Restrict Access**: Consider limiting the security group to only your IP address instead of 0.0.0.0/0
- **HTTPS**: For production use, consider setting up HTTPS with a domain and SSL certificate
- **Firewall**: Ensure your EC2 instance's firewall allows traffic on port 8000
- **Updates**: Regularly update your Ubuntu system and Python packages

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
