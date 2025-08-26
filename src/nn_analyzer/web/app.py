"""FastAPI web application for neural network analyzer."""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional, List, Dict, Any
import json
import os
from pathlib import Path

from ..core.analyzer import NetworkAnalyzer
from ..core.layers import create_layer, LAYER_TYPES
from ..core.diagram_generator import DiagramGenerator

# Get the project root directory
project_root = Path(__file__).parent.parent.parent.parent
templates_dir = project_root / "templates"
static_dir = project_root / "static"

app = FastAPI(title="Neural Network Analyzer", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(templates_dir))

# Global analyzer instance (in production, you'd use sessions or database)
analyzer = NetworkAnalyzer((224, 224, 3))  # Default input shape
diagram_generator = DiagramGenerator()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with the neural network builder interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "layer_types": list(LAYER_TYPES.keys()),
        "current_layers": analyzer.get_layer_details(),
        "summary": analyzer.get_analysis_summary()
    })


@app.post("/set_input_shape")
async def set_input_shape(
    height: int = Form(...),
    width: int = Form(...), 
    channels: int = Form(...)
):
    """Set the input shape for the network."""
    global analyzer
    try:
        analyzer = NetworkAnalyzer((height, width, channels))
        return JSONResponse({
            "status": "success",
            "message": f"Input shape set to {height}×{width}×{channels}",
            "summary": analyzer.get_analysis_summary()
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/add_layer")
async def add_layer(
    layer_type: str = Form(...),
    # Conv2D parameters
    filters: Optional[int] = Form(None),
    kernel_size: Optional[int] = Form(None),
    stride: Optional[int] = Form(1),
    padding: Optional[str] = Form("valid"),
    activation: Optional[str] = Form("relu"),
    # Pooling parameters
    pool_size: Optional[int] = Form(None),
    # Dropout parameters
    rate: Optional[float] = Form(0.5),
    # Dense parameters
    units: Optional[int] = Form(None),
    use_bias: Optional[bool] = Form(True),
    # General parameters
    name: Optional[str] = Form(None)
):
    """Add a layer to the network."""
    try:
        layer_params = {}
        
        if layer_type.lower() == 'conv2d':
            if not filters or not kernel_size:
                raise ValueError("Conv2D layer requires filters and kernel_size")
            layer_params = {
                'filters': filters,
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'activation': activation,
                'name': name
            }
        elif layer_type.lower() in ['maxpool2d', 'avgpool2d']:
            if not pool_size:
                raise ValueError(f"{layer_type} layer requires pool_size")
            layer_params = {
                'pool_size': pool_size,
                'stride': stride,
                'padding': padding,
                'name': name
            }
        elif layer_type.lower() == 'dropout':
            layer_params = {
                'rate': rate,
                'name': name
            }
        elif layer_type.lower() == 'dense':
            if not units:
                raise ValueError("Dense layer requires units")
            layer_params = {
                'units': units,
                'activation': activation,
                'use_bias': use_bias,
                'name': name
            }
        elif layer_type.lower() == 'activation':
            layer_params = {
                'activation': activation,
                'name': name
            }
        elif layer_type.lower() in ['batchnorm', 'globalavgpool2d', 'flatten']:
            layer_params = {'name': name}
        
        # Filter out None values
        layer_params = {k: v for k, v in layer_params.items() if v is not None}
        
        layer = create_layer(layer_type, **layer_params)
        analyzer.add_layer(layer)
        
        return JSONResponse({
            "status": "success",
            "message": f"{layer_type} layer added successfully",
            "layers": analyzer.get_layer_details(),
            "summary": analyzer.get_analysis_summary()
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/remove_layer/{layer_index}")
async def remove_layer(layer_index: int):
    """Remove a layer from the network."""
    try:
        analyzer.remove_layer(layer_index)
        return JSONResponse({
            "status": "success",
            "message": "Layer removed successfully",
            "layers": analyzer.get_layer_details(),
            "summary": analyzer.get_analysis_summary()
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/clear_layers")
async def clear_layers():
    """Clear all layers from the network."""
    try:
        analyzer.clear_layers()
        return JSONResponse({
            "status": "success",
            "message": "All layers cleared",
            "layers": [],
            "summary": analyzer.get_analysis_summary()
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/get_analysis")
async def get_analysis():
    """Get the current network analysis."""
    return JSONResponse({
        "layers": analyzer.get_layer_details(),
        "summary": analyzer.get_analysis_summary(),
        "input_shape": analyzer.input_shape
    })


@app.post("/generate_diagram")
async def generate_diagram():
    """Generate and return the network architecture diagram."""
    try:
        if not analyzer.layers:
            raise ValueError("No layers added to the network")
        
        # Generate diagram
        image_base64 = diagram_generator.generate_diagram(analyzer)
        
        # Generate detailed table
        table_html = diagram_generator.generate_detailed_table(analyzer)
        
        return JSONResponse({
            "status": "success",
            "image": image_base64,
            "table": table_html,
            "summary": analyzer.get_analysis_summary()
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/export_architecture")
async def export_architecture():
    """Export the complete architecture configuration."""
    try:
        architecture = analyzer.export_architecture()
        return JSONResponse({
            "status": "success",
            "architecture": architecture
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/layer_types")
async def get_layer_types():
    """Get available layer types and their required parameters."""
    layer_info = {
        'conv2d': {
            'name': 'Convolutional 2D',
            'category': 'Feature Extraction',
            'required': ['filters', 'kernel_size'],
            'optional': ['stride', 'padding', 'activation', 'name']
        },
        'maxpool2d': {
            'name': 'Max Pooling 2D',
            'category': 'Feature Extraction',
            'required': ['pool_size'],
            'optional': ['stride', 'padding', 'name']
        },
        'avgpool2d': {
            'name': 'Average Pooling 2D',
            'category': 'Feature Extraction',
            'required': ['pool_size'],
            'optional': ['stride', 'padding', 'name']
        },
        'batchnorm': {
            'name': 'Batch Normalization',
            'category': 'Regularization',
            'required': [],
            'optional': ['name']
        },
        'dropout': {
            'name': 'Dropout',
            'category': 'Regularization',
            'required': [],
            'optional': ['rate', 'name']
        },
        'globalavgpool2d': {
            'name': 'Global Average Pooling 2D',
            'category': 'Feature Extraction',
            'required': [],
            'optional': ['name']
        },
        'flatten': {
            'name': 'Flatten',
            'category': 'Transition',
            'required': [],
            'optional': ['name']
        },
        'dense': {
            'name': 'Dense/Linear',
            'category': 'Output',
            'required': ['units'],
            'optional': ['activation', 'use_bias', 'name']
        },
        'activation': {
            'name': 'Activation',
            'category': 'Activation',
            'required': ['activation'],
            'optional': ['name']
        }
    }
    
    return JSONResponse(layer_info)


@app.post("/configure_problem")
async def configure_problem(
    problem_type: str = Form(...),
    num_classes: Optional[int] = Form(None),
    output_activation: Optional[str] = Form("softmax")
):
    """Configure the problem type and add appropriate output layers."""
    try:
        layers_to_add = []
        
        if problem_type == "classification":
            if not num_classes:
                raise ValueError("Classification requires number of classes")
            
            # Add output layer for classification
            layers_to_add.append({
                'type': 'dense',
                'params': {
                    'units': num_classes,
                    'activation': output_activation,
                    'name': f'Classification_Output_{num_classes}'
                }
            })
            
        elif problem_type == "regression":
            output_units = num_classes if num_classes else 1
            layers_to_add.append({
                'type': 'dense',
                'params': {
                    'units': output_units,
                    'activation': 'linear',
                    'name': f'Regression_Output_{output_units}'
                }
            })
            
        elif problem_type == "binary_classification":
            layers_to_add.append({
                'type': 'dense',
                'params': {
                    'units': 1,
                    'activation': 'sigmoid',
                    'name': 'Binary_Classification_Output'
                }
            })
        
        # Add the layers to the network
        for layer_config in layers_to_add:
            layer = create_layer(layer_config['type'], **layer_config['params'])
            analyzer.add_layer(layer)
        
        return JSONResponse({
            "status": "success",
            "message": f"Configured for {problem_type}",
            "layers_added": len(layers_to_add),
            "layers": analyzer.get_layer_details(),
            "summary": analyzer.get_analysis_summary()
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/problem_types")
async def get_problem_types():
    """Get available problem types."""
    problem_types = {
        'classification': {
            'name': 'Multi-class Classification',
            'description': 'Classify input into one of multiple classes',
            'requires_classes': True,
            'default_activation': 'softmax'
        },
        'binary_classification': {
            'name': 'Binary Classification',
            'description': 'Classify input into one of two classes',
            'requires_classes': False,
            'default_activation': 'sigmoid'
        },
        'regression': {
            'name': 'Regression',
            'description': 'Predict continuous numerical values',
            'requires_classes': False,
            'default_activation': 'linear'
        }
    }
    
    return JSONResponse(problem_types)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
