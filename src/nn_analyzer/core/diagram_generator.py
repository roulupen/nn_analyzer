"""Generate neural network architecture diagrams."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import List, Dict, Any, Tuple
import io
import base64
from .analyzer import NetworkAnalyzer


class DiagramGenerator:
    """Generates visual diagrams of neural network architectures."""
    
    def __init__(self):
        self.colors = {
            'Conv2D': '#FF6B6B',
            'MaxPool2D': '#4ECDC4',
            'AvgPool2D': '#45B7D1',
            'BatchNorm2D': '#96CEB4',
            'Dropout': '#FFEAA7',
            'GlobalAvgPool2D': '#DDA0DD',
            'Flatten': '#FF9F43',
            'Dense': '#6C5CE7',
            'Activation': '#A29BFE',
            'input': '#74B9FF',
            'output': '#FD79A8'
        }
        
        self.box_height = 1.0
        self.box_spacing = 2.0
        self.text_size = 10
    
    def generate_diagram(self, analyzer: NetworkAnalyzer, save_path: str = None) -> str:
        """
        Generate a diagram of the neural network architecture.
        
        Args:
            analyzer: NetworkAnalyzer instance with layers
            save_path: Optional path to save the diagram
            
        Returns:
            Base64 encoded image string
        """
        # Clear any existing plots
        plt.clf()
        
        # Create figure and axis with improved dimensions
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Get analysis results
        layer_details = analyzer.get_layer_details()
        summary = analyzer.get_analysis_summary()
        
        # Calculate total width needed
        total_layers = len(layer_details) + 2  # +2 for input and output
        total_width = total_layers * self.box_spacing
        
        # Set up the plot with better spacing
        ax.set_xlim(-1, total_width + 1)
        ax.set_ylim(-3, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Set background color
        fig.patch.set_facecolor('white')
        
        # Draw input
        self._draw_layer_visual(ax, 0, analyzer.input_shape, 'Input', 'input', 0)
        
        # Draw layers
        for i, layer_detail in enumerate(layer_details):
            x_pos = (i + 1) * self.box_spacing
            layer_info = layer_detail['layer_info']
            layer_type = layer_info['type']
            
            # Create layer label
            layer_label = self._create_layer_label(layer_detail)
            
            self._draw_layer_visual(
                ax, x_pos, layer_detail['output_shape'], 
                layer_label, layer_type, layer_detail['parameters']
            )
            
            # Draw connection arrow with improved styling
            self._draw_connection_arrow(ax, x_pos - self.box_spacing + 0.8, x_pos - 0.8)
        
        # Draw output
        if layer_details:
            final_x = len(layer_details) * self.box_spacing
            final_shape = layer_details[-1]['output_shape']
            self._draw_layer_visual(ax, final_x + self.box_spacing, final_shape, 'Output', 'output', 0)
            self._draw_connection_arrow(ax, final_x + 0.8, final_x + self.box_spacing - 0.8)
        
        # Add title and summary
        title = f"Neural Network Architecture\nTotal Parameters: {summary['total_parameters']:,}"
        if summary['final_receptive_field'] != 'Global':
            title += f" | Final RF: {summary['final_receptive_field']}"
        else:
            title += f" | Final RF: Global"
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        self._add_legend(ax, total_width)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        
        plt.close()
        
        return image_base64
    
    def _draw_layer_visual(self, ax, x: float, shape: Tuple[int, int, int], 
                          label: str, layer_type: str, parameters: int):
        """Draw a sophisticated visual representation of a layer."""
        h, w, c = shape
        
        # Get color for layer type
        color = self.colors.get(layer_type, '#CCCCCC')
        
        if layer_type in ['Conv2D', 'MaxPool2D', 'AvgPool2D', 'GlobalAvgPool2D', 'BatchNorm2D', 'input']:
            # Draw as 3D-looking feature maps
            self._draw_feature_maps(ax, x, h, w, c, color, label, parameters)
        elif layer_type in ['Dense', 'output']:
            # Draw as neural network nodes
            self._draw_dense_nodes(ax, x, shape, color, label, parameters)
        elif layer_type in ['Flatten']:
            # Draw as transformation symbol
            self._draw_flatten_symbol(ax, x, shape, color, label)
        else:
            # Draw as specialized symbols for other layers
            self._draw_special_layer(ax, x, shape, color, label, layer_type, parameters)
    
    def _draw_feature_maps(self, ax, x: float, h: int, w: int, c: int, color: str, label: str, parameters: int):
        """Draw feature maps as stacked 3D-looking rectangles."""
        # Calculate dimensions based on spatial size
        base_width = min(max(0.3, np.log10(max(h * w, 1)) * 0.15), 1.0)
        base_height = base_width * 0.8
        
        # Number of visible channels to draw (max 5 for visual clarity)
        num_visible = min(c, 5)
        depth_offset = 0.08
        
        # Draw multiple feature maps with 3D effect
        for i in range(num_visible):
            offset = i * depth_offset
            
            # Main rectangle
            rect = FancyBboxPatch(
                (x - base_width/2 + offset, -base_height/2 + offset),
                base_width, base_height,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor='black',
                linewidth=1.0,
                alpha=0.7 - i*0.1
            )
            ax.add_patch(rect)
            
            # Add 3D side effect
            if i == 0:  # Only for the front-most
                side_points = np.array([
                    [x + base_width/2, -base_height/2],
                    [x + base_width/2 + offset, -base_height/2 + offset],
                    [x + base_width/2 + offset, base_height/2 + offset],
                    [x + base_width/2, base_height/2]
                ])
                side = patches.Polygon(side_points, facecolor=color, alpha=0.5, edgecolor='black', linewidth=0.5)
                ax.add_patch(side)
                
                top_points = np.array([
                    [x - base_width/2, base_height/2],
                    [x - base_width/2 + offset, base_height/2 + offset],
                    [x + base_width/2 + offset, base_height/2 + offset],
                    [x + base_width/2, base_height/2]
                ])
                top = patches.Polygon(top_points, facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
                ax.add_patch(top)
        
        # Add label
        ax.text(x, -base_height/2 - 0.3, label, ha='center', va='top', 
                fontsize=self.text_size-1, fontweight='bold')
        
        # Add dimensions
        dim_text = f"{h}×{w}×{c}" if c > 1 else f"{h}×{w}"
        ax.text(x, -base_height/2 - 0.15, dim_text, ha='center', va='top', 
                fontsize=self.text_size-2, color='gray')
        
        # Add parameter count
        if parameters > 0:
            param_text = self._format_parameters(parameters)
            ax.text(x, base_height/2 + 0.15, param_text, ha='center', va='bottom', 
                    fontsize=self.text_size-3, style='italic', color='blue')
    
    def _draw_dense_nodes(self, ax, x: float, shape: Tuple[int, int, int], color: str, label: str, parameters: int):
        """Draw dense layer as interconnected nodes."""
        _, _, units = shape
        
        # Calculate display parameters
        display_nodes = min(units, 8)  # Show max 8 nodes visually
        node_radius = 0.06
        layer_height = 1.2
        
        if display_nodes > 1:
            y_positions = np.linspace(-layer_height/2, layer_height/2, display_nodes)
        else:
            y_positions = [0]
        
        # Draw nodes
        for i, y_pos in enumerate(y_positions):
            if i == display_nodes - 1 and units > display_nodes:
                # Draw "..." for more nodes
                ax.text(x, y_pos, '⋮', ha='center', va='center', 
                       fontsize=self.text_size+2, fontweight='bold', color=color)
            else:
                circle = patches.Circle((x, y_pos), node_radius, 
                                      facecolor=color, edgecolor='black', 
                                      linewidth=1, alpha=0.8)
                ax.add_patch(circle)
        
        # Add label below
        ax.text(x, -layer_height/2 - 0.3, label, ha='center', va='top', 
                fontsize=self.text_size-1, fontweight='bold')
        
        # Add unit count
        ax.text(x, -layer_height/2 - 0.15, f"{units} units", ha='center', va='top', 
                fontsize=self.text_size-2, color='gray')
        
        # Add parameter count
        if parameters > 0:
            param_text = self._format_parameters(parameters)
            ax.text(x, layer_height/2 + 0.15, param_text, ha='center', va='bottom', 
                    fontsize=self.text_size-3, style='italic', color='blue')
    
    def _draw_flatten_symbol(self, ax, x: float, shape: Tuple[int, int, int], color: str, label: str):
        """Draw flatten layer as transformation arrows."""
        # Draw input representation (small 3D box)
        input_size = 0.3
        ax.add_patch(FancyBboxPatch(
            (x - 0.4 - input_size/2, -input_size/2),
            input_size, input_size,
            boxstyle="round,pad=0.02",
            facecolor='lightblue', edgecolor='black', alpha=0.6
        ))
        
        # Draw arrow
        ax.annotate('', xy=(x + 0.4, 0), xytext=(x - 0.4, 0),
                   arrowprops=dict(arrowstyle='->', lw=2, color=color))
        
        # Draw output representation (line of nodes)
        output_nodes = 5
        node_spacing = 0.08
        start_x = x + 0.4 - (output_nodes-1) * node_spacing / 2
        for i in range(output_nodes):
            node_x = start_x + i * node_spacing
            circle = patches.Circle((node_x, 0), 0.03, 
                                  facecolor=color, edgecolor='black', alpha=0.8)
            ax.add_patch(circle)
        
        # Add label
        ax.text(x, -0.4, label, ha='center', va='top', 
                fontsize=self.text_size-1, fontweight='bold')
    
    def _draw_special_layer(self, ax, x: float, shape: Tuple[int, int, int], color: str, 
                           label: str, layer_type: str, parameters: int):
        """Draw special layers with custom symbols."""
        if layer_type == 'Dropout':
            # Draw as dashed outline
            h, w, c = shape
            base_width = 0.6
            base_height = 0.5
            
            rect = FancyBboxPatch(
                (x - base_width/2, -base_height/2),
                base_width, base_height,
                boxstyle="round,pad=0.05",
                facecolor='none',
                edgecolor=color,
                linewidth=2,
                linestyle='--',
                alpha=0.8
            )
            ax.add_patch(rect)
            
        elif layer_type == 'Activation':
            # Draw as function symbol
            t = np.linspace(-0.3, 0.3, 50)
            
            # Draw different curves for different activations
            if 'relu' in label.lower():
                y = np.maximum(0, t * 2)  # ReLU curve
            elif 'sigmoid' in label.lower():
                y = 1 / (1 + np.exp(-t * 5)) - 0.5  # Sigmoid curve
            elif 'tanh' in label.lower():
                y = np.tanh(t * 3) * 0.3  # Tanh curve
            else:
                y = t  # Linear
            
            ax.plot(x + t, y, color=color, linewidth=3, alpha=0.8)
            
            # Add function box
            rect = FancyBboxPatch(
                (x - 0.4, -0.4),
                0.8, 0.8,
                boxstyle="round,pad=0.05",
                facecolor='none',
                edgecolor=color,
                linewidth=1,
                alpha=0.6
            )
            ax.add_patch(rect)
        else:
            # Default representation
            base_width = 0.6
            base_height = 0.5
            rect = FancyBboxPatch(
                (x - base_width/2, -base_height/2),
                base_width, base_height,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='black',
                linewidth=1,
                alpha=0.7
            )
            ax.add_patch(rect)
        
        # Add label
        ax.text(x, -0.6, label, ha='center', va='top', 
                fontsize=self.text_size-1, fontweight='bold')
        
        if parameters > 0:
            param_text = self._format_parameters(parameters)
            ax.text(x, 0.5, param_text, ha='center', va='bottom', 
                    fontsize=self.text_size-3, style='italic', color='blue')
    
    def _format_parameters(self, parameters: int) -> str:
        """Format parameter count for display."""
        if parameters >= 1000000:
            return f"{parameters/1000000:.1f}M"
        elif parameters >= 1000:
            return f"{parameters/1000:.1f}K"
        else:
            return f"{parameters}"
    
    def _draw_connection_arrow(self, ax, x1: float, x2: float):
        """Draw an enhanced connection arrow between layers."""
        # Draw main arrow
        ax.annotate('', xy=(x2, 0), xytext=(x1, 0),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#2C3E50', alpha=0.7))
        
        # Add data flow indicator
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, 0.1, '→', ha='center', va='bottom', 
                fontsize=8, color='#2C3E50', alpha=0.6)
    
    def _create_layer_label(self, layer_detail: Dict[str, Any]) -> str:
        """Create a descriptive label for a layer."""
        layer_info = layer_detail['layer_info']
        layer_type = layer_info['type']
        
        if layer_type == 'Conv2D':
            return f"Conv2D\n{layer_info['filters']}@{layer_info['kernel_size']}×{layer_info['kernel_size']}"
        elif layer_type in ['MaxPool2D', 'AvgPool2D']:
            return f"{layer_type}\n{layer_info['pool_size']}×{layer_info['pool_size']}"
        elif layer_type == 'Dropout':
            return f"Dropout\n{layer_info['rate']}"
        elif layer_type == 'Dense':
            activation = layer_info.get('activation', 'linear')
            return f"Dense\n{layer_info['units']} units\n({activation})"
        elif layer_type == 'Activation':
            return f"Activation\n({layer_info['activation']})"
        elif layer_type == 'Flatten':
            return "Flatten"
        else:
            return layer_type
    
    def _add_legend(self, ax, total_width: float):
        """Add a legend to the diagram."""
        legend_elements = []
        legend_labels = []
        
        # Get unique layer types from colors
        layer_types = ['Conv2D', 'MaxPool2D', 'AvgPool2D', 'BatchNorm2D', 'Dropout', 'Flatten', 'Dense', 'Activation']
        
        for layer_type in layer_types:
            if layer_type in self.colors:
                legend_elements.append(
                    patches.Rectangle((0, 0), 1, 1, facecolor=self.colors[layer_type], 
                                    edgecolor='black', alpha=0.8)
                )
                legend_labels.append(layer_type)
        
        # Position legend at the bottom
        ax.legend(legend_elements, legend_labels, 
                 loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                 ncol=len(legend_elements), frameon=False)
    
    def generate_detailed_table(self, analyzer: NetworkAnalyzer) -> str:
        """Generate a detailed table of layer information as HTML."""
        layer_details = analyzer.get_layer_details()
        summary = analyzer.get_analysis_summary()
        
        html = """
        <div class="table-container">
            <h3>Layer Details</h3>
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>Layer</th>
                        <th>Type</th>
                        <th>Input Shape</th>
                        <th>Output Shape</th>
                        <th>Parameters</th>
                        <th>Receptive Field</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for detail in layer_details:
            layer_info = detail['layer_info']
            layer_type = layer_info['type']
            
            # Format layer configuration
            config_text = ""
            if layer_type == 'Conv2D':
                config_text = f"{layer_info['filters']}@{layer_info['kernel_size']}×{layer_info['kernel_size']}, stride={layer_info['stride']}"
            elif layer_type in ['MaxPool2D', 'AvgPool2D']:
                config_text = f"{layer_info['pool_size']}×{layer_info['pool_size']}, stride={layer_info['stride']}"
            elif layer_type == 'Dropout':
                config_text = f"rate={layer_info['rate']}"
            
            # Format shapes
            input_shape = f"{detail['input_shape'][0]}×{detail['input_shape'][1]}×{detail['input_shape'][2]}"
            output_shape = f"{detail['output_shape'][0]}×{detail['output_shape'][1]}×{detail['output_shape'][2]}"
            
            # Format parameters
            params = detail['parameters']
            if params == 0:
                param_text = "0"
            elif params >= 1000000:
                param_text = f"{params:,} ({params/1000000:.1f}M)"
            elif params >= 1000:
                param_text = f"{params:,} ({params/1000:.1f}K)"
            else:
                param_text = f"{params:,}"
            
            # Format receptive field
            rf = detail['receptive_field']
            rf_text = str(rf) if rf != 'Global' else 'Global'
            
            html += f"""
                    <tr>
                        <td><strong>{detail['layer_name']}</strong><br><small>{config_text}</small></td>
                        <td>{layer_type}</td>
                        <td>{input_shape}</td>
                        <td>{output_shape}</td>
                        <td>{param_text}</td>
                        <td>{rf_text}</td>
                    </tr>
            """
        
        # Add summary row
        total_params = summary['total_parameters']
        if total_params >= 1000000:
            total_param_text = f"{total_params:,} ({total_params/1000000:.1f}M)"
        elif total_params >= 1000:
            total_param_text = f"{total_params:,} ({total_params/1000:.1f}K)"
        else:
            total_param_text = f"{total_params:,}"
        
        html += f"""
                    <tr class="summary-row">
                        <td colspan="4"><strong>Total</strong></td>
                        <td><strong>{total_param_text}</strong></td>
                        <td><strong>{summary['final_receptive_field']}</strong></td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
        
        return html
