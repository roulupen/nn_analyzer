// Neural Network Analyzer JavaScript

class NeuralNetworkAnalyzer {
    constructor() {
        this.initializeEventListeners();
        this.initializeLayerParameters();
        this.loadInitialState();
    }

    initializeEventListeners() {
        // Input shape form
        document.getElementById('inputShapeForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.setInputShape();
        });

        // Layer form
        document.getElementById('layerForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addLayer();
        });

        // Layer type change
        document.getElementById('layerType').addEventListener('change', (e) => {
            this.updateLayerParameters(e.target.value);
        });

        // Problem form
        document.getElementById('problemForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.configureProblem();
        });

        // Problem type change
        document.getElementById('problemType').addEventListener('change', (e) => {
            this.updateProblemParameters(e.target.value);
        });

        // Clear layers button
        document.getElementById('clearLayers').addEventListener('click', () => {
            this.clearLayers();
        });

        // Generate diagram button
        document.getElementById('generateDiagram').addEventListener('click', () => {
            this.generateDiagram();
        });

        // Export architecture button
        document.getElementById('exportArchitecture').addEventListener('click', () => {
            this.exportArchitecture();
        });

        // Collapsible section handlers
        this.initializeCollapsibleSections();
    }

    initializeCollapsibleSections() {
        // Handle section header clicks for collapsible functionality
        document.querySelectorAll('.section-header').forEach(header => {
            header.addEventListener('click', () => {
                const toggleIcon = header.querySelector('.toggle-icon');
                const isExpanded = header.getAttribute('aria-expanded') === 'true';
                
                // Update aria-expanded
                header.setAttribute('aria-expanded', !isExpanded);
                
                // Update icon
                if (!isExpanded) {
                    toggleIcon.classList.remove('fa-chevron-right');
                    toggleIcon.classList.add('fa-chevron-down');
                } else {
                    toggleIcon.classList.remove('fa-chevron-down');
                    toggleIcon.classList.add('fa-chevron-right');
                }
            });
        });

        // Listen to Bootstrap collapse events to update icons
        document.querySelectorAll('.collapse').forEach(collapse => {
            collapse.addEventListener('show.bs.collapse', () => {
                const header = document.querySelector(`[data-bs-target="#${collapse.id}"]`);
                if (header) {
                    const toggleIcon = header.querySelector('.toggle-icon');
                    toggleIcon.classList.remove('fa-chevron-right');
                    toggleIcon.classList.add('fa-chevron-down');
                }
            });
            
            collapse.addEventListener('hide.bs.collapse', () => {
                const header = document.querySelector(`[data-bs-target="#${collapse.id}"]`);
                if (header) {
                    const toggleIcon = header.querySelector('.toggle-icon');
                    toggleIcon.classList.remove('fa-chevron-down');
                    toggleIcon.classList.add('fa-chevron-right');
                }
            });
        });
    }

    initializeLayerParameters() {
        this.layerParameterTemplates = {
            'conv2d': `
                <div class="compact-params">
                    <div class="row g-1 mb-1">
                        <div class="col-6">
                            <input type="number" class="form-control form-control-sm" name="filters" id="filters" required min="1" value="32" placeholder="Filters">
                        </div>
                        <div class="col-6">
                            <select class="form-select form-select-sm" name="kernel_size" id="kernel_size" required>
                                <option value="1">1×1</option>
                                <option value="3" selected>3×3</option>
                                <option value="5">5×5</option>
                                <option value="7">7×7</option>
                            </select>
                        </div>
                    </div>
                    <div class="row g-1 mb-1">
                        <div class="col-4">
                            <input type="number" class="form-control form-control-sm" name="stride" id="stride" min="1" value="1" placeholder="Stride">
                        </div>
                        <div class="col-4">
                            <select class="form-select form-select-sm" name="padding" id="padding">
                                <option value="valid" selected>Valid</option>
                                <option value="same">Same</option>
                            </select>
                        </div>
                        <div class="col-4">
                            <select class="form-select form-select-sm" name="activation" id="activation">
                                <option value="relu" selected>ReLU</option>
                                <option value="sigmoid">Sigmoid</option>
                                <option value="tanh">Tanh</option>
                                <option value="linear">Linear</option>
                            </select>
                        </div>
                    </div>
                </div>
            `,
            'maxpool2d': `
                <div class="compact-params">
                    <div class="row g-1 mb-1">
                        <div class="col-4">
                            <select class="form-select form-select-sm" name="pool_size" id="pool_size" required>
                                <option value="2" selected>2×2</option>
                                <option value="3">3×3</option>
                                <option value="4">4×4</option>
                            </select>
                        </div>
                        <div class="col-4">
                            <input type="number" class="form-control form-control-sm" name="stride" id="stride" min="1" value="2" placeholder="Stride">
                        </div>
                        <div class="col-4">
                            <select class="form-select form-select-sm" name="padding" id="padding">
                                <option value="valid" selected>Valid</option>
                                <option value="same">Same</option>
                            </select>
                        </div>
                    </div>
                </div>
            `,
            'avgpool2d': `
                <div class="compact-params">
                    <div class="row g-1 mb-1">
                        <div class="col-4">
                            <select class="form-select form-select-sm" name="pool_size" id="pool_size" required>
                                <option value="2" selected>2×2</option>
                                <option value="3">3×3</option>
                                <option value="4">4×4</option>
                            </select>
                        </div>
                        <div class="col-4">
                            <input type="number" class="form-control form-control-sm" name="stride" id="stride" min="1" value="2" placeholder="Stride">
                        </div>
                        <div class="col-4">
                            <select class="form-select form-select-sm" name="padding" id="padding">
                                <option value="valid" selected>Valid</option>
                                <option value="same">Same</option>
                            </select>
                        </div>
                    </div>
                </div>
            `,
            'dropout': `
                <div class="compact-params">
                    <input type="number" class="form-control form-control-sm" name="rate" id="rate" 
                           min="0" max="1" step="0.1" value="0.5" placeholder="Dropout Rate (0.0-1.0)">
                </div>
            `,
            'batchnorm': `
                <div class="compact-params">
                    <small class="text-muted">No parameters required</small>
                </div>
            `,
            'globalavgpool2d': `
                <div class="compact-params">
                    <small class="text-muted">No parameters required</small>
                </div>
            `,
            'flatten': `
                <div class="compact-params">
                    <small class="text-muted">Converts 2D → 1D</small>
                </div>
            `,
            'dense': `
                <div class="compact-params">
                    <div class="row g-1 mb-1">
                        <div class="col-6">
                            <input type="number" class="form-control form-control-sm" name="units" id="units" required min="1" value="128" placeholder="Units">
                        </div>
                        <div class="col-6">
                            <select class="form-select form-select-sm" name="activation" id="activation">
                                <option value="relu" selected>ReLU</option>
                                <option value="sigmoid">Sigmoid</option>
                                <option value="tanh">Tanh</option>
                                <option value="softmax">Softmax</option>
                                <option value="linear">Linear</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-check form-check-sm">
                        <input class="form-check-input form-check-input-sm" type="checkbox" name="use_bias" id="use_bias" checked>
                        <label class="form-check-label small" for="use_bias">Use Bias</label>
                    </div>
                </div>
            `,
            'activation': `
                <div class="compact-params">
                    <select class="form-select form-select-sm" name="activation" id="activation" required>
                        <option value="relu" selected>ReLU</option>
                        <option value="sigmoid">Sigmoid</option>
                        <option value="tanh">Tanh</option>
                        <option value="softmax">Softmax</option>
                        <option value="linear">Linear</option>
                        <option value="leaky_relu">Leaky ReLU</option>
                        <option value="swish">Swish</option>
                    </select>
                </div>
            `
        };
    }

    updateLayerParameters(layerType) {
        const parametersContainer = document.getElementById('layerParameters');
        
        if (layerType && this.layerParameterTemplates[layerType]) {
            parametersContainer.innerHTML = this.layerParameterTemplates[layerType];
            parametersContainer.style.display = 'block';
        } else {
            parametersContainer.innerHTML = '';
            parametersContainer.style.display = 'none';
        }
    }

    updateProblemParameters(problemType) {
        const parametersContainer = document.getElementById('problemParameters');
        const numClassesInput = document.getElementById('numClasses');
        
        if (problemType === 'classification') {
            parametersContainer.style.display = 'block';
            numClassesInput.placeholder = 'e.g., 10 for CIFAR-10, 1000 for ImageNet';
            numClassesInput.required = true;
        } else if (problemType === 'regression') {
            parametersContainer.style.display = 'block';
            numClassesInput.placeholder = 'Number of output values (default: 1)';
            numClassesInput.required = false;
        } else if (problemType === 'binary_classification') {
            parametersContainer.style.display = 'none';
            numClassesInput.required = false;
        } else {
            parametersContainer.style.display = 'none';
            numClassesInput.required = false;
        }
    }

    async configureProblem() {
        const formData = new FormData(document.getElementById('problemForm'));
        
        try {
            const response = await fetch('/configure_problem', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showToast(result.message, 'success');
                this.updateLayersList(result.layers);
                this.updateSummary(result.summary);
                this.resetProblemForm();
            } else {
                this.showToast(result.detail || 'Error configuring problem type', 'error');
            }
        } catch (error) {
            this.showToast('Network error occurred', 'error');
            console.error('Error:', error);
        }
    }

    resetProblemForm() {
        document.getElementById('problemForm').reset();
        document.getElementById('problemParameters').style.display = 'none';
    }

    async setInputShape() {
        const formData = new FormData(document.getElementById('inputShapeForm'));
        
        try {
            const response = await fetch('/set_input_shape', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showToast(result.message, 'success');
                this.updateSummary(result.summary);
                this.refreshLayers();
            } else {
                this.showToast(result.detail || 'Error setting input shape', 'error');
            }
        } catch (error) {
            this.showToast('Network error occurred', 'error');
            console.error('Error:', error);
        }
    }

    async addLayer() {
        const formData = new FormData(document.getElementById('layerForm'));
        
        try {
            const response = await fetch('/add_layer', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showToast(result.message, 'success');
                this.updateLayersList(result.layers);
                this.updateSummary(result.summary);
                this.resetLayerForm();
            } else {
                this.showToast(result.detail || 'Error adding layer', 'error');
            }
        } catch (error) {
            this.showToast('Network error occurred', 'error');
            console.error('Error:', error);
        }
    }

    async removeLayer(layerIndex) {
        try {
            const response = await fetch(`/remove_layer/${layerIndex}`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showToast(result.message, 'success');
                this.updateLayersList(result.layers);
                this.updateSummary(result.summary);
                this.hideDiagram();
            } else {
                this.showToast(result.detail || 'Error removing layer', 'error');
            }
        } catch (error) {
            this.showToast('Network error occurred', 'error');
            console.error('Error:', error);
        }
    }

    async clearLayers() {
        if (!confirm('Are you sure you want to clear all layers?')) {
            return;
        }
        
        try {
            const response = await fetch('/clear_layers', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showToast(result.message, 'success');
                this.updateLayersList(result.layers);
                this.updateSummary(result.summary);
                this.hideDiagram();
            } else {
                this.showToast(result.detail || 'Error clearing layers', 'error');
            }
        } catch (error) {
            this.showToast('Network error occurred', 'error');
            console.error('Error:', error);
        }
    }

    async generateDiagram() {
        const diagramCard = document.getElementById('diagramCard');
        const analysisCard = document.getElementById('analysisCard');
        const spinner = document.getElementById('diagramSpinner');
        const diagram = document.getElementById('architectureDiagram');
        
        // Show loading state
        diagramCard.style.display = 'block';
        spinner.style.display = 'block';
        diagram.style.display = 'none';
        
        try {
            const response = await fetch('/generate_diagram', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                // Display diagram
                diagram.src = `data:image/png;base64,${result.image}`;
                diagram.style.display = 'block';
                
                // Display analysis table
                document.getElementById('analysisTable').innerHTML = result.table;
                analysisCard.style.display = 'block';
                
                this.showToast('Architecture diagram generated successfully', 'success');
            } else {
                this.showToast(result.detail || 'Error generating diagram', 'error');
                diagramCard.style.display = 'none';
            }
        } catch (error) {
            this.showToast('Network error occurred', 'error');
            console.error('Error:', error);
            diagramCard.style.display = 'none';
        } finally {
            spinner.style.display = 'none';
        }
    }

    async exportArchitecture() {
        try {
            const response = await fetch('/export_architecture');
            const result = await response.json();
            
            if (response.ok) {
                const dataStr = JSON.stringify(result.architecture, null, 2);
                const dataBlob = new Blob([dataStr], {type: 'application/json'});
                
                const link = document.createElement('a');
                link.href = URL.createObjectURL(dataBlob);
                link.download = 'neural_network_architecture.json';
                link.click();
                
                this.showToast('Architecture exported successfully', 'success');
            } else {
                this.showToast(result.detail || 'Error exporting architecture', 'error');
            }
        } catch (error) {
            this.showToast('Network error occurred', 'error');
            console.error('Error:', error);
        }
    }

    async loadInitialState() {
        try {
            const response = await fetch('/get_analysis');
            const result = await response.json();
            
            this.updateLayersList(result.layers);
            this.updateSummary(result.summary);
            this.updateInputShapeDisplay(result.input_shape);
        } catch (error) {
            console.error('Error loading initial state:', error);
        }
    }

    async refreshLayers() {
        try {
            const response = await fetch('/get_analysis');
            const result = await response.json();
            
            this.updateLayersList(result.layers);
            this.hideDiagram();
        } catch (error) {
            console.error('Error refreshing layers:', error);
        }
    }

    updateInputShapeDisplay(inputShape) {
        const [h, w, c] = inputShape;
        document.getElementById('height').value = h;
        document.getElementById('width').value = w;
        document.getElementById('channels').value = c;
    }

    updateSummary(summary) {
        document.getElementById('totalLayers').textContent = summary.total_layers;
        document.getElementById('totalParams').textContent = this.formatNumber(summary.total_parameters);
        document.getElementById('inputShape').textContent = `${summary.input_shape[0]}×${summary.input_shape[1]}×${summary.input_shape[2]}`;
        document.getElementById('outputShape').textContent = `${summary.output_shape[0]}×${summary.output_shape[1]}×${summary.output_shape[2]}`;
        document.getElementById('receptiveField').textContent = summary.final_receptive_field;
    }

    updateLayersList(layers) {
        const layersContainer = document.getElementById('layersList');
        
        if (layers.length === 0) {
            layersContainer.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-info-circle"></i> No layers added yet. Start by adding layers from the left panel.
                </div>
            `;
            return;
        }
        
        // Create compact table format
        let layersHTML = `
            <div class="layers-table-container">
                <table class="layers-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Layer</th>
                            <th>Type</th>
                            <th>Input Shape</th>
                            <th>Output Shape</th>
                            <th>Parameters</th>
                            <th>RF</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        layers.forEach((layer, index) => {
            const layerInfo = layer.layer_info;
            const layerType = layerInfo.type.toLowerCase().replace('2d', '2d');
            
            // Create layer configuration summary
            let configSummary = '';
            if (layerInfo.type === 'Conv2D') {
                configSummary = `${layerInfo.filters}@${layerInfo.kernel_size}×${layerInfo.kernel_size}`;
            } else if (layerInfo.type.includes('Pool')) {
                configSummary = `${layerInfo.pool_size}×${layerInfo.pool_size}`;
            } else if (layerInfo.type === 'Dense') {
                configSummary = `${layerInfo.units} units`;
            } else if (layerInfo.type === 'Dropout') {
                configSummary = `${layerInfo.rate}`;
            } else if (layerInfo.type === 'Activation') {
                configSummary = layerInfo.activation;
            }
            
            layersHTML += `
                <tr class="layer-row fade-in" data-layer-type="${layerType}">
                    <td class="layer-index">${index + 1}</td>
                    <td class="layer-name">
                        <div class="layer-name-main">${layer.layer_name}</div>
                        ${configSummary ? `<div class="layer-config">${configSummary}</div>` : ''}
                    </td>
                    <td>
                        <span class="layer-badge layer-type-${layerType}">${layerInfo.type}</span>
                    </td>
                    <td class="shape-cell">${layer.input_shape[0]}×${layer.input_shape[1]}×${layer.input_shape[2]}</td>
                    <td class="shape-cell">${layer.output_shape[0]}×${layer.output_shape[1]}×${layer.output_shape[2]}</td>
                    <td class="params-cell">${this.formatNumber(layer.parameters)}</td>
                    <td class="rf-cell">${layer.receptive_field}</td>
                    <td class="action-cell">
                        <button class="btn btn-remove-small" onclick="app.removeLayer(${index})" title="Remove layer">
                            <i class="fas fa-times"></i>
                        </button>
                    </td>
                </tr>
            `;
        });
        
        layersHTML += `
                    </tbody>
                </table>
            </div>
        `;
        
        layersContainer.innerHTML = layersHTML;
    }

    resetLayerForm() {
        document.getElementById('layerForm').reset();
        document.getElementById('layerParameters').innerHTML = '';
        document.getElementById('layerType').value = '';
    }

    hideDiagram() {
        document.getElementById('diagramCard').style.display = 'none';
        document.getElementById('analysisCard').style.display = 'none';
    }

    formatNumber(num) {
        if (num === 0) return '0';
        if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
        if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
        return num.toLocaleString();
    }

    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        const toastMessage = document.getElementById('toastMessage');
        
        // Set message and style
        toastMessage.textContent = message;
        
        // Remove existing classes
        toast.classList.remove('bg-success', 'bg-danger', 'bg-info', 'bg-warning');
        
        // Add appropriate class based on type
        switch (type) {
            case 'success':
                toast.classList.add('bg-success', 'text-white');
                break;
            case 'error':
                toast.classList.add('bg-danger', 'text-white');
                break;
            case 'warning':
                toast.classList.add('bg-warning');
                break;
            default:
                toast.classList.add('bg-info', 'text-white');
        }
        
        // Show toast
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new NeuralNetworkAnalyzer();
});
