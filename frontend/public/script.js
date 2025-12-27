// CKD Insight - Frontend JavaScript
// Connects to Flask API for predictions

// ============================================================================
// CONFIGURATION
// ============================================================================

const API_BASE_URL = 'http://localhost:5000';

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

let currentMode = 'user';
let selectedModel = 'random_forest';
let availableModels = [];
let currentBatchResults = null;

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 CKD Insight initializing...');
    
    // Load available models
    await loadModels();
    
    // Initialize model selector for researcher mode
    initializeModelSelector();
    
    // Initialize charts
    initializeCharts();
    
    console.log('✓ Initialization complete');
});

// ============================================================================
// API FUNCTIONS
// ============================================================================

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/models`);
        const data = await response.json();
        
        if (data.success) {
            availableModels = data.models;
            console.log(`✓ Loaded ${availableModels.length} models`);
            return availableModels;
        } else {
            console.error('Failed to load models:', data.error);
            showNotification('Failed to load models', 'error');
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showNotification('Backend connection error. Make sure Flask server is running on port 5000.', 'error');
    }
}

async function makePrediction(features, modelName = 'random_forest') {
    try {
        showLoading(true);
        
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: modelName,
                features: features
            })
        });
        
        const data = await response.json();
        
        showLoading(false);
        
        if (data.success) {
            return data.result;
        } else {
            throw new Error(data.error || 'Prediction failed');
        }
    } catch (error) {
        showLoading(false);
        console.error('Prediction error:', error);
        showNotification('Prediction failed: ' + error.message, 'error');
        return null;
    }
}

async function makeBatchPrediction(file, modelName = 'random_forest') {
    try {
        showLoading(true);
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', modelName);
        
        const response = await fetch(`${API_BASE_URL}/api/predict/batch`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        showLoading(false);
        
        if (data.success) {
            return data;
        } else {
            throw new Error(data.error || 'Batch prediction failed');
        }
    } catch (error) {
        showLoading(false);
        console.error('Batch prediction error:', error);
        showNotification('Batch prediction failed: ' + error.message, 'error');
        return null;
    }
}

async function getFeatureImportance(modelName = 'random_forest') {
    try {
        const response = await fetch(`${API_BASE_URL}/api/feature/importance?model=${modelName}`);
        const data = await response.json();
        
        if (data.success) {
            return data.feature_importance;
        }
        return null;
    } catch (error) {
        console.error('Error getting feature importance:', error);
        return null;
    }
}

// ============================================================================
// USER MODE FUNCTIONS
// ============================================================================

function collectFormData(prefix = '') {
    const features = {};
    
    const fields = [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
        'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
        'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
    ];
    
    fields.forEach(field => {
        const element = document.getElementById(prefix + field);
        if (element) {
            const value = element.value;
            if (value !== '') {
                // Check if it's a select (categorical) or input (numeric)
                if (element.tagName === 'SELECT') {
                    features[field] = value;
                } else {
                    features[field] = parseFloat(value);
                }
            }
        }
    });
    
    return features;
}

async function generateDiagnosis() {
    // Collect form data
    const features = collectFormData();
    
    // Validate that at least some fields are filled
    if (Object.keys(features).length === 0) {
        showNotification('Please fill in at least some fields', 'warning');
        return;
    }
    
    console.log('Generating diagnosis with features:', features);
    
    // Make prediction
    const result = await makePrediction(features, selectedModel);
    
    if (result) {
        displayResult(result);
        await displayFeatureImportance(result);
        showTreatmentRecommendations(result);
    }
}

async function generateResearchDiagnosis() {
    // Collect form data from research mode
    const features = collectFormData('research_');
    
    // Validate that at least some fields are filled
    if (Object.keys(features).length === 0) {
        showNotification('Please fill in at least some fields', 'warning');
        return;
    }
    
    console.log('Generating research diagnosis with features:', features);
    
    // Make prediction
    const result = await makePrediction(features, selectedModel);
    
    if (result) {
        showNotification(`Prediction: ${result.prediction} (${(result.confidence_score * 100).toFixed(1)}% confidence)`, 'success');
        console.log('Research prediction result:', result);
    }
}

function displayResult(result) {
    const resultCard = document.getElementById('resultCard');
    
    const prediction = result.prediction;
    const probability = result.probabilities;
    const confidence = result.confidence;
    
    // Determine if CKD
    const isCKD = prediction === 'CKD';
    const ckdProb = probability['CKD'] || 0;
    const notCkdProb = probability['No Disease'] || 0;
    
    const statusColor = isCKD ? 'var(--error)' : 'var(--success)';
    const statusText = isCKD ? 'CKD Detected' : 'No CKD Detected';
    const statusIcon = isCKD ? 
        '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>' :
        '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>';
    
    resultCard.innerHTML = `
        <div class="card-content" style="padding: 2rem;">
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="width: 4rem; height: 4rem; border-radius: 50%; background: ${statusColor}15; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center;">
                    <svg style="width: 2rem; height: 2rem; color: ${statusColor};" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        ${statusIcon}
                    </svg>
                </div>
                <h2 style="font-size: 1.5rem; font-weight: 700; color: ${statusColor}; margin-bottom: 0.5rem;">
                    ${statusText}
                </h2>
                <p style="color: var(--muted-text); font-size: 0.875rem;">
                    Confidence: ${confidence.toUpperCase()} (${(result.confidence_score * 100).toFixed(1)}%)
                </p>
            </div>
            
            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.875rem;">
                    <span>CKD Probability</span>
                    <span style="font-weight: 600;">${(ckdProb * 100).toFixed(1)}%</span>
                </div>
                <div style="height: 0.5rem; background: var(--accent-bg); border-radius: 1rem; overflow: hidden;">
                    <div style="height: 100%; background: ${statusColor}; width: ${ckdProb * 100}%; transition: width 0.5s ease;"></div>
                </div>
            </div>
            
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.875rem;">
                    <span>No CKD Probability</span>
                    <span style="font-weight: 600;">${(notCkdProb * 100).toFixed(1)}%</span>
                </div>
                <div style="height: 0.5rem; background: var(--accent-bg); border-radius: 1rem; overflow: hidden;">
                    <div style="height: 100%; background: var(--success); width: ${notCkdProb * 100}%; transition: width 0.5s ease;"></div>
                </div>
            </div>
            
            <div style="margin-top: 1.5rem; padding: 1rem; background: var(--accent-bg); border-radius: 0.5rem; font-size: 0.875rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <strong>Model Used:</strong>
                    <span>${result.model_used.replace('_', ' ').toUpperCase()}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <strong>Features Provided:</strong>
                    <span>${result.features_received} / ${result.total_features}</span>
                </div>
            </div>
        </div>
    `;
}

async function displayFeatureImportance(result) {
    const explainSection = document.getElementById('explainabilitySection');
    explainSection.classList.remove('hidden');
    
    // Get feature importance from the model
    const importance = await getFeatureImportance(result.model_used);
    
    if (importance) {
        // Get top 6 features
        const features_list = Object.entries(importance).slice(0, 6);
        const labels = features_list.map(([name, _]) => name.toUpperCase());
        const values = features_list.map(([_, value]) => value * 100);
        
        // Create chart
        const canvas = document.getElementById('featureImportanceChart');
        const ctx = canvas.getContext('2d');
        
        // Clear previous chart
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw bar chart
        drawBarChart(ctx, labels, values, canvas.width, canvas.height);
        
        // Update explanation text
        updateExplanationText(features_list);
    }
}

function updateExplanationText(topFeatures) {
    const top1 = topFeatures[0];
    const top2 = topFeatures[1];
    
    const explanationHTML = `
        <h4 style="font-weight: 600; margin-bottom: 0.5rem;">AI Explanation</h4>
        <p style="font-size: 0.875rem; line-height: 1.6;">
            The model identified <strong>${top1[0].toUpperCase()} (${(top1[1] * 100).toFixed(0)}%)</strong> and 
            <strong>${top2[0].toUpperCase()} (${(top2[1] * 100).toFixed(0)}%)</strong> as the most influential 
            factors in this prediction. These biomarkers play a crucial role in assessing kidney function and 
            determining CKD risk.
        </p>
    `;
    
    const infoBox = document.querySelector('#explainabilitySection .info-box div:last-child');
    if (infoBox) {
        infoBox.innerHTML = explanationHTML;
    }
}

function drawBarChart(ctx, labels, values, width, height) {
    ctx.clearRect(0, 0, width, height);
    
    const padding = 60;
    const barWidth = (width - padding * 2) / labels.length - 10;
    const maxValue = Math.max(...values);
    
    // Draw bars
    values.forEach((value, i) => {
        const barHeight = (value / maxValue) * (height - padding * 2);
        const x = padding + i * (barWidth + 10);
        const y = height - padding - barHeight;
        
        // Bar gradient
        const gradient = ctx.createLinearGradient(x, y, x, height - padding);
        gradient.addColorStop(0, '#1E88E5');
        gradient.addColorStop(1, '#1565C0');
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth, barHeight);
        
        // Value label
        ctx.fillStyle = '#333';
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${value.toFixed(1)}%`, x + barWidth / 2, y - 5);
        
        // Feature label
        ctx.save();
        ctx.translate(x + barWidth / 2, height - 10);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = 'right';
        ctx.font = '11px sans-serif';
        ctx.fillText(labels[i], 0, 0);
        ctx.restore();
    });
}

function showTreatmentRecommendations(result) {
    const treatmentCard = document.getElementById('treatmentCard');
    treatmentCard.classList.remove('hidden');
}

// ============================================================================
// CSV UPLOAD FUNCTIONS
// ============================================================================

async function handleCSVUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    console.log('CSV file selected:', file.name);
    
    // Validate file type
    if (!file.name.endsWith('.csv')) {
        showNotification('Please upload a CSV file', 'error');
        return;
    }
    
    // Show loading
    showNotification('Processing CSV file...', 'info');
    
    // Make batch prediction
    const result = await makeBatchPrediction(file, selectedModel);
    
    if (result) {
        currentBatchResults = result;
        displayBatchResults(result);
        showNotification(`Processed ${result.summary.total_rows} rows successfully!`, 'success');
    }
    
    // Reset file input
    event.target.value = '';
}

async function handleResearchCSVUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    console.log('Research CSV file selected:', file.name);
    
    // Validate file type
    if (!file.name.endsWith('.csv')) {
        showNotification('Please upload a CSV file', 'error');
        return;
    }
    
    // Show loading
    showNotification('Processing CSV file...', 'info');
    
    // Make batch prediction
    const result = await makeBatchPrediction(file, selectedModel);
    
    if (result) {
        currentBatchResults = result;
        displayBatchResultsResearch(result);
        showNotification(`Processed ${result.summary.total_rows} rows successfully!`, 'success');
    }
    
    // Reset file input
    event.target.value = '';
}

function displayBatchResults(batchData) {
    const resultCard = document.getElementById('resultCard');
    const treatmentCard = document.getElementById('treatmentCard');
    const explainSection = document.getElementById('explainabilitySection');
    
    const summary = batchData.summary;
    const results = batchData.results;
    
    // Display summary
    resultCard.innerHTML = `
        <div class="card-content" style="padding: 2rem;">
            <h3 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1.5rem; text-align: center;">
                Batch Prediction Results
            </h3>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                <div style="padding: 1rem; background: var(--accent-bg); border-radius: 0.5rem;">
                    <div style="font-size: 0.75rem; color: var(--muted-text); margin-bottom: 0.25rem;">Total Rows</div>
                    <div style="font-size: 1.5rem; font-weight: 700;">${summary.total_rows}</div>
                </div>
                <div style="padding: 1rem; background: var(--accent-bg); border-radius: 0.5rem;">
                    <div style="font-size: 0.75rem; color: var(--muted-text); margin-bottom: 0.25rem;">Successful</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--success);">${summary.successful_predictions}</div>
                </div>
                <div style="padding: 1rem; background: rgba(244, 67, 54, 0.1); border-radius: 0.5rem;">
                    <div style="font-size: 0.75rem; color: var(--muted-text); margin-bottom: 0.25rem;">CKD Detected</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--error);">${summary.ckd_detected}</div>
                </div>
                <div style="padding: 1rem; background: rgba(67, 160, 71, 0.1); border-radius: 0.5rem;">
                    <div style="font-size: 0.75rem; color: var(--muted-text); margin-bottom: 0.25rem;">No Disease</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--success);">${summary.no_disease}</div>
                </div>
            </div>
            
            <div style="max-height: 300px; overflow-y: auto; border: 1px solid var(--border); border-radius: 0.5rem;">
                <table style="width: 100%; font-size: 0.875rem;">
                    <thead style="background: var(--accent-bg); position: sticky; top: 0;">
                        <tr>
                            <th style="padding: 0.75rem; text-align: left;">Row</th>
                            <th style="padding: 0.75rem; text-align: left;">Prediction</th>
                            <th style="padding: 0.75rem; text-align: left;">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${results.slice(0, 20).map(r => `
                            <tr style="border-top: 1px solid var(--border);">
                                <td style="padding: 0.75rem;">#${r.row_number}</td>
                                <td style="padding: 0.75rem;">
                                    <span style="color: ${r.prediction === 'CKD' ? 'var(--error)' : 'var(--success)'}; font-weight: 600;">
                                        ${r.prediction}
                                    </span>
                                </td>
                                <td style="padding: 0.75rem;">${r.confidence ? r.confidence.toUpperCase() : 'N/A'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            
            ${results.length > 20 ? `<p style="text-align: center; margin-top: 1rem; font-size: 0.875rem; color: var(--muted-text);">Showing first 20 of ${results.length} results</p>` : ''}
            
            <button class="btn btn-outline" style="width: 100%; margin-top: 1rem;" onclick="downloadBatchResults()">
                Download Full Results (CSV)
            </button>
        </div>
    `;
    
    // Hide other sections
    treatmentCard.classList.add('hidden');
    explainSection.classList.add('hidden');
}

function displayBatchResultsResearch(batchData) {
    // Similar to displayBatchResults but for research mode
    console.log('Batch results for research mode:', batchData);
    showNotification(
        `Batch prediction complete: ${batchData.summary.ckd_detected} CKD cases detected out of ${batchData.summary.total_rows} samples`,
        'success'
    );
}

function downloadBatchResults() {
    if (!currentBatchResults) {
        showNotification('No batch results to download', 'warning');
        return;
    }
    
    // Create CSV content
    let csvContent = "Row,Prediction,Confidence,CKD_Probability,NoDisease_Probability\n";
    
    currentBatchResults.results.forEach(r => {
        if (r.prediction !== 'Error') {
            csvContent += `${r.row_number},${r.prediction},${r.confidence},${r.probabilities.CKD},${r.probabilities['No Disease']}\n`;
        }
    });
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ckd_predictions_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    showNotification('Results downloaded successfully!', 'success');
}

// ============================================================================
// RESEARCHER MODE FUNCTIONS
// ============================================================================

function initializeModelSelector() {
    const selector = document.getElementById('modelSelector');
    if (!selector) return;
    
    const models = [
        { id: 'random_forest', name: 'Random Forest', accuracy: 99.25 },
        { id: 'logistic_regression', name: 'Logistic Regression', accuracy: 99.00 },
        { id: 'svm', name: 'Support Vector Machine', accuracy: 98.75 },
        { id: 'xgboost', name: 'XGBoost', accuracy: 98.75 },
        { id: 'knn', name: 'K-Nearest Neighbors', accuracy: 95.75 },
        { id: 'neural_network', name: 'Neural Network', accuracy: 94.50 }
    ];
    
    selector.innerHTML = models.map(model => `
        <div class="model-option ${model.id === 'random_forest' ? 'selected' : ''}" 
             onclick="selectModel('${model.id}')">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600;">${model.name}</span>
                <span style="color: var(--success);">${model.accuracy}%</span>
            </div>
        </div>
    `).join('');
}

function selectModel(modelId) {
    selectedModel = modelId;
    
    // Update UI
    document.querySelectorAll('.model-option').forEach(option => {
        option.classList.remove('selected');
    });
    event.target.closest('.model-option').classList.add('selected');
    
    console.log(`Selected model: ${modelId}`);
}

function initializeCharts() {
    console.log('Charts initialized');
}

// ============================================================================
// UI FUNCTIONS
// ============================================================================

function showLanding() {
    document.getElementById('landingPage').classList.remove('hidden');
    document.getElementById('dashboard').classList.add('hidden');
}

function showDashboard(mode) {
    document.getElementById('landingPage').classList.add('hidden');
    document.getElementById('dashboard').classList.remove('hidden');
    
    currentMode = mode;
    
    if (mode === 'user') {
        document.getElementById('userMode').classList.remove('hidden');
        document.getElementById('researcherMode').classList.add('hidden');
        document.getElementById('modeSwitch').classList.remove('active');
    } else {
        document.getElementById('userMode').classList.add('hidden');
        document.getElementById('researcherMode').classList.remove('hidden');
        document.getElementById('modeSwitch').classList.add('active');
    }
}

function toggleMode() {
    currentMode = currentMode === 'user' ? 'researcher' : 'user';
    showDashboard(currentMode);
}

function switchTab(tabName) {
    if (tabName === 'manual') {
        document.getElementById('manualTab').classList.add('active');
        document.getElementById('csvTab').classList.remove('active');
        document.querySelectorAll('#userMode .tab-trigger')[0].classList.add('active');
        document.querySelectorAll('#userMode .tab-trigger')[1].classList.remove('active');
    } else {
        document.getElementById('manualTab').classList.remove('active');
        document.getElementById('csvTab').classList.add('active');
        document.querySelectorAll('#userMode .tab-trigger')[0].classList.remove('active');
        document.querySelectorAll('#userMode .tab-trigger')[1].classList.add('active');
    }
}

function switchResearchTab(tabName) {
    if (tabName === 'manual') {
        document.getElementById('researchManualTab').classList.add('active');
        document.getElementById('researchCSVTab').classList.remove('active');
        document.querySelectorAll('#researcherMode .tabs-list .tab-trigger')[0].classList.add('active');
        document.querySelectorAll('#researcherMode .tabs-list .tab-trigger')[1].classList.remove('active');
    } else {
        document.getElementById('researchManualTab').classList.remove('active');
        document.getElementById('researchCSVTab').classList.add('active');
        document.querySelectorAll('#researcherMode .tabs-list .tab-trigger')[0].classList.remove('active');
        document.querySelectorAll('#researcherMode .tabs-list .tab-trigger')[1].classList.add('active');
    }
}

function switchTrainingTab(tabName) {
    if (tabName === 'accuracy') {
        document.getElementById('accuracyTab').classList.add('active');
        document.getElementById('lossTab').classList.remove('active');
        const triggers = document.querySelectorAll('#researcherMode .card:has(#accuracyTab) .tab-trigger');
        if (triggers[0]) triggers[0].classList.add('active');
        if (triggers[1]) triggers[1].classList.remove('active');
    } else {
        document.getElementById('accuracyTab').classList.remove('active');
        document.getElementById('lossTab').classList.add('active');
        const triggers = document.querySelectorAll('#researcherMode .card:has(#accuracyTab) .tab-trigger');
        if (triggers[0]) triggers[0].classList.remove('active');
        if (triggers[1]) triggers[1].classList.add('active');
    }
}

function showLoading(show) {
    if (show) {
        console.log('⏳ Loading...');
        // Could add a loading overlay here
    } else {
        console.log('✓ Loading complete');
    }
}

function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // Simple alert for now - you can implement a better toast notification
    const icons = {
        'info': 'ℹ️',
        'success': '✓',
        'warning': '⚠️',
        'error': '✗'
    };
    
    alert(`${icons[type] || ''} ${message}`);
}