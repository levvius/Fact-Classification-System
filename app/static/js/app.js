/**
 * Main Application Logic for Fact Classification System
 * Initializes and coordinates API client and UI controller
 */

// Global instances
let api;
let ui;

/**
 * Initialize the application on page load
 */
document.addEventListener('DOMContentLoaded', async () => {
    // Create API client and UI controller instances
    api = new APIClient();
    ui = new UIController();

    // Initial health check
    await checkHealth();

    // Load and display topics
    await loadTopics();

    // Set up event listeners
    setupEventListeners();

    // Start periodic health check (every 30 seconds)
    setInterval(checkHealth, 30000);
});

/**
 * Check API health status
 */
async function checkHealth() {
    try {
        const health = await api.checkHealth();
        ui.updateHealthStatus(health.models_loaded, health.status);

        if (!health.models_loaded) {
            console.warn('Models not loaded yet. Waiting for initialization...');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        ui.updateHealthStatus(false, 'error');
    }
}

/**
 * Load Wikipedia topics from API
 */
async function loadTopics() {
    try {
        const topics = await api.getTopics();
        ui.renderTopics(topics);
    } catch (error) {
        console.error('Failed to load topics:', error);
        ui.showError(error.message || 'Failed to load topics', 'network');
    }
}

/**
 * Set up all event listeners
 */
function setupEventListeners() {
    const { form, textInput } = ui.elements;

    // Form submission
    form.addEventListener('submit', handleFormSubmit);

    // Character counter update
    textInput.addEventListener('input', () => {
        const length = textInput.value.length;
        ui.updateCharCounter(length);
    });

    // Initialize character counter
    ui.updateCharCounter(textInput.value.length);
}

/**
 * Handle form submission
 * @param {Event} event - Form submit event
 */
async function handleFormSubmit(event) {
    event.preventDefault();

    const text = ui.elements.textInput.value.trim();

    // Client-side validation
    if (text.length < 10) {
        ui.showError('Text must be at least 10 characters long.', 'validation');
        return;
    }

    if (text.length > 5000) {
        ui.showError('Text must not exceed 5000 characters.', 'validation');
        return;
    }

    // Hide any previous errors
    ui.hideError();

    // Show loading state
    ui.showLoading();

    try {
        // Call classification API
        const result = await api.classifyText(text);

        // Hide loading and render results
        ui.hideLoading();
        ui.renderResults(result);

    } catch (error) {
        // Hide loading
        ui.hideLoading();

        // Display appropriate error message
        handleClassificationError(error);
    }
}

/**
 * Handle classification errors
 * @param {Object} error - Error object with type and message
 */
function handleClassificationError(error) {
    console.error('Classification error:', error);

    switch (error.type) {
        case 'server-offline':
            ui.showError(
                `🔴 API Server Not Running\n\n${error.message}\n\n${error.suggestion}`,
                'server-offline'
            );
            break;

        case 'rate-limit':
            ui.showError(
                `⚠️ ${error.message}\n\nYou can make 10 requests per minute. Please wait before trying again.`,
                'rate-limit'
            );
            break;

        case 'validation':
            ui.showError(
                `❌ Validation Error: ${error.message}`,
                'validation'
            );
            break;

        case 'not-ready':
            ui.showError(
                `⏳ ${error.message}\n\n${error.suggestion || 'Please wait a moment and try again.'}`,
                'not-ready'
            );
            // Retry health check after 5 seconds
            setTimeout(checkHealth, 5000);
            break;

        case 'server-error':
            ui.showError(
                `🔴 Server Error (${error.status || 'Unknown'}): ${error.message}\n\nPlease check the server logs or try again.`,
                'server-error'
            );
            break;

        case 'network':
            ui.showError(
                `🌐 Network Error: ${error.message}`,
                'network'
            );
            break;

        default:
            ui.showError(
                `❌ An unexpected error occurred: ${error.message || 'Unknown error'}`,
                'error'
            );
    }
}

/**
 * Log application info to console (for debugging)
 */
console.log('%c Fact Classification System ', 'background: #667eea; color: white; font-size: 16px; font-weight: bold; padding: 4px 8px; border-radius: 4px;');
console.log('API Base URL:', '/api/v1');
console.log('Documentation:', window.location.origin + '/docs');
console.log('GitHub:', 'https://github.com/levvius/IS-hallucination-detection');
