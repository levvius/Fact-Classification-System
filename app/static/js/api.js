/**
 * API Client for Fact Classification System
 * Handles all HTTP communication with the backend REST API
 */
class APIClient {
    constructor(baseURL = '/api/v1') {
        this.baseURL = baseURL;
    }

    /**
     * Check API health status
     * @returns {Promise<Object>} Health status object { status, models_loaded, kb_size }
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseURL}/health`);

            if (!response.ok) {
                throw new Error(`Health check failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Health check error:', error);
            return {
                status: 'error',
                models_loaded: false,
                kb_size: 0,
                error: error.message
            };
        }
    }

    /**
     * Get available Wikipedia topics from KB
     * @returns {Promise<Object>} Topics object { total_topics, categories }
     */
    async getTopics() {
        try {
            const response = await fetch(`${this.baseURL}/topics`);

            if (!response.ok) {
                throw new Error(`Failed to fetch topics: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Get topics error:', error);
            throw {
                type: 'network',
                message: 'Failed to load topics. Please check your connection.',
                original: error
            };
        }
    }

    /**
     * Classify text using NLI and Wikipedia evidence
     * @param {string} text - Text to classify (10-5000 chars)
     * @returns {Promise<Object>} Classification result
     */
    async classifyText(text) {
        try {
            const response = await fetch(`${this.baseURL}/classify`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text })
            });

            // Handle different error types
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));

                // Rate limit exceeded
                if (response.status === 429) {
                    const retryAfter = response.headers.get('Retry-After') || 60;
                    throw {
                        type: 'rate-limit',
                        message: `Rate limit exceeded. Please try again in ${retryAfter} seconds.`,
                        retryAfter: parseInt(retryAfter)
                    };
                }

                // Validation error (XSS, length, etc.)
                if (response.status === 422) {
                    const detail = errorData.detail || 'Invalid input';
                    throw {
                        type: 'validation',
                        message: Array.isArray(detail)
                            ? detail.map(e => e.msg).join(', ')
                            : detail
                    };
                }

                // Models not loaded yet
                if (response.status === 503) {
                    throw {
                        type: 'not-ready',
                        message: 'Models are still loading. Please wait a few seconds and try again.'
                    };
                }

                // Generic server error
                throw {
                    type: 'server-error',
                    message: errorData.message || `Server error: ${response.status}`
                };
            }

            return await response.json();
        } catch (error) {
            // If error is already formatted, re-throw
            if (error.type) {
                throw error;
            }

            // Network error or other issues
            console.error('Classification error:', error);
            throw {
                type: 'network',
                message: 'Network error. Please check your connection and try again.',
                original: error
            };
        }
    }

    /**
     * Get cache statistics (for debugging)
     * @returns {Promise<Object>} Cache info { size, maxsize }
     */
    async getCacheInfo() {
        try {
            const response = await fetch(`${this.baseURL}/cache-info`);

            if (!response.ok) {
                throw new Error(`Failed to get cache info: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Get cache info error:', error);
            return { size: 0, maxsize: 0, error: error.message };
        }
    }
}
