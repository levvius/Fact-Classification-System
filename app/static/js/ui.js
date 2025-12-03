/**
 * UI Controller for Fact Classification System
 * Handles all DOM manipulation and user interface updates
 */
class UIController {
    constructor() {
        // Cache DOM elements
        this.elements = {
            // Form elements
            form: document.getElementById('classify-form'),
            textInput: document.getElementById('text-input'),
            submitBtn: document.getElementById('submit-btn'),
            charCounter: document.getElementById('char-counter'),

            // Display sections
            topicsContainer: document.getElementById('topics-container'),
            resultsSection: document.getElementById('results-section'),
            overallResult: document.getElementById('overall-result'),
            claimsList: document.getElementById('claims-list'),

            // Status elements
            statusIndicator: document.getElementById('status-indicator'),
            statusText: document.getElementById('status-text'),

            // Overlay and errors
            loadingOverlay: document.getElementById('loading-overlay'),
            errorBanner: document.getElementById('error-banner'),
            errorMessage: document.getElementById('error-message'),
            errorClose: document.getElementById('error-close')
        };

        // Set up error close button
        this.elements.errorClose.addEventListener('click', () => this.hideError());
    }

    /**
     * Update health status indicator
     * @param {boolean} isHealthy - Whether models are loaded and ready
     * @param {string} status - Status text ('healthy', 'not_ready', 'error')
     */
    updateHealthStatus(isHealthy, status = 'healthy') {
        const indicator = this.elements.statusIndicator;
        const text = this.elements.statusText;

        indicator.className = 'status-indicator';

        if (isHealthy) {
            indicator.classList.add('healthy');
            text.textContent = 'API Ready';
        } else if (status === 'error') {
            indicator.classList.add('not-ready');
            text.textContent = 'API Error';
        } else {
            indicator.classList.add('checking');
            text.textContent = 'Models Loading...';
        }
    }

    /**
     * Render Wikipedia topics grouped by category
     * @param {Object} topicsData - Object with categories and topics
     */
    renderTopics(topicsData) {
        const container = this.elements.topicsContainer;
        container.innerHTML = '';

        const { categories } = topicsData;

        Object.entries(categories).forEach(([categoryName, topics]) => {
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'topic-category';

            // Category header (collapsible)
            const header = document.createElement('div');
            header.className = 'category-header';
            header.innerHTML = `
                <span class="category-icon">▼</span>
                <h3>${categoryName}</h3>
                <span style="color: #6b7280; font-size: 0.9rem;">${topics.length} topics</span>
            `;

            // Topics grid
            const grid = document.createElement('div');
            grid.className = 'topics-grid';

            topics.forEach(topic => {
                const card = document.createElement('div');
                card.className = 'topic-card';
                card.innerHTML = `<div class="topic-name">${topic}</div>`;

                // Click handler - insert example text
                card.addEventListener('click', () => {
                    this.insertExampleText(topic);
                });

                grid.appendChild(card);
            });

            // Toggle collapse on header click
            header.addEventListener('click', () => {
                const icon = header.querySelector('.category-icon');
                const isCollapsed = icon.classList.toggle('collapsed');
                grid.style.display = isCollapsed ? 'none' : 'grid';
            });

            categoryDiv.appendChild(header);
            categoryDiv.appendChild(grid);
            container.appendChild(categoryDiv);
        });
    }

    /**
     * Insert example text for a topic into the input field
     * @param {string} topic - Topic name
     */
    insertExampleText(topic) {
        const examples = {
            'Albert Einstein': 'Albert Einstein was born in 1879 and won the Nobel Prize in Physics in 1921.',
            'Barack Obama': 'Barack Obama was the 44th president of the United States and served two terms.',
            'World War II': 'World War II ended in 1945 and involved many countries around the world.',
            'COVID-19': 'COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus.',
            'Climate change': 'Climate change refers to long-term shifts in temperatures and weather patterns.',
            'default': `${topic} is a widely studied topic in modern science and has significant implications.`
        };

        const text = examples[topic] || examples['default'];
        this.elements.textInput.value = text;
        this.elements.textInput.focus();

        // Trigger input event to update character counter
        this.elements.textInput.dispatchEvent(new Event('input'));

        // Smooth scroll to input
        this.elements.textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    /**
     * Update character counter and submit button state
     * @param {number} length - Current text length
     */
    updateCharCounter(length) {
        const counter = this.elements.charCounter;
        const submitBtn = this.elements.submitBtn;

        counter.textContent = `${length} / 5000`;

        // Update color based on validity
        counter.className = 'char-counter';
        if (length < 10 || length > 5000) {
            counter.classList.add('invalid');
            submitBtn.disabled = true;
        } else {
            counter.classList.add('valid');
            submitBtn.disabled = false;
        }
    }

    /**
     * Show loading overlay
     */
    showLoading() {
        this.elements.loadingOverlay.style.display = 'flex';
        this.elements.submitBtn.disabled = true;
    }

    /**
     * Hide loading overlay
     */
    hideLoading() {
        this.elements.loadingOverlay.style.display = 'none';
        const length = this.elements.textInput.value.length;
        this.elements.submitBtn.disabled = (length < 10 || length > 5000);
    }

    /**
     * Show error banner
     * @param {string} message - Error message
     * @param {string} type - Error type ('validation', 'rate-limit', 'server-error', 'network')
     */
    showError(message, type = 'error') {
        this.elements.errorMessage.textContent = message;
        this.elements.errorBanner.style.display = 'flex';

        // Auto-hide after 10 seconds for non-critical errors
        if (type !== 'rate-limit') {
            setTimeout(() => this.hideError(), 10000);
        }

        // Scroll to error
        this.elements.errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    /**
     * Hide error banner
     */
    hideError() {
        this.elements.errorBanner.style.display = 'none';
    }

    /**
     * Render classification results
     * @param {Object} result - Classification response from API
     */
    renderResults(result) {
        // Show results section
        this.elements.resultsSection.style.display = 'block';

        // Render overall classification
        this.renderOverallResult(result);

        // Render individual claims
        this.renderClaims(result.claims);

        // Smooth scroll to results
        this.elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    /**
     * Render overall classification result
     * @param {Object} result - Classification response
     */
    renderOverallResult(result) {
        const { overall_classification, confidence } = result;
        const confidencePercent = Math.round(confidence * 100);

        this.elements.overallResult.innerHTML = `
            <div class="overall-classification">
                <div>Overall Classification:</div>
                <div class="classification-badge classification-${overall_classification}">
                    ${overall_classification.toUpperCase()}
                </div>
            </div>
            <div class="confidence-bar-container">
                <div class="confidence-label">Confidence: ${confidencePercent}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidencePercent}%">
                        ${confidencePercent}%
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render claims list with evidence
     * @param {Array} claims - Array of claim objects
     */
    renderClaims(claims) {
        this.elements.claimsList.innerHTML = '';

        if (!claims || claims.length === 0) {
            this.elements.claimsList.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: #6b7280;">
                    <p>No factual claims found in the text.</p>
                    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
                        Try entering text with specific facts, dates, or verifiable statements.
                    </p>
                </div>
            `;
            return;
        }

        claims.forEach((claim, index) => {
            const claimCard = this.createClaimCard(claim, index);
            this.elements.claimsList.appendChild(claimCard);
        });
    }

    /**
     * Create a claim card element
     * @param {Object} claim - Claim object
     * @param {number} index - Claim index
     * @returns {HTMLElement} Claim card element
     */
    createClaimCard(claim, index) {
        const card = document.createElement('div');
        card.className = 'claim-card';

        const confidencePercent = Math.round(claim.confidence * 100);

        card.innerHTML = `
            <div class="claim-header">
                <span class="claim-expand-icon">▶</span>
                <div class="claim-content">
                    <div class="claim-text">${this.escapeHtml(claim.claim)}</div>
                    <div class="claim-meta">
                        <span class="claim-classification classification-${claim.classification}">
                            ${claim.classification}
                        </span>
                        <span class="claim-confidence">
                            ${confidencePercent}% confident
                        </span>
                    </div>
                </div>
            </div>
            <div class="claim-details">
                ${this.renderEvidence(claim.best_evidence)}
            </div>
        `;

        // Add click handler for expand/collapse
        const header = card.querySelector('.claim-header');
        const details = card.querySelector('.claim-details');
        const icon = card.querySelector('.claim-expand-icon');

        header.addEventListener('click', () => {
            const isExpanded = details.classList.toggle('show');
            icon.classList.toggle('expanded', isExpanded);
            card.classList.toggle('expanded', isExpanded);
        });

        return card;
    }

    /**
     * Render evidence for a claim
     * @param {Object} evidence - Evidence object
     * @returns {string} HTML string for evidence
     */
    renderEvidence(evidence) {
        if (!evidence) {
            return '<p style="color: #6b7280;">No evidence found.</p>';
        }

        const nliPercent = Math.round(evidence.nli_score * 100);
        const retrievalPercent = Math.round(evidence.retrieval_score * 100);

        return `
            <div class="evidence-card">
                <div class="evidence-label">Best Evidence from Wikipedia</div>
                <div class="evidence-snippet">"${this.escapeHtml(evidence.snippet)}"</div>
                <div class="evidence-source">
                    <span>Source:</span>
                    <a href="${evidence.source}" target="_blank" rel="noopener noreferrer">
                        ${this.shortenUrl(evidence.source)}
                    </a>
                </div>
                <div class="evidence-scores">
                    <div class="score-item">
                        <span class="score-label">NLI Score</span>
                        <span class="score-value">${nliPercent}%</span>
                    </div>
                    <div class="score-item">
                        <span class="score-label">Retrieval Score</span>
                        <span class="score-value">${retrievalPercent}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Escape HTML to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Shorten URL for display
     * @param {string} url - Full URL
     * @returns {string} Shortened URL
     */
    shortenUrl(url) {
        try {
            const urlObj = new URL(url);
            return urlObj.hostname + urlObj.pathname.substring(0, 40) +
                (urlObj.pathname.length > 40 ? '...' : '');
        } catch {
            return url.substring(0, 50) + (url.length > 50 ? '...' : '');
        }
    }
}
