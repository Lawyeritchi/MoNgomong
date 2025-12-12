// ===== ui.js - SIMPLIFIED =====
class UIManager {
    constructor() {
        this.elements = {
            // Status
            apiStatus: document.getElementById('apiStatus'),
            cameraStatus: document.getElementById('cameraStatus'),
            detectionStatus: document.getElementById('detectionStatus'),
            
            // Buttons
            startBtn: document.getElementById('startBtn'),
            detectBtn: document.getElementById('detectBtn'),
            stopBtn: document.getElementById('stopBtn'),
            clearBtn: document.getElementById('clearBtn'),
            themeToggle: document.getElementById('themeToggle'),
            
            // Video
            videoElement: document.getElementById('videoElement'),
            canvasElement: document.getElementById('canvasElement'),
            videoOverlay: document.getElementById('videoOverlay'),
            fpsCounter: document.getElementById('fpsCounter'),
            
            // Results
            predictionLetter: document.getElementById('predictionLetter'),
            confidenceValue: document.getElementById('confidenceValue'),
            confidenceFill: document.getElementById('confidenceFill'),
            translationText: document.getElementById('translationText'),
            historyList: document.getElementById('historyList')
        };

        this.initTheme();
    }

    initTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
    }

    toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const newTheme = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        this.updateThemeIcon(newTheme);
    }

    updateThemeIcon(theme) {
        const icon = this.elements.themeToggle.querySelector('.theme-icon');
        icon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }

    updateStatus(type, value, color = '') {
        const element = this.elements[`${type}Status`];
        if (element) {
            element.textContent = value;
            if (color) {
                element.style.color = color;
            }
        }
    }

    updatePrediction(letter, confidence) {
        this.elements.predictionLetter.textContent = letter;
        this.elements.confidenceValue.textContent = `${confidence}%`;
        this.elements.confidenceFill.style.width = `${confidence}%`;
    }

    appendTranslation(letter) {
        // Remove empty state if exists
        const emptyState = this.elements.translationText.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }
        
        // Append letter
        this.elements.translationText.textContent += letter;
    }

    clearTranslation() {
        this.elements.translationText.innerHTML = '<div class="empty-state">Belum ada terjemahan</div>';
    }

    addToHistory(letter, confidence) {
        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `
            <span class="history-letter">${letter}</span>
            <span class="history-confidence">${confidence}%</span>
        `;

        if (this.elements.historyList.querySelector('.empty-state')) {
            this.elements.historyList.innerHTML = '';
        }

        this.elements.historyList.insertBefore(item, this.elements.historyList.firstChild);

        // Keep only last 5
        while (this.elements.historyList.children.length > 5) {
            this.elements.historyList.removeChild(this.elements.historyList.lastChild);
        }
    }

    updateFPS(fps) {
        this.elements.fpsCounter.textContent = `${fps} FPS`;
    }

    hideVideoOverlay() {
        this.elements.videoOverlay.style.display = 'none';
    }

    showVideoOverlay() {
        this.elements.videoOverlay.style.display = 'flex';
    }
}