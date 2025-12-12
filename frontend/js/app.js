// ===== app.js - DUAL MODE VERSION =====
class SignSpeakApp {
    constructor() {
        this.ui = new UIManager();
        this.mediaPipe = new MediaPipeHandler();
        
        this.isDetecting = false;
        this.isCameraActive = false;
        this.currentMode = 'letter';  // 'letter' or 'word'
        this.lastPrediction = null;
        this.lastPredictionTime = 0;
        
        this.fpsTracker = {
            frameCount: 0,
            lastTime: performance.now(),
            fps: 0
        };

        this.init();
    }

    async init() {
        await this.checkAPIHealth();
        this.setupEventListeners();
        this.setupCanvas();
    }

    async checkAPIHealth() {
        this.ui.updateStatus('api', 'Checking...', 'orange');
        const health = await api.checkHealth();
        
        if (health.status === 'healthy') {
            const letterOK = health.letter_model_loaded ? '✓' : '✗';
            const wordOK = health.word_model_loaded ? '✓' : '✗';
            
            this.ui.updateStatus('api', `Connected (L:${letterOK} W:${wordOK})`, 'green');
        } else {
            this.ui.updateStatus('api', 'Error ✗', 'red');
            alert('Backend API tidak tersedia! Pastikan Flask server berjalan.');
        }
    }

    setupEventListeners() {
        // Buttons
        this.ui.elements.startBtn.addEventListener('click', () => this.startCamera());
        this.ui.elements.detectBtn.addEventListener('click', () => this.toggleDetection());
        this.ui.elements.stopBtn.addEventListener('click', () => this.stop());
        this.ui.elements.clearBtn.addEventListener('click', () => {
            this.ui.clearTranslation();
            this.lastPrediction = null;
        });

        // Mode toggle buttons
        const modeButtons = document.querySelectorAll('.mode-btn');
        modeButtons.forEach(btn => {
            btn.addEventListener('click', async () => {
                if (btn.disabled) return;
                
                const mode = btn.dataset.mode;
                await this.switchMode(mode);
            });
        });

        // Theme
        this.ui.elements.themeToggle.addEventListener('click', () => {
            this.ui.toggleTheme();
        });
    }

    async switchMode(mode) {
        console.log(`🔄 Switching to ${mode} mode...`);
        
        // Update active button
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.mode === mode) {
                btn.classList.add('active');
            }
        });
        
        this.currentMode = mode;
        this.lastPrediction = null;
        
        // Update UI
        if (mode === 'letter') {
            this.ui.updateStatus('detection', 'Mode: Huruf', 'blue');
        } else {
            this.ui.updateStatus('detection', 'Mode: Kata', 'blue');
        }
        
        // Update MediaPipe max hands if camera is active
        if (this.isCameraActive) {
            const maxHands = mode === 'letter' ? CONFIG.LETTER.maxHands : CONFIG.WORD.maxHands;
            await this.mediaPipe.updateMaxHands(maxHands);
        }
        
        console.log(`✅ Switched to ${mode} mode`);
    }

    setupCanvas() {
        const canvas = this.ui.elements.canvasElement;
        const video = this.ui.elements.videoElement;
        
        const updateCanvasSize = () => {
            canvas.width = video.videoWidth || 1280;
            canvas.height = video.videoHeight || 720;
        };

        video.addEventListener('loadedmetadata', updateCanvasSize);
        updateCanvasSize();
    }

    async startCamera() {
        try {
            this.ui.updateStatus('camera', 'Starting...', 'orange');
            
            const maxHands = this.currentMode === 'letter' ? CONFIG.LETTER.maxHands : CONFIG.WORD.maxHands;
            
            await this.mediaPipe.initialize(
                this.ui.elements.videoElement,
                (results) => this.onMediaPipeResults(results),
                maxHands
            );

            await this.mediaPipe.start();
            
            this.isCameraActive = true;
            this.ui.updateStatus('camera', 'Active ✓', 'green');
            this.ui.hideVideoOverlay();
            
            this.ui.elements.startBtn.disabled = true;
            this.ui.elements.detectBtn.disabled = false;
            this.ui.elements.stopBtn.disabled = false;
            
        } catch (error) {
            console.error('Camera start error:', error);
            this.ui.updateStatus('camera', 'Error ✗', 'red');
            alert('Gagal mengakses kamera!');
        }
    }

    toggleDetection() {
        this.isDetecting = !this.isDetecting;
        
        if (this.isDetecting) {
            const modeText = this.currentMode === 'letter' ? 'Huruf' : 'Kata';
            this.ui.updateStatus('detection', `Active (${modeText}) ✓`, 'green');
            this.ui.elements.detectBtn.innerHTML = '<span class="btn-icon">⏸️</span><span>Pause</span>';
            this.ui.elements.detectBtn.classList.remove('btn-success');
            this.ui.elements.detectBtn.classList.add('btn-warning');
        } else {
            this.ui.updateStatus('detection', 'Paused', 'orange');
            this.ui.elements.detectBtn.innerHTML = '<span class="btn-icon">🎯</span><span>Resume</span>';
            this.ui.elements.detectBtn.classList.remove('btn-warning');
            this.ui.elements.detectBtn.classList.add('btn-success');
        }
    }

    stop() {
        this.mediaPipe.stop();
        this.isDetecting = false;
        this.isCameraActive = false;
        
        this.ui.updateStatus('camera', 'Stopped', 'gray');
        this.ui.updateStatus('detection', 'Inactive', 'gray');
        this.ui.showVideoOverlay();
        
        this.ui.elements.startBtn.disabled = false;
        this.ui.elements.detectBtn.disabled = true;
        this.ui.elements.stopBtn.disabled = true;
        
        const canvas = this.ui.elements.canvasElement;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    async onMediaPipeResults(results) {
        this.drawResults(results);
        this.updateFPS();
        
        if (this.isDetecting && results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = this.mediaPipe.extractLandmarks(results, this.currentMode);
            
            if (landmarks) {
                const now = performance.now();
                const config = this.currentMode === 'letter' ? CONFIG.LETTER : CONFIG.WORD;
                
                if (now - this.lastPredictionTime > config.debounceMs) {
                    this.lastPredictionTime = now;
                    await this.makePrediction(landmarks);
                }
            }
        } else if (this.isDetecting) {
            this.ui.updatePrediction('-', 0);
        }
    }

    async makePrediction(landmarks) {
        try {
            const config = this.currentMode === 'letter' ? CONFIG.LETTER : CONFIG.WORD;
            const endpoint = config.endpoint;
            
            const result = await api.predictByMode(landmarks, this.currentMode);
            
            if (result.status === 'success' && result.prediction) {
                const text = this.currentMode === 'letter' ? result.prediction.letter : result.prediction.word;
                const confidence_percent = result.prediction.confidence_percent;
                
                console.log(`🎯 [${this.currentMode}] Predicted: ${text} (${confidence_percent}%)`);
                
                // Always update UI
                this.ui.updatePrediction(text, confidence_percent);
                
                // Append if confidence is high enough
                if (confidence_percent >= config.minConfidence * 100) {
                    if (!this.lastPrediction || this.lastPrediction.text !== text) {
                        console.log(`✅ Appending: ${text}`);
                        
                        if (this.currentMode === 'letter') {
                            this.ui.appendTranslation(text);
                        } else {
                            this.ui.appendTranslation(` ${text} `);  // Spasi untuk kata
                        }
                        
                        this.ui.addToHistory(text, confidence_percent);
                        this.lastPrediction = { text, confidence_percent };
                    }
                }
            }
        } catch (error) {
            console.error('❌ Prediction error:', error);
        }
    }

    drawResults(results) {
        const canvas = this.ui.elements.canvasElement;
        const ctx = canvas.getContext('2d');
        
        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (results.multiHandLandmarks) {
            for (const landmarks of results.multiHandLandmarks) {
                drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                    color: '#00FF00',
                    lineWidth: 2
                });
                
                drawLandmarks(ctx, landmarks, {
                    color: '#FF0000',
                    lineWidth: 1,
                    radius: 3
                });
            }
        }
        
        ctx.restore();
    }

    updateFPS() {
        this.fpsTracker.frameCount++;
        const now = performance.now();
        const elapsed = now - this.fpsTracker.lastTime;
        
        if (elapsed >= 1000) {
            this.fpsTracker.fps = Math.round((this.fpsTracker.frameCount * 1000) / elapsed);
            this.ui.updateFPS(this.fpsTracker.fps);
            this.fpsTracker.frameCount = 0;
            this.fpsTracker.lastTime = now;
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new SignSpeakApp();
    console.log('✅ SignSpeak App initialized (Dual Mode)');
});