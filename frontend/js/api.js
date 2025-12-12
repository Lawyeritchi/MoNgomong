// ===== api.js - FIXED VERSION =====
class API {
    constructor(baseURL) {
        this.baseURL = baseURL;
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.baseURL}/api/health`); // FIX: pakai backtick
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'error', model_loaded: false };
        }
    }

    async getModelInfo() {
        try {
            const response = await fetch(`${this.baseURL}/api/model-info`); // FIX: pakai backtick
            return await response.json();
        } catch (error) {
            console.error('Get model info failed:', error);
            return { status: 'error' };
        }
    }

    async predict(landmarks) {
        const startTime = performance.now();
        
        try {
            const response = await fetch(`${this.baseURL}/api/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ landmarks })
            });

            const data = await response.json();
            const latency = Math.round(performance.now() - startTime);
            
            console.log('✅ Letter prediction received:', data);
            
            return { ...data, latency };
        } catch (error) {
            console.error('❌ Prediction failed:', error);
            return { 
                status: 'error', 
                message: error.message,
                latency: Math.round(performance.now() - startTime)
            };
        }
    }

    async predictWord(landmarks) {
        const startTime = performance.now();
        
        try {
            const response = await fetch(`${this.baseURL}/api/predict-word`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ landmarks })
            });

            const data = await response.json();
            const latency = Math.round(performance.now() - startTime);
            
            console.log('✅ Word prediction received:', data);
            
            return { ...data, latency };
        } catch (error) {
            console.error('❌ Word prediction failed:', error);
            return { 
                status: 'error', 
                message: error.message,
                latency: Math.round(performance.now() - startTime)
            };
        }
    }

    async predictByMode(landmarks, mode) {
        if (mode === 'letter') {
            return await this.predict(landmarks);
        } else {
            return await this.predictWord(landmarks);
        }
    }
}

const api = new API(CONFIG.API_BASE_URL);