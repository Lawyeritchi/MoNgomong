// ===== config.js - PRODUCTION READY =====

// Auto-detect API URL (development vs production)
const getAPIBaseURL = () => {
    // Check if we're on localhost
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return 'http://localhost:5000';
    }
    
    // Production: Set your Railway/Render URL here after deployment
    // Example: 'https://your-app-name.up.railway.app'
    return 'https://your-backend-url.railway.app';  // ⚠️ GANTI INI SETELAH DEPLOY!
};

const CONFIG = {
    API_BASE_URL: getAPIBaseURL(),
    
    MEDIAPIPE: {
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    },
    
    // Letter Mode Config
    LETTER: {
        maxHands: 1,
        minConfidence: 0.20,
        debounceMs: 300,
        endpoint: '/api/predict'
    },
    
    // Word Mode Config
    WORD: {
        maxHands: 2,
        minConfidence: 0.40,
        debounceMs: 500,
        endpoint: '/api/predict-word'
    }
};

console.log('🌐 API URL:', CONFIG.API_BASE_URL);