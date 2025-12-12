// ===== mediapipe.js - DUAL MODE SUPPORT =====
class MediaPipeHandler {
    constructor() {
        this.hands = null;
        this.camera = null;
        this.isInitialized = false;
        this.onResults = null;
        this.currentMaxHands = 1;
    }

    async initialize(videoElement, onResultsCallback, maxHands = 1) {
        this.onResults = onResultsCallback;
        this.currentMaxHands = maxHands;

        console.log(`🤖 Initializing MediaPipe with ${maxHands} hands`);

        // Initialize MediaPipe Hands
        this.hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });

        this.hands.setOptions({
            maxNumHands: maxHands,  // Dynamic: 1 or 2
            modelComplexity: CONFIG.MEDIAPIPE.modelComplexity,
            minDetectionConfidence: CONFIG.MEDIAPIPE.minDetectionConfidence,
            minTrackingConfidence: CONFIG.MEDIAPIPE.minTrackingConfidence,
        });

        this.hands.onResults((results) => {
            if (this.onResults) {
                this.onResults(results);
            }
        });

        // Initialize camera
        this.camera = new Camera(videoElement, {
            onFrame: async () => {
                await this.hands.send({ image: videoElement });
            },
            width: 1280,
            height: 720
        });

        this.isInitialized = true;
        return this.camera;
    }

    async updateMaxHands(maxHands) {
        /**
         * Update max hands dynamically when switching modes
         */
        if (!this.hands) return;
        
        this.currentMaxHands = maxHands;
        
        console.log(`🔄 Updating max hands to: ${maxHands}`);
        
        this.hands.setOptions({
            maxNumHands: maxHands,
            modelComplexity: CONFIG.MEDIAPIPE.modelComplexity,
            minDetectionConfidence: CONFIG.MEDIAPIPE.minDetectionConfidence,
            minTrackingConfidence: CONFIG.MEDIAPIPE.minTrackingConfidence,
        });
    }

    start() {
        if (this.camera) {
            this.camera.start();
        }
    }

    stop() {
        if (this.camera) {
            this.camera.stop();
        }
    }

    extractLandmarks(results, mode = 'letter') {
        /**
         * Extract landmarks based on mode
         * - Letter mode: 1 hand (126 features)
         * - Word mode: 1-2 hands (126 or 252 features)
         */
        if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
            return null;
        }

        const landmarks = [];
        const numHands = results.multiHandLandmarks.length;

        if (mode === 'letter') {
            // Letter mode: Always 1 hand (126 features)
            const hand = results.multiHandLandmarks[0];
            
            for (let landmark of hand) {
                landmarks.push(landmark.x, landmark.y, landmark.z);
            }
            
            // Duplicate for model compatibility (126 -> 126 x 2)
            for (let landmark of hand) {
                landmarks.push(1 - landmark.x, landmark.y, landmark.z);
            }
        } else {
            // Word mode: 1 or 2 hands
            if (numHands === 1) {
                // 1 hand: Extract 126 features
                const hand = results.multiHandLandmarks[0];
                for (let landmark of hand) {
                    landmarks.push(landmark.x, landmark.y, landmark.z);
                }
                
                // Duplicate for compatibility (126 x 2 = 252)
                for (let landmark of hand) {
                    landmarks.push(1 - landmark.x, landmark.y, landmark.z);
                }
            } else if (numHands >= 2) {
                // 2 hands: Extract 126 features from each (252 total)
                // Tapi karena model cuma 126, ambil hand 1 aja
                const hand = results.multiHandLandmarks[0];
                for (let landmark of hand) {
                    landmarks.push(landmark.x, landmark.y, landmark.z);
                }
                
                // Duplicate
                for (let landmark of hand) {
                    landmarks.push(1 - landmark.x, landmark.y, landmark.z);
                }
            }
        }

        return landmarks;
    }
}