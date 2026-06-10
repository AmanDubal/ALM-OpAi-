// DOM Elements
const audioFileInput = document.getElementById('audioFile');
const processBtn = document.getElementById('processBtn');
const uploadBox = document.querySelector('.upload-box');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const downloadBtn = document.getElementById('downloadBtn');
const newBtn = document.getElementById('newBtn');
const errorCloseBtn = document.getElementById('errorCloseBtn');

let selectedFile = null;
let analysisResults = null;

// File Upload Handling
audioFileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    handleFileSelect(e.dataTransfer.files[0]);
});

function handleFileSelect(file) {
    if (file && file.type.startsWith('audio/')) {
        selectedFile = file;
        audioFileInput.files = new DataTransfer().items;
        
        const label = uploadBox.querySelector('.file-label');
        label.innerHTML = `<span>✓ ${file.name}</span><small>${(file.size / (1024 * 1024)).toFixed(2)} MB</small>`;
        
        processBtn.disabled = false;
    } else {
        showError('Please select a valid audio file');
        processBtn.disabled = true;
    }
}

// Process Button
processBtn.addEventListener('click', () => {
    if (selectedFile) {
        processAudio();
    }
});

// Download Results
downloadBtn.addEventListener('click', () => {
    if (analysisResults) {
        downloadResults();
    }
});

// New Audio
newBtn.addEventListener('click', () => {
    resetUI();
});

// Error Close
errorCloseBtn.addEventListener('click', () => {
    errorSection.classList.add('hidden');
});

// API Functions
async function processAudio() {
    if (!selectedFile) return;

    try {
        progressSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('audio_file', selectedFile);

        // Step 1: Upload and preprocess
        updateStep(1, 'active');
        const uploadResponse = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) throw new Error('Upload failed');
        const uploadData = await uploadResponse.json();
        updateStep(1, 'completed');

        // Step 2: Speech Recognition
        updateStep(2, 'active');
        const speechResponse = await fetch('/api/speech-recognition', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_id: uploadData.file_id })
        });

        if (!speechResponse.ok) throw new Error('Speech recognition failed');
        const speechData = await speechResponse.json();
        updateStep(2, 'completed');

        // Step 3: Sound Detection
        updateStep(3, 'active');
        const soundResponse = await fetch('/api/sound-detection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_id: uploadData.file_id })
        });

        if (!soundResponse.ok) throw new Error('Sound detection failed');
        const soundData = await soundResponse.json();
        updateStep(3, 'completed');

        // Step 4: Emotion Analysis
        updateStep(4, 'active');
        const emotionResponse = await fetch('/api/emotion-analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_id: uploadData.file_id })
        });

        if (!emotionResponse.ok) throw new Error('Emotion analysis failed');
        const emotionData = await emotionResponse.json();
        updateStep(4, 'completed');

        // Step 5: Context Integration
        updateStep(5, 'active');
        const contextResponse = await fetch('/api/context-integration', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_id: uploadData.file_id,
                transcript: speechData.transcript,
                sounds: soundData.events,
                emotion: emotionData.emotion
            })
        });

        if (!contextResponse.ok) throw new Error('Context integration failed');
        const contextData = await contextResponse.json();
        updateStep(5, 'completed');

        // Step 6: AI Reasoning
        updateStep(6, 'active');
        const inferenceResponse = await fetch('/api/inference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_id: uploadData.file_id,
                context: contextData.context,
                transcript: speechData.transcript
            })
        });

        if (!inferenceResponse.ok) throw new Error('Inference generation failed');
        const inferenceData = await inferenceResponse.json();
        updateStep(6, 'completed');

        // Display Results
        analysisResults = {
            audio: uploadData,
            speech: speechData,
            sound: soundData,
            emotion: emotionData,
            context: contextData,
            inference: inferenceData,
            timestamp: new Date().toISOString()
        };

        displayResults(analysisResults);

        // Hide progress and show results
        setTimeout(() => {
            progressSection.classList.add('hidden');
            resultsSection.classList.remove('hidden');
        }, 500);

    } catch (error) {
        console.error('Error:', error);
        showError(`Processing failed: ${error.message}`);
        resetProgressSteps();
    }
}

function updateStep(stepNum, status) {
    const step = document.getElementById(`step${stepNum}`);
    step.classList.remove('active', 'completed');
    if (status) {
        step.classList.add(status);
    }
}

function resetProgressSteps() {
    for (let i = 1; i <= 6; i++) {
        updateStep(i, null);
    }
}

function displayResults(results) {
    // Audio Information
    document.getElementById('duration').textContent = 
        results.audio.duration ? `${parseFloat(results.audio.duration).toFixed(2)}s` : '--';
    document.getElementById('sampleRate').textContent = 
        results.audio.sample_rate ? `${results.audio.sample_rate} Hz` : '--';
    document.getElementById('peakAmplitude').textContent = 
        results.audio.peak_amplitude ? parseFloat(results.audio.peak_amplitude).toFixed(3) : '--';

    // Speech Recognition
    document.getElementById('transcript').textContent = 
        results.speech.transcript || 'No speech detected';
    document.getElementById('language').textContent = 
        results.speech.language ? results.speech.language.toUpperCase() : '--';

    // Sound Events
    const soundEventsDiv = document.getElementById('soundEvents');
    if (results.sound.events && results.sound.events.length > 0) {
        soundEventsDiv.innerHTML = results.sound.events
            .map(event => `<div class="event-item">${event}</div>`)
            .join('');
    } else {
        soundEventsDiv.innerHTML = '<p>No sound events detected</p>';
    }

    // Emotion Analysis
    document.getElementById('emotion').textContent = 
        results.emotion.emotion ? results.emotion.emotion.toUpperCase() : '--';
    document.getElementById('emotionConfidence').textContent = 
        results.emotion.confidence ? `${(results.emotion.confidence * 100).toFixed(1)}%` : '--';

    // Context
    document.getElementById('context').textContent = 
        results.context.formatted_context || 'No context available';

    // Inference
    document.getElementById('inference').textContent = 
        results.inference.inference || 'No inference available';
}

function downloadResults() {
    if (!analysisResults) return;

    const timestamp = new Date().toLocaleString();
    const resultsText = `
╔═══════════════════════════════════════════════════════════════╗
║        AUDIO LANGUAGE MODEL (ALM) — ANALYSIS RESULTS         ║
║           Generated: ${timestamp}                    ║
╚═══════════════════════════════════════════════════════════════╝

1. AUDIO INFORMATION
   Duration        : ${document.getElementById('duration').textContent}
   Sample Rate     : ${document.getElementById('sampleRate').textContent}
   Peak Amplitude  : ${document.getElementById('peakAmplitude').textContent}

2. SPEECH RECOGNITION
   Transcript      : ${document.getElementById('transcript').textContent}
   Language        : ${document.getElementById('language').textContent}

3. SOUND EVENTS
   Detected        : ${document.getElementById('soundEvents').textContent}

4. EMOTION ANALYSIS
   Emotion         : ${document.getElementById('emotion').textContent}
   Confidence      : ${document.getElementById('emotionConfidence').textContent}

5. INTEGRATED CONTEXT
${document.getElementById('context').textContent}

6. AI INFERENCE (OpenRouter)
${document.getElementById('inference').textContent}

════════════════════════════════════════════════════════════════
    `;

    const blob = new Blob([resultsText], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `alm_analysis_${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    errorSection.classList.remove('hidden');
}

function resetUI() {
    selectedFile = null;
    analysisResults = null;
    audioFileInput.value = '';
    
    const label = uploadBox.querySelector('.file-label');
    label.innerHTML = '<span>Click to upload or drag and drop</span><small>Supported: WAV, MP3, M4A, FLAC, OGG (Max 50MB)</small>';
    
    processBtn.disabled = true;
    progressSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    resetProgressSteps();
}

// Initialize
console.log('✓ ALM Frontend loaded successfully');
