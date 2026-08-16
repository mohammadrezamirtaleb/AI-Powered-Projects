/**
 * J.A.R.V.I.S. GLM-OCR & Vision Laboratory Controller
 */

class JarvisOCRLab {
    constructor() {
        this.currentImageBase64 = null;
        this.extractedText = '';
        this.init();
    }

    init() {
        this.dropzone = document.getElementById('ocrDropzone');
        this.fileInput = document.getElementById('ocrFileInput');
        this.previewImg = document.getElementById('ocrPreview');
        this.resultBox = document.getElementById('ocrResultBox');
        this.scanBtn = document.getElementById('ocrScanBtn');
        this.sendToPromptBtn = document.getElementById('ocrSendPromptBtn');
        this.copyBtn = document.getElementById('ocrCopyBtn');
        this.screenCaptureBtn = document.getElementById('ocrScreenCaptureBtn');
        this.modelSelect = document.getElementById('ocrModelSelect');

        this._bindEvents();
    }

    _bindEvents() {
        if (!this.dropzone) return;

        // Click to pick file
        this.dropzone.addEventListener('click', () => this.fileInput.click());

        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                this.loadImageFile(e.target.files[0]);
            }
        });

        // Drag & Drop
        ['dragenter', 'dragover'].forEach(name => {
            this.dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            this.dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.dropzone.classList.remove('dragover');
            });
        });

        this.dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                this.loadImageFile(e.dataTransfer.files[0]);
            }
        });

        // Global Paste (Ctrl+V) anywhere on page
        window.addEventListener('paste', (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (let item of items) {
                if (item.kind === 'file' && item.type.startsWith('image/')) {
                    const blob = item.getAsFile();
                    this.loadImageFile(blob);
                    // Switch to OCR tab if not active
                    if (window.switchHUDTab) window.switchHUDTab('tabOcr');
                    break;
                }
            }
        });

        // Screenshot capture button
        if (this.screenCaptureBtn) {
            this.screenCaptureBtn.addEventListener('click', () => this.captureScreenToLab());
        }

        // Scan button
        if (this.scanBtn) {
            this.scanBtn.addEventListener('click', () => this.executeScan());
        }

        // Copy button
        if (this.copyBtn) {
            this.copyBtn.addEventListener('click', () => {
                if (!this.extractedText) return;
                navigator.clipboard.writeText(this.extractedText);
                if (window.jarvisAudio) window.jarvisAudio.playClick();
                this.copyBtn.innerText = '✓ COPIED';
                setTimeout(() => { this.copyBtn.innerText = '📋 COPY'; }, 2000);
            });
        }

        // Send to Prompt button
        if (this.sendToPromptBtn) {
            this.sendToPromptBtn.addEventListener('click', () => {
                if (!this.extractedText) return;
                const promptInput = document.getElementById('chatPromptInput');
                if (promptInput) {
                    promptInput.value = `لطفاً متن استخراج‌شده زیر از تصویر/سند را بررسی، خلاصه‌سازی و تحلیل کن:\n\n\`\`\`\n${this.extractedText}\n\`\`\``;
                    promptInput.focus();
                    if (window.switchHUDTab) window.switchHUDTab('tabTerminal'); // or center view
                    if (window.jarvisAudio) window.jarvisAudio.playClick();
                }
            });
        }
    }

    loadImageFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.currentImageBase64 = e.target.result;
            this.previewImg.src = this.currentImageBase64;
            this.previewImg.style.display = 'block';
            this.dropzone.querySelector('.dropzone-text').style.display = 'none';
            this.resultBox.innerText = '// Image loaded. Ready for GLM-OCR neural pass...';
            this.scanBtn.disabled = false;
            if (window.jarvisAudio) window.jarvisAudio.playClick();
        };
        reader.readAsDataURL(file);
    }

    async captureScreenToLab() {
        if (window.jarvisAudio) window.jarvisAudio.playClick();
        this.resultBox.innerText = '// Capturing high-resolution desktop HUD screenshot...';

        try {
            const res = await fetch('/api/screenshot', { method: 'POST' });
            const data = await res.json();
            if (data.success && data.base64_data) {
                this.currentImageBase64 = data.base64_data;
                this.previewImg.src = data.base64_data;
                this.previewImg.style.display = 'block';
                this.dropzone.querySelector('.dropzone-text').style.display = 'none';
                this.resultBox.innerText = `// Screenshot captured [${data.filename}]. Ready for OCR neural pass...`;
                this.scanBtn.disabled = false;
                // Auto trigger scan
                this.executeScan();
            } else {
                this.resultBox.innerText = `// Capture error: ${data.error || 'Failed'}`;
            }
        } catch (e) {
            this.resultBox.innerText = `// Screenshot request failed: ${e.message}`;
        }
    }

    async executeScan() {
        if (!this.currentImageBase64) return;

        const selectedModel = this.modelSelect ? this.modelSelect.value : 'glm-ocr:latest';
        this.resultBox.innerText = `// Deploying ${selectedModel} neural vision weights...\n// Processing optical tensors in progress...`;
        this.scanBtn.disabled = true;

        if (window.arcReactor) window.arcReactor.setState('THINKING');
        if (window.jarvisAudio) window.jarvisAudio.playAlert();

        try {
            const res = await fetch('/api/ocr', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: this.currentImageBase64,
                    model: selectedModel,
                    prompt: "Extract all text, formulas, and code blocks from this image in clean Markdown."
                })
            });

            const data = await res.json();
            if (window.arcReactor) window.arcReactor.setState('IDLE');
            this.scanBtn.disabled = false;

            if (data.success && data.text) {
                this.extractedText = data.text;
                this.resultBox.innerText = data.text;
                if (this.sendToPromptBtn) this.sendToPromptBtn.style.display = 'inline-flex';
                if (this.copyBtn) this.copyBtn.style.display = 'inline-flex';
                if (window.jarvisAudio) window.jarvisAudio.playClick();
            } else {
                this.resultBox.innerText = `// OCR Extraction Failed: ${data.error || 'No recognizable text detected'}`;
            }
        } catch (e) {
            if (window.arcReactor) window.arcReactor.setState('IDLE');
            this.scanBtn.disabled = false;
            this.resultBox.innerText = `// Neural connection error: ${e.message}`;
        }
    }
}

window.jarvisOCR = new JarvisOCRLab();
