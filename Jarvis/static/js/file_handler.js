class FileHandler {
    constructor() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'drag-drop-overlay';
        this.overlay.innerHTML = '<h2>Drop Files to Upload to Neural Vault</h2>';
        document.body.appendChild(this.overlay);

        this.bindEvents();
    }

    bindEvents() {
        window.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.overlay.style.display = 'flex';
        });

        window.addEventListener('dragleave', (e) => {
            e.preventDefault();
            if (e.clientX === 0 || e.clientY === 0) {
                this.overlay.style.display = 'none';
            }
        });

        window.addEventListener('drop', (e) => {
            e.preventDefault();
            this.overlay.style.display = 'none';

            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                this.handleFiles(e.dataTransfer.files);
            }
        });
    }

    async handleFiles(files) {
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            console.log("Processing dropped file:", file.name);

            // Simple integration: Read as text and add as note to Vault
            if (file.type.startsWith('text/') || file.name.endsWith('.md') || file.name.endsWith('.py') || file.name.endsWith('.js')) {
                const reader = new FileReader();
                reader.onload = async (e) => {
                    const text = e.target.result;
                    if (window.jarvisApp) {
                        try {
                            const res = await fetch('/api/vault/notes', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    title: `File: ${file.name}`,
                                    content: text.substring(0, 10000) // Truncate large files
                                })
                            });
                            if (res.ok) {
                                window.jarvisApp.logTerminal(`FILE STORED: ${file.name}`);
                                window.jarvisApp.loadVaultNotes();
                            }
                        } catch (err) {
                            console.error(err);
                        }
                    }
                };
                reader.readAsText(file);
            } else if (file.type.startsWith('image/')) {
                // Image handling: load to OCR
                const reader = new FileReader();
                reader.onload = (e) => {
                    const base64 = e.target.result;
                    if (window.jarvisOCR) {
                        window.jarvisOCR.currentImageBase64 = base64;
                        if (window.jarvisOCR.previewImg) {
                            window.jarvisOCR.previewImg.src = base64;
                            window.jarvisOCR.previewImg.style.display = 'block';
                            window.jarvisApp.switchTab('tabOcr');
                        }
                    }
                };
                reader.readAsDataURL(file);
            } else {
                if (window.jarvisApp) {
                    window.jarvisApp.logTerminal(`UNSUPPORTED FILE TYPE: ${file.name}`);
                }
            }
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.fileHandler = new FileHandler();
});
