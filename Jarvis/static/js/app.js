/**
 * J.A.R.V.I.S. Mark-85 Master Application Controller
 * Connects SSE Streaming (OpenRouter & Ollama), Neural Markdown, Audio/Visuals, Terminal & Memory Vault
 */

class JarvisApp {
    constructor() {
        this.conversationHistory = [];
        this.activeConversationId = null;
        this.isStreaming = false;
        this.activeTab = 'tabOcr';
        this.activeProvider = 'openrouter';
        this.activeModel = 'google/gemma-4-26b-a4b-it:free';
        
        // Load initial TTS settings
        this.ttsEngine = localStorage.getItem('jarvis_tts_engine') || 'edge';
        this.ttsVoice = localStorage.getItem('jarvis_tts_voice') || 'en_US-Male';
        
        this.init();
    }

    async init() {
        this.chatStream = document.getElementById('chatStream');
        this.promptInput = document.getElementById('chatPromptInput');
        this.sendBtn = document.getElementById('chatSendBtn');
        this.terminalOutput = document.getElementById('terminalOutput');
        this.terminalInput = document.getElementById('terminalInput');
        this.sfxToggleBtn = document.getElementById('sfxToggleBtn');
        this.voiceToggleBtn = document.getElementById('voiceToggleBtn');
        this.fullscreenToggleBtn = document.getElementById('fullscreenToggleBtn');
        this.headerModelSelect = document.getElementById('headerModelSelect');

        // 1. Run Epic 3-Second Fullscreen Arc Reactor Boot Sequence IMMEDIATELY (0ms start)
        this._runFullscreenBootSequence();

        // 2. Bind UI events & load configs asynchronously in background
        this._bindEvents();
        this._bindTabs();
        this._bindConfigModal();
        this._bindChatUI();
        this._bindKeyboardShortcuts();
        this._initNotifications();
        
        await this.loadConversations();
        await this.startNewConversation();
    }

    async loadConversations() {
        try {
            const res = await fetch('/api/conversations');
            if (res.ok) {
                const convos = await res.json();
                this.renderConversationList(convos);
            }
        } catch (e) {
            console.error("Error loading conversations", e);
        }
    }

    renderConversationList(convos) {
        const listEl = document.getElementById('chatHistoryList');
        if (!listEl) return;
        listEl.innerHTML = '';
        convos.forEach(c => {
            const btn = document.createElement('div');
            btn.className = `action-badge ${c.id === this.activeConversationId ? 'active-convo' : ''}`;
            btn.style.cursor = 'pointer';
            btn.innerHTML = `<span>💬 ${c.title}</span>`;
            btn.onclick = () => this.loadConversation(c.id);
            listEl.appendChild(btn);
        });
    }

    async startNewConversation() {
        try {
            const res = await fetch('/api/conversations', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({title: 'Mark-86 Session'})
            });
            if (res.ok) {
                const data = await res.json();
                this.activeConversationId = data.id;
                this.conversationHistory = [];
                this.chatStream.innerHTML = ''; // clear chat
                await this.loadConversations();
            }
        } catch (e) {
            console.error(e);
        }
    }

    async loadConversation(cid) {
        try {
            const res = await fetch(`/api/conversations/${cid}`);
            if (res.ok) {
                const messages = await res.json();
                this.activeConversationId = cid;
                this.conversationHistory = messages;
                this.chatStream.innerHTML = '';
                messages.forEach(m => {
                    if (m.role === 'user') {
                        this.addUserMessage(m.content);
                    } else if (m.role === 'assistant') {
                        const jarvisCard = this.createJarvisPlaceholder();
                        this.handleGenerationDone(jarvisCard, m.content, [], false);
                    }
                });
                await this.loadConversations();
            }
        } catch (e) {
            console.error(e);
        }
    }

    _bindKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K: Focus chat prompt
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.promptInput.focus();
            }
            // Ctrl/Cmd + N: New Chat
            if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
                e.preventDefault();
                this.startNewConversation();
            }
            // Ctrl/Cmd + Shift + S: Screenshot
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'S') {
                e.preventDefault();
                // Send screenshot action silently
                const btn = document.getElementById('ocrScreenCaptureBtn');
                if (btn) btn.click();
            }
            // Escape: Close Modals
            if (e.key === 'Escape') {
                const modal = document.getElementById('configModalOverlay');
                if (modal && modal.classList.contains('active')) {
                    modal.classList.remove('active');
                }
            }
            // Ctrl/Cmd + C (only if no text is selected)
            if ((e.ctrlKey || e.metaKey) && e.key === 'c' && !window.getSelection().toString()) {
                if (this.isStreaming && this.activeWs) {
                    e.preventDefault();
                    this.activeWs.close();
                    this.logTerminal("STREAM CANCELLED BY USER");
                }
            }
        });
    }

    _initNotifications() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.notifWs = new WebSocket(`${protocol}//${window.location.host}/ws/notifications`);
        this.notifWs.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'notification') {
                    this.showNotification(data.title, data.message);
                }
            } catch (e) {}
        };
        this.notifWs.onclose = () => {
            setTimeout(() => this._initNotifications(), 5000); // Reconnect
        };
    }

    showNotification(title, message) {
        const container = document.getElementById('notificationContainer');
        if (!container) return;

        const notif = document.createElement('div');
        notif.style.background = 'rgba(0, 20, 40, 0.9)';
        notif.style.border = '1px solid var(--stark-cyan)';
        notif.style.borderRadius = '4px';
        notif.style.padding = '10px 15px';
        notif.style.color = 'var(--stark-cyan)';
        notif.style.fontFamily = 'var(--font-sci-fi)';
        notif.style.boxShadow = '0 0 15px rgba(0, 240, 255, 0.2)';
        notif.style.animation = 'fadeInMsg 0.3s ease-out';
        notif.innerHTML = `<strong style="display:block; font-size:0.85rem;">⚠️ ${title}</strong><span style="font-size:0.75rem;">${message}</span>`;
        
        container.appendChild(notif);
        
        if (window.jarvisAudio) window.jarvisAudio.playAlert();
        
        setTimeout(() => {
            notif.style.opacity = '0';
            notif.style.transition = 'opacity 0.5s';
            setTimeout(() => notif.remove(), 500);
        }, 5000);
    }

    _bindChatUI() {
        this.loadVaultNotes();
    }

    _runFullscreenBootSequence() {
        const bootOverlay = document.getElementById('starkBootOverlay');
        const progressFill = document.getElementById('bootProgressFill');
        const bootStatusText = document.getElementById('bootStatusText');

        // Init Hero Fullscreen Arc Reactor
        let heroReactor = null;
        if (window.initArcReactor && document.getElementById('heroReactorCanvas')) {
            heroReactor = window.initArcReactor('heroReactorCanvas');
            heroReactor.setState('OVERCHARGE');
        }

        // Play SFX & Speak Greeting IMMEDIATELY (0 delay)
        if (window.jarvisAudio) {
            window.jarvisAudio.playBoot();
            window.jarvisAudio.speakGreeting(
                "Good day, Sir. All Mark-85 systems are online and fully operational. Standing by for your directives."
            );
            if (heroReactor) {
                window.jarvisAudio.onSpeakingStateChange = (speaking) => {
                    if (heroReactor) heroReactor.setState(speaking ? 'SPEAKING' : 'OVERCHARGE');
                    if (window.arcReactor) window.arcReactor.setState(speaking ? 'SPEAKING' : 'IDLE');
                };
            }
        }

        // Animate progress bar over 3.0 seconds
        if (progressFill) {
            setTimeout(() => { progressFill.style.width = '100%'; }, 50);
        }

        // Dynamic boot status milestones
        setTimeout(() => {
            if (bootStatusText) bootStatusText.innerText = "SYNCHRONIZING NEURAL LINK WITH OPENROUTER & GLM-OCR...";
        }, 1100);

        setTimeout(() => {
            if (bootStatusText) bootStatusText.innerText = "ALL SYSTEMS NOMINAL. ENGAGING MAIN HUD INTERFACE...";
        }, 2200);

        // Transition from Fullscreen Orb to Main UI at 3.0 seconds
        setTimeout(() => {
            if (bootOverlay) {
                bootOverlay.classList.add('boot-completed');
                setTimeout(() => { bootOverlay.style.display = 'none'; }, 850);
            }

            // Init Main Sidebar Arc Reactor
            window.arcReactor = window.initArcReactor('arcReactorCanvas');
            if (window.jarvisAudio) {
                window.jarvisAudio.onSpeakingStateChange = (speaking) => {
                    if (window.arcReactor) {
                        window.arcReactor.setState(speaking ? 'SPEAKING' : 'IDLE');
                    }
                };
            }

            this.logTerminal("J.A.R.V.I.S. Mark-85 Core Initialized.");
            this.logTerminal(`Active Neural Link: ${this.activeProvider.toUpperCase()} [${this.activeModel}]`);

            // Add Pure English Welcome Message in Chat (Voice already spoken by speakGreeting)
            const providerName = this.activeProvider === 'openrouter' ? 'OpenRouter Cloud (Gemma-4 26B)' : 'Ollama Local (Qwen 3.5 4B)';
            const welcomeText = `Good day, Commander. All Mark-85 neural pathways are online and operating at peak nominal capacity.\n\nNeural link established with **${providerName}** and **GLM-OCR Vision Engine**. Standing by for your directives, Sir.`;
            this.addJarvisMessage(welcomeText, [], false);
        }, 3000);
    }

    _bindEvents() {
        // Send button
        this.sendBtn.addEventListener('click', () => this.handleUserSubmit());

        // Input Enter key
        this.promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleUserSubmit();
            }
        });

        // Quick prompt chips
        document.querySelectorAll('.chip-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const prompt = btn.dataset.prompt;
                if (prompt) {
                    this.promptInput.value = prompt;
                    this.handleUserSubmit();
                }
            });
        });

        // Header Model Dropdown change
        if (this.headerModelSelect) {
            this.headerModelSelect.addEventListener('change', async (e) => {
                const [prov, model] = e.target.value.split('|');
                this.activeProvider = prov;
                this.activeModel = model;
                this.updateProviderHUD();
                if (window.jarvisAudio) window.jarvisAudio.playClick();

                // Persist to backend
                try {
                    await fetch('/api/config/provider', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            active_provider: prov,
                            openrouter_model: prov === 'openrouter' ? model : undefined,
                            ollama_model: prov === 'ollama' ? model : undefined
                        })
                    });
                    this.logTerminal(`[SYSTEM]: Neural Provider switched to ${prov.toUpperCase()} (${model})`);
                } catch (err) {}
            });
        }

        // Terminal command input
        if (this.terminalInput) {
            this.terminalInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.executeTerminalCmd(this.terminalInput.value);
                    this.terminalInput.value = '';
                }
            });
        }

        // SFX toggle
        if (this.sfxToggleBtn) {
            this.sfxToggleBtn.addEventListener('click', () => {
                window.jarvisAudio.sfxEnabled = !window.jarvisAudio.sfxEnabled;
                this.sfxToggleBtn.classList.toggle('active', window.jarvisAudio.sfxEnabled);
                this.sfxToggleBtn.innerText = window.jarvisAudio.sfxEnabled ? '🔊 SFX: ON' : '🔈 SFX: OFF';
            });
        }

        // Voice toggle
        if (this.voiceToggleBtn) {
            this.voiceToggleBtn.addEventListener('click', () => {
                window.jarvisAudio.voiceEnabled = !window.jarvisAudio.voiceEnabled;
                this.voiceToggleBtn.classList.toggle('active', window.jarvisAudio.voiceEnabled);
                this.voiceToggleBtn.innerText = window.jarvisAudio.voiceEnabled ? '🎙️ VOICE: ON' : '🔇 VOICE: OFF';
                if (!window.jarvisAudio.voiceEnabled) window.jarvisAudio.stopSpeaking();
            });
        }

        // Fullscreen Toggle
        if (this.fullscreenToggleBtn) {
            this.fullscreenToggleBtn.addEventListener('click', () => {
                if (window.jarvisAudio) window.jarvisAudio.playClick();
                if (!document.fullscreenElement) {
                    document.documentElement.requestFullscreen().catch(err => {
                        console.warn("Fullscreen request blocked:", err);
                    });
                } else {
                    if (document.exitFullscreen) document.exitFullscreen();
                }
            });

            document.addEventListener('fullscreenchange', () => {
                const isFS = !!document.fullscreenElement;
                this.fullscreenToggleBtn.classList.toggle('active', isFS);
                this.fullscreenToggleBtn.innerText = isFS ? '🗗 WINDOWED' : '⛶ FULLSCREEN';
            });
        }

        // Add Note Form
        const addNoteBtn = document.getElementById('saveNoteBtn');
        if (addNoteBtn) {
            addNoteBtn.addEventListener('click', () => this.handleSaveNote());
        }

        // New Chat Button
        const newChatBtn = document.getElementById('newChatBtn');
        if (newChatBtn) {
            newChatBtn.addEventListener('click', () => this.startNewConversation());
        }
    }

    _bindTabs() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.dataset.tab;
                this.switchTab(targetTab);
            });
        });
    }

    _bindConfigModal() {
        const modalOverlay = document.getElementById('configModalOverlay');
        const openBtn = document.getElementById('configModalOpenBtn');
        const closeBtn = document.getElementById('configModalCloseBtn');
        const saveBtn = document.getElementById('saveProviderConfigBtn');
        const provOpenRouterBtn = document.getElementById('provOpenRouterBtn');
        const provOllamaBtn = document.getElementById('provOllamaBtn');
        const openrouterGroup = document.getElementById('openrouterSettingsGroup');
        const ollamaGroup = document.getElementById('ollamaSettingsGroup');
        const modelSelect = document.getElementById('openrouterModelSelect');
        const customModelInput = document.getElementById('customModelInput');
        const apiKeyInput = document.getElementById('openrouterApiKeyInput');
        const toggleVisBtn = document.getElementById('toggleApiKeyVisBtn');
        const statusMsg = document.getElementById('configStatusMsg');

        let selectedModalProvider = 'openrouter';

        // Load TTS Settings
        const ttsEngine = localStorage.getItem('jarvis_tts_engine') || 'edge';
        const ttsVoice = localStorage.getItem('jarvis_tts_voice') || 'en_US-Male';
        this.ttsEngine = ttsEngine;
        this.ttsVoice = ttsVoice;

        if (openBtn && modalOverlay) {
            openBtn.addEventListener('click', async () => {
                modalOverlay.classList.add('active');
                
                // Sync UI with current settings
                const ttsEngineSelect = document.getElementById('ttsEngineSelect');
                const ttsVoiceSelect = document.getElementById('ttsVoiceSelect');
                if (ttsEngineSelect) ttsEngineSelect.value = this.ttsEngine;
                if (ttsVoiceSelect) ttsVoiceSelect.value = this.ttsVoice;

                if (window.jarvisAudio) window.jarvisAudio.playClick();
                await this.loadProviderConfig();
            });
        }

        if (closeBtn && modalOverlay) {
            closeBtn.addEventListener('click', () => {
                modalOverlay.classList.remove('active');
                if (window.jarvisAudio) window.jarvisAudio.playClick();
            });
        }

        // Switch between OpenRouter and Ollama in modal
        if (provOpenRouterBtn && provOllamaBtn) {
            provOpenRouterBtn.addEventListener('click', () => {
                selectedModalProvider = 'openrouter';
                provOpenRouterBtn.classList.add('active');
                provOllamaBtn.classList.remove('active');
                openrouterGroup.style.display = 'flex';
                ollamaGroup.style.display = 'none';
            });

            provOllamaBtn.addEventListener('click', () => {
                selectedModalProvider = 'ollama';
                provOllamaBtn.classList.add('active');
                provOpenRouterBtn.classList.remove('active');
                ollamaGroup.style.display = 'flex';
                openrouterGroup.style.display = 'none';
            });
        }

        // Custom Model ID visibility
        if (modelSelect && customModelInput) {
            modelSelect.addEventListener('change', () => {
                customModelInput.style.display = modelSelect.value === 'custom' ? 'block' : 'none';
            });
        }

        // Toggle API Key visibility
        if (toggleVisBtn && apiKeyInput) {
            toggleVisBtn.addEventListener('click', () => {
                apiKeyInput.type = apiKeyInput.type === 'password' ? 'text' : 'password';
            });
        }

        // Save Config
        if (saveBtn) {
            saveBtn.addEventListener('click', async () => {
                saveBtn.disabled = true;
                statusMsg.innerText = 'Applying neural configuration...';

                let targetModel = '';
                if (selectedModalProvider === 'openrouter') {
                    targetModel = modelSelect.value === 'custom' ? customModelInput.value.trim() : modelSelect.value;
                } else {
                    targetModel = document.getElementById('ollamaModelInput').value.trim() || 'qwen3.5:4b';
                }

                const payload = {
                    active_provider: selectedModalProvider,
                    openrouter_api_key: apiKeyInput.value.trim(),
                    openrouter_model: selectedModalProvider === 'openrouter' ? targetModel : undefined,
                    ollama_model: selectedModalProvider === 'ollama' ? targetModel : undefined
                };

                try {
                    // Save TTS settings locally
                    const ttsEngineSelect = document.getElementById('ttsEngineSelect');
                    const ttsVoiceSelect = document.getElementById('ttsVoiceSelect');
                    if (ttsEngineSelect) {
                        this.ttsEngine = ttsEngineSelect.value;
                        localStorage.setItem('jarvis_tts_engine', this.ttsEngine);
                    }
                    if (ttsVoiceSelect) {
                        this.ttsVoice = ttsVoiceSelect.value;
                        localStorage.setItem('jarvis_tts_voice', this.ttsVoice);
                    }

                    const res = await fetch('/api/config/provider', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();
                    if (data.success) {
                        this.activeProvider = selectedModalProvider;
                        this.activeModel = targetModel;
                        this.updateProviderHUD();
                        statusMsg.style.color = 'var(--stark-green)';
                        statusMsg.innerText = '✓ Configuration Saved & Synced!';
                        if (window.jarvisAudio) window.jarvisAudio.playBoot();
                        setTimeout(() => {
                            modalOverlay.classList.remove('active');
                            statusMsg.innerText = '';
                            saveBtn.disabled = false;
                        }, 1200);
                    }
                } catch (e) {
                    statusMsg.style.color = 'var(--stark-red)';
                    statusMsg.innerText = `Error: ${e.message}`;
                    saveBtn.disabled = false;
                }
            });
        }
    }

    async loadProviderConfig() {
        try {
            // 1. Fetch all available models from server (auto-fetched from OpenRouter)
            const modelsRes = await fetch('/api/models');
            if (modelsRes.ok) {
                const modelsData = await modelsRes.json();
                const allModels = modelsData.models || [];

                // Re-populate Header Dropdown
                if (this.headerModelSelect && allModels.length > 0) {
                    this.headerModelSelect.innerHTML = '';
                    allModels.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = `${m.provider}|${m.id}`;
                        const prefix = m.provider === 'openrouter' ? '🌐 ' : '⚡ ';
                        opt.innerText = `${prefix}${m.name}`;
                        this.headerModelSelect.appendChild(opt);
                    });
                }

                // Re-populate Modal Select
                const modalSelect = document.getElementById('openrouterModelSelect');
                if (modalSelect) {
                    const openrouterList = allModels.filter(m => m.provider === 'openrouter');
                    modalSelect.innerHTML = '';
                    openrouterList.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m.id;
                        opt.innerText = m.name;
                        modalSelect.appendChild(opt);
                    });
                    const customOpt = document.createElement('option');
                    customOpt.value = 'custom';
                    customOpt.innerText = '-- Custom Model ID (from OpenRouter) --';
                    modalSelect.appendChild(customOpt);
                }
            }

            // 2. Fetch active configuration
            const res = await fetch('/api/config/provider');
            if (res.ok) {
                const cfg = await res.json();
                this.activeProvider = cfg.active_provider || 'openrouter';
                this.activeModel = this.activeProvider === 'openrouter' 
                    ? (cfg.openrouter_model || 'google/gemma-4-26b-a4b-it:free') 
                    : (cfg.ollama_model || 'qwen3.5:4b');

                // Select current active model in Header Select
                if (this.headerModelSelect) {
                    const matchVal = `${this.activeProvider}|${this.activeModel}`;
                    const opt = Array.from(this.headerModelSelect.options).find(o => o.value === matchVal);
                    if (opt) {
                        this.headerModelSelect.value = matchVal;
                    } else if (this.activeProvider === 'openrouter') {
                        const newOpt = document.createElement('option');
                        newOpt.value = matchVal;
                        newOpt.innerText = `🌐 OpenRouter: ${this.activeModel}`;
                        this.headerModelSelect.appendChild(newOpt);
                        this.headerModelSelect.value = matchVal;
                    }
                }

                // Update Modal fields
                const apiKeyInput = document.getElementById('openrouterApiKeyInput');
                if (apiKeyInput && cfg.openrouter_api_key) {
                    apiKeyInput.value = cfg.openrouter_api_key;
                }

                const modalSelect = document.getElementById('openrouterModelSelect');
                if (modalSelect && this.activeProvider === 'openrouter') {
                    const exists = Array.from(modalSelect.options).some(o => o.value === this.activeModel);
                    if (exists) {
                        modalSelect.value = this.activeModel;
                    }
                }

                this.updateProviderHUD();
            }
        } catch (e) {}
    }

    updateProviderHUD() {
        const pill = document.getElementById('hudActiveProviderPill');
        const dot = document.getElementById('providerDot');
        const statusSpan = document.getElementById('centerLinkStatus');

        if (this.activeProvider === 'openrouter') {
            if (pill) pill.innerText = `LINK: OPENROUTER // ${this.activeModel.split('/')[1] || this.activeModel}`.toUpperCase();
            if (dot) dot.style.background = 'var(--stark-green)';
            if (statusSpan) statusSpan.innerText = `CLOUD LINK ACTIVE (${this.activeModel})`;
        } else {
            if (pill) pill.innerText = `LINK: OLLAMA // ${this.activeModel}`.toUpperCase();
            if (dot) dot.style.background = 'var(--stark-cyan)';
            if (statusSpan) statusSpan.innerText = `LOCAL LINK ACTIVE (${this.activeModel})`;
        }
    }

    switchTab(tabId) {
        this.activeTab = tabId;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === tabId));
        if (window.jarvisAudio) window.jarvisAudio.playClick();
    }

    logTerminal(text) {
        if (!this.terminalOutput) return;
        const time = new Date().toLocaleTimeString();
        const line = document.createElement('div');
        line.innerHTML = `<span style="color:var(--text-dim)">[${time}]</span> ${this.escapeHtml(text)}`;
        this.terminalOutput.appendChild(line);
        this.terminalOutput.scrollTop = this.terminalOutput.scrollHeight;
    }

    async executeTerminalCmd(cmd) {
        if (!cmd.trim()) return;
        this.logTerminal(`> ${cmd}`);
        try {
            const res = await fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: cmd })
            });
            const data = await res.json();
            if (data.stdout) this.logTerminal(data.stdout);
            if (data.stderr) this.logTerminal(`ERR: ${data.stderr}`);
            if (data.error) this.logTerminal(`DIRECTIVE REJECTED: ${data.error}`);
        } catch (e) {
            this.logTerminal(`EXECUTION FAILED: ${e.message}`);
        }
    }

    async handleUserSubmit() {
        const text = this.promptInput.value.trim();
        if (!text || this.isStreaming) return;

        this.promptInput.value = '';
        if (window.jarvisAudio) window.jarvisAudio.playClick();

        // Add user message to UI
        this.addUserMessage(text);
        this.conversationHistory.push({ role: 'user', content: text });

        // Trigger Arc Reactor thinking state
        if (window.arcReactor) window.arcReactor.setState('THINKING');
        this.isStreaming = true;
        this.sendBtn.disabled = true;

        // Prepare streaming message placeholder in UI
        const jarvisCard = this.createJarvisPlaceholder();
        const msgBody = jarvisCard.querySelector('.msg-body');

        let fullText = '';

        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            this.activeWs = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);

            this.activeWs.onopen = () => {
                this.activeWs.send(JSON.stringify({
                    messages: this.conversationHistory,
                    provider: this.activeProvider,
                    model: this.activeModel,
                    conversation_id: this.activeConversationId
                }));
            };

            this.activeWs.onmessage = (event) => {
                try {
                    const parsed = JSON.parse(event.data);
                    if (parsed.type === 'token') {
                        fullText += parsed.token;
                        msgBody.innerHTML = this.renderMarkdown(fullText);
                        this.chatStream.scrollTop = this.chatStream.scrollHeight;
                    } else if (parsed.type === 'done') {
                        this.handleGenerationDone(jarvisCard, parsed.full_text, parsed.actions);
                        this.activeWs.close();
                    } else if (parsed.type === 'error') {
                        msgBody.innerHTML = `<span style="color:var(--stark-red)">${parsed.error}</span>`;
                        this.activeWs.close();
                    }
                } catch (e) {}
            };

            this.activeWs.onerror = (e) => {
                msgBody.innerHTML = `<span style="color:var(--stark-red)">Neural Link Disrupted</span>`;
                this.isStreaming = false;
                this.sendBtn.disabled = false;
                if (window.arcReactor && !window.jarvisAudio.isSpeaking) {
                    window.arcReactor.setState('IDLE');
                }
            };

            this.activeWs.onclose = () => {
                this.isStreaming = false;
                this.sendBtn.disabled = false;
                if (window.arcReactor && !window.jarvisAudio.isSpeaking) {
                    window.arcReactor.setState('IDLE');
                }
                this.activeWs = null;
            };

        } catch (e) {
            msgBody.innerHTML = `<span style="color:var(--stark-red)">Neural Link Init Failed: ${e.message}</span>`;
            this.isStreaming = false;
            this.sendBtn.disabled = false;
            this.activeWs = null;
            if (window.arcReactor && !window.jarvisAudio.isSpeaking) {
                window.arcReactor.setState('IDLE');
            }
        }
    }

    handleGenerationDone(cardEl, fullResponse, actions = [], speakVoice = true) {
        this.conversationHistory.push({ role: 'assistant', content: fullResponse });

        // Add Copy Button
        const copyBtn = document.createElement('button');
        copyBtn.className = 'hud-btn-icon';
        copyBtn.style.position = 'absolute';
        copyBtn.style.top = '10px';
        copyBtn.style.right = '10px';
        copyBtn.innerText = '📋 COPY';
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(fullResponse);
            copyBtn.innerText = '✔️ COPIED';
            setTimeout(() => copyBtn.innerText = '📋 COPY', 2000);
        };
        cardEl.appendChild(copyBtn);

        // Clean action tags from displayed text if any remain
        const msgBody = cardEl.querySelector('.msg-body');
        const cleanDisplayText = fullResponse.replace(/\[\[ACTION:[^\]]+\]\]/g, '').trim();
        msgBody.innerHTML = this.renderMarkdown(cleanDisplayText || fullResponse);

        // Detect Avatar Trigger
        if (fullResponse.includes('[[ACTION:show_avatar]]')) {
            if (window.showHolographicAvatar) window.showHolographicAvatar();
        }
        if (fullResponse.includes('[[ACTION:hide_avatar]]')) {
            if (window.hideHolographicAvatar) window.hideHolographicAvatar();
        }

        // Render Action badges & Holographic Report Cards
        if (actions && actions.length > 0) {
            const badgeContainer = document.createElement('div');
            badgeContainer.className = 'action-badge-container';

            const badgeRow = document.createElement('div');
            badgeRow.className = 'action-badge-row';
            badgeContainer.appendChild(badgeRow);

            let actionResultsStr = "System Note: The following actions were executed. Use this information to answer the user.\n";
            let shouldTriggerLoop = false;

            actions.forEach(act => {
                const badge = document.createElement('div');
                badge.className = 'action-badge';
                let icon = '⚡';
                let label = act.action;
                
                // Add to feedback
                shouldTriggerLoop = true;
                actionResultsStr += `Action: ${act.action} | Result: ${JSON.stringify(act.result || act.error || 'Done')}\n`;

                if (act.action === 'open_app') {
                    icon = '🚀';
                    label = `LAUNCHED: ${act.app}`;
                } else if (act.action === 'screenshot') {
                    icon = '📸';
                    label = `SCREENSHOT CAPTURED`;
                    if (act.result?.base64_data && window.jarvisOCR) {
                        window.jarvisOCR.currentImageBase64 = act.result.base64_data;
                        if (window.jarvisOCR.previewImg) {
                            window.jarvisOCR.previewImg.src = act.result.base64_data;
                            window.jarvisOCR.previewImg.style.display = 'block';
                        }
                    }

                    // Render Chat Screenshot Preview Card
                    if (act.result?.base64_data) {
                        const ssCard = document.createElement('div');
                        ssCard.className = 'chat-screenshot-card';
                        ssCard.innerHTML = `
                            <div class="report-header">
                                <span>📸 DESKTOP OPTICAL CAPTURE</span>
                                <span style="font-size:0.7rem; color:var(--text-secondary);">${new Date().toLocaleTimeString()}</span>
                            </div>
                            <img src="${act.result.base64_data}" class="chat-screenshot-img" title="Click to inspect" />
                        `;
                        badgeContainer.appendChild(ssCard);
                    }
                } else if (act.action === 'system_vitals' || (act.action === 'protocol' && (act.id === 'diagnostics' || act.id === 'threat_scan'))) {
                    icon = '📊';
                    label = act.action === 'system_vitals' ? 'VITALS CHECKED' : `PROTOCOL: ${act.id.toUpperCase()}`;

                    // Extract vitals data
                    const vitals = act.result?.vitals || act.result;
                    if (vitals && (vitals.cpu || vitals.memory)) {
                        const cpuVal = vitals.cpu?.percent || 0;
                        const memVal = vitals.memory?.percent || 0;
                        const diskVal = vitals.disk?.percent || 0;
                        const powerVal = vitals.battery?.percent ?? 100;
                        const hostName = vitals.system?.hostname || 'STARK-CORE';
                        const osName = vitals.system?.platform || 'Windows 11';
                        const procs = (vitals.top_processes || []).slice(0, 4);

                        const reportCard = document.createElement('div');
                        reportCard.className = 'holographic-report-card';
                        reportCard.innerHTML = `
                            <div class="report-header">
                                <span>⚡ STARK MARK-85 TELEMETRY REPORT</span>
                                <span style="font-size:0.7rem; color:var(--stark-green); font-weight:700;">STATUS: NOMINAL</span>
                            </div>
                            <div class="report-metrics-grid">
                                <div class="report-metric-box">
                                    <span class="report-metric-title">CPU LOAD</span>
                                    <span class="report-metric-val">${cpuVal}%</span>
                                    <div class="report-bar-bg"><div class="report-bar-fill" style="width: ${cpuVal}%"></div></div>
                                </div>
                                <div class="report-metric-box">
                                    <span class="report-metric-title">MEMORY</span>
                                    <span class="report-metric-val">${memVal}%</span>
                                    <div class="report-bar-bg"><div class="report-bar-fill" style="width: ${memVal}%"></div></div>
                                </div>
                                <div class="report-metric-box">
                                    <span class="report-metric-title">STORAGE</span>
                                    <span class="report-metric-val">${diskVal}%</span>
                                    <div class="report-bar-bg"><div class="report-bar-fill" style="width: ${diskVal}%"></div></div>
                                </div>
                                <div class="report-metric-box">
                                    <span class="report-metric-title">POWER</span>
                                    <span class="report-metric-val" style="color:var(--stark-green);">${powerVal}%</span>
                                    <div class="report-bar-bg"><div class="report-bar-fill" style="width: ${powerVal}%; background:var(--stark-green);"></div></div>
                                </div>
                            </div>
                            <div class="report-procs-section">
                                <span>ACTIVE HIGH-DEMAND SUB-PROCESSES:</span>
                                <div class="report-proc-chips">
                                    ${procs.map(p => `<span class="report-proc-chip">${p.name} (${p.cpu}% CPU)</span>`).join('') || '<span class="report-proc-chip">None</span>'}
                                </div>
                            </div>
                        `;
                        badgeContainer.appendChild(reportCard);
                    }
                } else if (act.action === 'protocol') {
                    icon = '🛡️';
                    label = `PROTOCOL: ${act.id.toUpperCase()}`;
                } else if (act.action === 'save_note') {
                    icon = '💾';
                    label = `NOTE MEMORIZED`;
                    this.loadVaultNotes();
                }

                badge.innerHTML = `<span>${icon} ${label}</span>`;
                badgeRow.appendChild(badge);
            });

            cardEl.appendChild(badgeContainer);
            this.chatStream.scrollTop = this.chatStream.scrollHeight;
            
            // Trigger multi-step agentic loop if actions occurred and we aren't already looping too deep
            if (shouldTriggerLoop) {
                this.triggerAgenticLoop(actionResultsStr);
            }
        }

        // Voice speak synthesis
        if (speakVoice && window.jarvisAudio && window.jarvisAudio.voiceEnabled) {
            window.jarvisAudio.speak(cleanDisplayText);
        }

        this.chatStream.scrollTop = this.chatStream.scrollHeight;
    }

    async triggerAgenticLoop(systemFeedback) {
        if (this.isStreaming) return;
        
        // Add system feedback as a developer/system role to history
        this.conversationHistory.push({ role: 'system', content: systemFeedback });
        
        if (window.arcReactor) window.arcReactor.setState('THINKING');
        this.isStreaming = true;
        this.sendBtn.disabled = true;

        const jarvisCard = this.createJarvisPlaceholder();
        const msgBody = jarvisCard.querySelector('.msg-body');
        msgBody.innerHTML = '<span style="color:var(--text-secondary); font-style:italic;">Processing action results...</span>';
        let fullText = '';

        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            this.activeWs = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);

            this.activeWs.onopen = () => {
                this.activeWs.send(JSON.stringify({
                    messages: this.conversationHistory,
                    provider: this.activeProvider,
                    model: this.activeModel,
                    conversation_id: this.activeConversationId
                }));
            };

            this.activeWs.onmessage = (event) => {
                try {
                    const parsed = JSON.parse(event.data);
                    if (parsed.type === 'token') {
                        fullText += parsed.token;
                        msgBody.innerHTML = this.renderMarkdown(fullText);
                        this.chatStream.scrollTop = this.chatStream.scrollHeight;
                    } else if (parsed.type === 'done') {
                        this.handleGenerationDone(jarvisCard, parsed.full_text, parsed.actions);
                        this.activeWs.close();
                    } else if (parsed.type === 'error') {
                        msgBody.innerHTML = `<span style="color:var(--stark-red)">${parsed.error}</span>`;
                        this.activeWs.close();
                    }
                } catch (e) {}
            };

            this.activeWs.onerror = (e) => {
                this.isStreaming = false;
                this.sendBtn.disabled = false;
                if (window.arcReactor && !window.jarvisAudio.isSpeaking) window.arcReactor.setState('IDLE');
            };
            this.activeWs.onclose = () => {
                this.isStreaming = false;
                this.sendBtn.disabled = false;
                if (window.arcReactor && !window.jarvisAudio.isSpeaking) window.arcReactor.setState('IDLE');
                this.activeWs = null;
            };
        } catch (e) {
            this.isStreaming = false;
            this.sendBtn.disabled = false;
            this.activeWs = null;
        }
    }

    addUserMessage(text) {
        const card = document.createElement('div');
        card.className = 'message-card user-message';
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        const isRTL = /[\u0600-\u06FF]/.test(text);

        card.innerHTML = `
            <div class="msg-sender-row">
                <span style="color:var(--stark-gold); font-weight:700;">COMMANDER</span>
                <span class="msg-time">${time}</span>
            </div>
            <div class="msg-body" ${isRTL ? 'dir="rtl"' : ''}>${this.escapeHtml(text)}</div>
        `;
        this.chatStream.appendChild(card);
        this.chatStream.scrollTop = this.chatStream.scrollHeight;
    }

    createJarvisPlaceholder() {
        const card = document.createElement('div');
        card.className = 'message-card jarvis-message';
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        card.innerHTML = `
            <div class="msg-sender-row">
                <span style="font-weight:700;">J.A.R.V.I.S.</span>
                <span class="msg-time">${time}</span>
            </div>
            <div class="msg-body">
                <span style="color:var(--stark-cyan); font-family:var(--font-telemetry);">Analyzing neural patterns...</span>
            </div>
        `;
        this.chatStream.appendChild(card);
        this.chatStream.scrollTop = this.chatStream.scrollHeight;
        return card;
    }

    addJarvisMessage(text, actions = [], speakVoice = true) {
        const card = this.createJarvisPlaceholder();
        this.handleGenerationDone(card, text, actions, speakVoice);
    }

    renderMarkdown(text) {
        if (!text) return '';
        const isRTL = /[\u0600-\u06FF]/.test(text);
        
        let html = text
            // Code blocks
            .replace(/```([a-zA-Z]*)\n([\s\S]*?)```/g, (match, lang, code) => {
                const cleanCode = this.escapeHtml(code.trim());
                return `
                    <pre><button class="code-copy-btn" onclick="navigator.clipboard.writeText(decodeURIComponent('${encodeURIComponent(code.trim())}')); this.innerText='✓ COPIED'; setTimeout(()=>this.innerText='COPY', 2000);">COPY</button><code>${cleanCode}</code></pre>
                `;
            })
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Bold
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            // Blockquotes
            .replace(/^>\s*(.+)$/gm, '<blockquote style="border-left:3px solid var(--stark-cyan); padding-left:8px; color:var(--text-secondary); margin:4px 0;">$1</blockquote>')
            // Line breaks
            .replace(/\n/g, '<br/>');

        return `<div ${isRTL ? 'dir="rtl"' : ''}>${html}</div>`;
    }

    escapeHtml(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Neural Memory Vault methods
    async loadVaultNotes() {
        try {
            const res = await fetch('/api/vault');
            if (res.ok) {
                const data = await res.json();
                this.renderNotes(data.notes || []);
            }
        } catch (e) {}
    }

    renderNotes(notes) {
        const container = document.getElementById('vaultNotesList');
        if (!container) return;

        if (notes.length === 0) {
            container.innerHTML = '<div style="color:var(--text-dim); font-size:0.8rem; font-family:var(--font-telemetry);">// Memory Vault empty. No active directives recorded.</div>';
            return;
        }

        container.innerHTML = notes.map(n => `
            <div style="background:rgba(0,240,255,0.03); border:1px solid rgba(0,240,255,0.15); border-radius:6px; padding:8px; display:flex; flex-direction:column; gap:4px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:var(--font-sci-fi); font-size:0.8rem; color:var(--stark-gold); font-weight:700;">${this.escapeHtml(n.title)}</span>
                    <button style="background:none; border:none; color:var(--stark-red); cursor:pointer; font-size:0.85rem;" onclick="window.jarvisApp.deleteNote(${n.id})">✕</button>
                </div>
                <div style="font-size:0.8rem; color:var(--text-secondary);">${this.escapeHtml(n.content)}</div>
                <div style="font-family:var(--font-telemetry); font-size:0.65rem; color:var(--text-dim);">${n.created_at || ''}</div>
            </div>
        `).join('');
    }

    async handleSaveNote() {
        const titleEl = document.getElementById('noteTitleInput');
        const contentEl = document.getElementById('noteContentInput');
        if (!titleEl || !contentEl || !titleEl.value.trim()) return;

        try {
            await fetch('/api/vault/note', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: titleEl.value.trim(), content: contentEl.value.trim() })
            });
            titleEl.value = '';
            contentEl.value = '';
            this.loadVaultNotes();
            if (window.jarvisAudio) window.jarvisAudio.playClick();
        } catch (e) {}
    }

    async deleteNote(id) {
        try {
            await fetch(`/api/vault/note/${id}`, { method: 'DELETE' });
            this.loadVaultNotes();
            if (window.jarvisAudio) window.jarvisAudio.playClick();
        } catch (e) {}
    }
}

window.switchHUDTab = (tabId) => {
    if (window.jarvisApp) window.jarvisApp.switchTab(tabId);
};

window.logToTerminal = (text) => {
    if (window.jarvisApp) window.jarvisApp.logTerminal(text);
};

window.addJarvisMessage = (text, actions) => {
    if (window.jarvisApp) window.jarvisApp.addJarvisMessage(text, actions);
};

document.addEventListener('DOMContentLoaded', () => {
    window.jarvisApp = new JarvisApp();
});
