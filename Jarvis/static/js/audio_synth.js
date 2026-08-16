/**
 * J.A.R.V.I.S. Sci-Fi Audio Synthesizer & Speech Engine
 * Powered by Web Audio API & Web Speech API
 */

class JarvisAudioEngine {
    constructor() {
        this.ctx = null;
        this.sfxEnabled = true;
        this.voiceEnabled = true;
        this.currentUtterance = null;
        this.isSpeaking = false;
        this.onSpeakingStateChange = null;
        this.voices = [];
        this.hasGreeted = false;
        
        // Backend TTS State
        this.audioQueue = [];
        this.isPlayingQueue = false;
        this.currentAudioElement = null;

        this._initVoices();
    }

    _initVoices() {
        if ('speechSynthesis' in window) {
            this.voices = window.speechSynthesis.getVoices() || [];
            window.speechSynthesis.onvoiceschanged = () => {
                this.voices = window.speechSynthesis.getVoices() || [];
            };
        }
    }

    _initContext() {
        if (!this.ctx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (AudioContext) {
                this.ctx = new AudioContext();
            }
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    // Play high-tech holographic UI click
    playClick() {
        if (!this.sfxEnabled) return;
        this._initContext();
        if (!this.ctx) return;

        try {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(1400, this.ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(800, this.ctx.currentTime + 0.05);

            gain.gain.setValueAtTime(0.12, this.ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.05);

            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start();
            osc.stop(this.ctx.currentTime + 0.05);
        } catch (e) {
            console.warn("Audio SFX error:", e);
        }
    }

    // Play Arc Reactor power boot sound
    playBoot() {
        if (!this.sfxEnabled) return;
        this._initContext();
        if (!this.ctx) return;

        try {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(100, this.ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(600, this.ctx.currentTime + 0.6);

            gain.gain.setValueAtTime(0.01, this.ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0.2, this.ctx.currentTime + 0.3);
            gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.8);

            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start();
            osc.stop(this.ctx.currentTime + 0.8);
        } catch (e) {}
    }

    // Play Tactical Alert Tone
    playAlert() {
        if (!this.sfxEnabled) return;
        this._initContext();
        if (!this.ctx) return;

        try {
            const now = this.ctx.currentTime;
            [0, 0.15].forEach(delay => {
                const osc = this.ctx.createOscillator();
                const gain = this.ctx.createGain();
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(880, now + delay);
                gain.gain.setValueAtTime(0.18, now + delay);
                gain.gain.exponentialRampToValueAtTime(0.001, now + delay + 0.1);
                osc.connect(gain);
                gain.connect(this.ctx.destination);
                osc.start(now + delay);
                osc.stop(now + delay + 0.1);
            });
        } catch (e) {}
    }

    _getBestEnglishVoice() {
        const voices = this.voices.length > 0 ? this.voices : (window.speechSynthesis ? window.speechSynthesis.getVoices() : []);
        if (!voices || voices.length === 0) return null;

        // Comprehensive list of female keywords to strictly exclude
        const femalePatterns = [
            'female', 'woman', 'girl', 'zira', 'susan', 'hazel', 'jenny', 'aria', 'sonia', 
            'libby', 'maisie', 'samantha', 'victoria', 'karen', 'linda', 'heera', 'catherine', 
            'ana', 'mia', 'ava', 'emma', 'olivia', 'sophia', 'emily', 'charlotte', 'steffi', 
            'helena', 'elena', 'hedda', 'sabina', 'eva', 'katja', 'monica', 'alice', 'clara',
            'laura', 'julie', 'marie', 'amalie', 'lucia', 'lauren', 'amber', 'cora', 'zoe',
            'serena', 'fiona', 'tessa', 'salli', 'joanna', 'kendra', 'kimberly', 'ivy'
        ];

        // Specific Male Voice names in Windows, Edge, Chrome, Google TTS, Mac, Linux
        const malePatterns = [
            'david', 'george', 'ryan', 'guy', 'mark', 'christopher', 'eric', 
            'brian', 'roger', 'stephen', 'richard', 'paul', 'james', 'thomas', 
            'charles', 'william', 'john', 'robert', 'michael', 'arthur', 'male',
            'english male', 'uk english male', 'us english male', 'great britain male'
        ];

        const isKnownFemale = (name) => {
            const low = (name || '').toLowerCase();
            return femalePatterns.some(p => low.includes(p));
        };

        const isKnownMale = (name) => {
            const low = (name || '').toLowerCase();
            return malePatterns.some(p => low.includes(p)) && !isKnownFemale(name);
        };

        const englishVoices = voices.filter(v => 
            (v.lang.startsWith('en') || v.lang.includes('en-') || v.lang.includes('en_')) && !isKnownFemale(v.name)
        );

        // 1. Definite British Male (George, Ryan, Oliver, UK English Male)
        const ukMale = englishVoices.find(v => {
            const l = v.lang.toLowerCase();
            return (l.includes('gb') || l.includes('uk') || v.name.toLowerCase().includes('united kingdom')) && isKnownMale(v.name);
        });
        if (ukMale) return ukMale;

        // 2. Definite US/English Male (David, Guy, Christopher, Mark, US English Male)
        const usMale = englishVoices.find(v => isKnownMale(v.name));
        if (usMale) return usMale;

        // 3. Any English voice that is verified not female
        if (englishVoices.length > 0) return englishVoices[0];

        // 4. Any system voice that is verified not female
        const nonFemale = voices.filter(v => !isKnownFemale(v.name));
        if (nonFemale.length > 0) return nonFemale[0];

        return voices[0] || null;
    }

    // Internal: Speak using Browser API
    _speakBrowser(cleanText, onStarted) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(cleanText);
        const voice = this._getBestEnglishVoice();
        
        utterance.lang = voice ? voice.lang : 'en-GB';
        if (voice) utterance.voice = voice;
        
        utterance.rate = 0.93;
        utterance.pitch = 0.78;

        utterance.onstart = () => {
            this.isSpeaking = true;
            if (onStarted) onStarted();
            if (this.onSpeakingStateChange) this.onSpeakingStateChange(true);
        };

        utterance.onend = () => {
            this.isSpeaking = false;
            if (this.onSpeakingStateChange) this.onSpeakingStateChange(false);
        };

        utterance.onerror = () => {
            this.isSpeaking = false;
            if (this.onSpeakingStateChange) this.onSpeakingStateChange(false);
        };

        window.speechSynthesis.speak(utterance);
    }

    // Synthesize Voice output
    async speak(text, onStarted = null) {
        if (!this.voiceEnabled) return;
        
        const cleanText = text
            .replace(/\[\[ACTION:[^\]]+\]\]/g, '')
            .replace(/```[\s\S]*?```/g, 'Code block generated.')
            .replace(/[*#`_~]/g, '')
            .trim();

        if (!cleanText) return;

        // Determine TTS Engine
        let engine = 'edge';
        let voice = 'en_US-Male';
        if (window.jarvisApp) {
            engine = window.jarvisApp.ttsEngine || 'edge';
            voice = window.jarvisApp.ttsVoice || 'en_US-Male';
        }

        if (engine === 'browser') {
            this._speakBrowser(cleanText, onStarted);
            return;
        }

        // Backend TTS Flow (Edge/Piper)
        try {
            const res = await fetch('/api/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: cleanText, engine: engine, voice: voice })
            });

            if (res.ok) {
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                this.audioQueue.push({ url, onStarted });
                this._playNextInQueue();
            } else {
                console.warn("Backend TTS failed, falling back to Browser TTS.");
                this._speakBrowser(cleanText, onStarted);
            }
        } catch (e) {
            console.error("TTS Error:", e);
            this._speakBrowser(cleanText, onStarted);
        }
    }

    _playNextInQueue() {
        if (this.isPlayingQueue || this.audioQueue.length === 0) return;
        
        this.isPlayingQueue = true;
        const currentItem = this.audioQueue.shift();
        
        this.currentAudioElement = new Audio(currentItem.url);
        this.currentAudioElement.onplay = () => {
            this.isSpeaking = true;
            if (currentItem.onStarted) currentItem.onStarted();
            if (this.onSpeakingStateChange) this.onSpeakingStateChange(true);
        };
        
        this.currentAudioElement.onended = () => {
            URL.revokeObjectURL(currentItem.url);
            this.currentAudioElement = null;
            this.isPlayingQueue = false;
            if (this.audioQueue.length > 0) {
                this._playNextInQueue();
            } else {
                this.isSpeaking = false;
                if (this.onSpeakingStateChange) this.onSpeakingStateChange(false);
            }
        };

        this.currentAudioElement.onerror = () => {
            URL.revokeObjectURL(currentItem.url);
            this.currentAudioElement = null;
            this.isPlayingQueue = false;
            this.isSpeaking = false;
            if (this.onSpeakingStateChange) this.onSpeakingStateChange(false);
            this._playNextInQueue();
        };

        this.currentAudioElement.play().catch(e => {
            console.error("Audio Playback Blocked:", e);
            this.isPlayingQueue = false;
            this._playNextInQueue();
        });
    }

    speakGreeting(greetingText = "Good day, Sir. All Mark-85 systems are online and fully operational. Standing by for your directives.") {
        if (this.hasSuccessfullySpokenGreeting) return;

        const removeListeners = () => {
            window.removeEventListener('pointerdown', unlockAudio);
            window.removeEventListener('keydown', unlockAudio);
            window.removeEventListener('click', unlockAudio);
        };

        const unlockAudio = () => {
            this._initContext();
            if (!this.hasSuccessfullySpokenGreeting) {
                this.speak(greetingText, () => {
                    this.hasSuccessfullySpokenGreeting = true;
                    removeListeners();
                });
            }
            removeListeners();
        };

        // Try speaking immediately on load
        this.speak(greetingText, () => {
            this.hasSuccessfullySpokenGreeting = true;
            removeListeners();
        });

        // Register one-time listener strictly as fallback if autoplay was restricted
        window.addEventListener('pointerdown', unlockAudio, { once: true });
        window.addEventListener('keydown', unlockAudio, { once: true });
        window.addEventListener('click', unlockAudio, { once: true });
    }

    stopSpeaking() {
        // Stop browser TTS
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        
        // Stop backend audio playback
        if (this.currentAudioElement) {
            this.currentAudioElement.pause();
            this.currentAudioElement.removeAttribute('src');
            this.currentAudioElement.load();
            this.currentAudioElement = null;
        }
        
        // Clear queue
        this.audioQueue.forEach(item => URL.revokeObjectURL(item.url));
        this.audioQueue = [];
        this.isPlayingQueue = false;
        
        this.isSpeaking = false;
        if (this.onSpeakingStateChange) this.onSpeakingStateChange(false);
    }
}

window.jarvisAudio = new JarvisAudioEngine();
