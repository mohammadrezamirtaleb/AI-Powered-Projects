/**
 * J.A.R.V.I.S. Interactive Holographic Arc Reactor Canvas Engine
 * Stark Industries Mark-85 Core Visualization
 */

class ArcReactorHUD {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d', { alpha: true });
        
        this.state = 'IDLE'; // IDLE, THINKING, SPEAKING, OVERCHARGE, THREAT
        this.angle1 = 0;
        this.angle2 = 0;
        this.angle3 = 0;
        this.pulse = 0;
        this.particles = [];
        this.audioReactLevel = 0;

        this._setupCanvas();
        this._initParticles();
        this._bindEvents();
        this.animate = this.animate.bind(this);
        // Immediate first frame paint
        this.animate();
    }

    _setupCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        const attrW = parseInt(this.canvas.getAttribute('width')) || 0;
        const attrH = parseInt(this.canvas.getAttribute('height')) || 0;
        this.width = rect.width || attrW || (this.canvas.id === 'heroReactorCanvas' ? 320 : 220);
        this.height = rect.height || attrH || (this.canvas.id === 'heroReactorCanvas' ? 320 : 220);
        this.canvas.width = this.width * dpr;
        this.canvas.height = this.height * dpr;
        this.ctx.scale(dpr, dpr);
    }

    _initParticles() {
        this.particles = [];
        for (let i = 0; i < 35; i++) {
            this.particles.push({
                x: this.width / 2,
                y: this.height / 2,
                angle: Math.random() * Math.PI * 2,
                speed: 0.4 + Math.random() * 1.2,
                radius: 1 + Math.random() * 2,
                life: Math.random() * 100,
                maxLife: 60 + Math.random() * 60
            });
        }
    }

    _bindEvents() {
        window.addEventListener('resize', () => this._setupCanvas());
        this.canvas.addEventListener('click', () => {
            if (window.jarvisAudio) window.jarvisAudio.playClick();
            this.triggerOvercharge();
        });
    }

    setState(newState) {
        this.state = newState;
        const container = this.canvas.parentElement;
        if (container) {
            container.className = 'reactor-canvas-container';
            if (newState === 'THINKING') container.classList.add('active-thinking');
            if (newState === 'THREAT') container.classList.add('active-threat');
        }
    }

    triggerOvercharge() {
        const prevState = this.state;
        this.setState('OVERCHARGE');
        if (window.jarvisAudio) window.jarvisAudio.playBoot();
        setTimeout(() => {
            this.setState(prevState === 'OVERCHARGE' ? 'IDLE' : prevState);
        }, 3000);
    }

    animate() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        const cx = this.width / 2;
        const cy = this.height / 2;
        const radius = Math.min(cx, cy) - 10;

        // Speed multipliers based on state
        let speedMult = 1;
        let mainColor = '#00f0ff';
        let glowColor = 'rgba(0, 240, 255, ';

        if (this.state === 'THINKING') {
            speedMult = 3.5;
            mainColor = '#ffb700';
            glowColor = 'rgba(255, 183, 0, ';
        } else if (this.state === 'SPEAKING') {
            speedMult = 1.8;
            this.audioReactLevel = 0.5 + Math.sin(Date.now() * 0.015) * 0.45;
        } else if (this.state === 'THREAT') {
            speedMult = 4;
            mainColor = '#ff2a55';
            glowColor = 'rgba(255, 42, 85, ';
        } else if (this.state === 'OVERCHARGE') {
            speedMult = 5;
            mainColor = '#ffffff';
            glowColor = 'rgba(255, 255, 255, ';
        } else {
            this.audioReactLevel = 0;
        }

        this.angle1 += 0.008 * speedMult;
        this.angle2 -= 0.014 * speedMult;
        this.angle3 += 0.02 * speedMult;
        this.pulse = Math.sin(Date.now() * 0.003) * 0.15 + 0.85;

        // 1. Draw outer telemetry track
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        this.ctx.strokeStyle = glowColor + '0.2)';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // 2. Draw rotating tick marks ring
        this.ctx.save();
        this.ctx.translate(cx, cy);
        this.ctx.rotate(this.angle1);
        const ticks = 40;
        for (let i = 0; i < ticks; i++) {
            const a = (i / ticks) * Math.PI * 2;
            const isLong = i % 4 === 0;
            const r1 = radius - (isLong ? 8 : 4);
            const r2 = radius - 1;
            this.ctx.beginPath();
            this.ctx.moveTo(Math.cos(a) * r1, Math.sin(a) * r1);
            this.ctx.lineTo(Math.cos(a) * r2, Math.sin(a) * r2);
            this.ctx.strokeStyle = isLong ? mainColor : glowColor + '0.4)';
            this.ctx.lineWidth = isLong ? 2 : 1;
            this.ctx.stroke();
        }
        this.ctx.restore();

        // 3. Draw middle segmented arc reactor coils (10 Stark Energy Coils)
        const coilRadius = radius * 0.72;
        const coilCount = 10;
        this.ctx.save();
        this.ctx.translate(cx, cy);
        this.ctx.rotate(this.angle2);

        for (let i = 0; i < coilCount; i++) {
            const a = (i / coilCount) * Math.PI * 2;
            const coilX = Math.cos(a) * coilRadius;
            const coilY = Math.sin(a) * coilRadius;

            // Draw coil block
            this.ctx.save();
            this.ctx.translate(coilX, coilY);
            this.ctx.rotate(a + Math.PI / 2);
            
            this.ctx.fillStyle = glowColor + (0.5 * this.pulse) + ')';
            this.ctx.shadowColor = mainColor;
            this.ctx.shadowBlur = 8;
            this.ctx.fillRect(-6, -3, 12, 6);

            this.ctx.strokeStyle = mainColor;
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(-7, -4, 14, 8);
            this.ctx.restore();
        }
        this.ctx.restore();

        // 4. Draw Inner Kinetic Ring & Tri-Segments
        const innerRadius = radius * 0.48;
        this.ctx.save();
        this.ctx.translate(cx, cy);
        this.ctx.rotate(this.angle3);

        this.ctx.beginPath();
        this.ctx.arc(0, 0, innerRadius, 0, Math.PI * 2);
        this.ctx.strokeStyle = mainColor;
        this.ctx.lineWidth = 2.5;
        this.ctx.shadowColor = mainColor;
        this.ctx.shadowBlur = 10;
        this.ctx.stroke();

        // Triangular core guides
        for (let i = 0; i < 3; i++) {
            const a = (i / 3) * Math.PI * 2;
            this.ctx.beginPath();
            this.ctx.arc(0, 0, innerRadius - 4, a, a + 0.8);
            this.ctx.strokeStyle = '#fff';
            this.ctx.lineWidth = 3;
            this.ctx.stroke();
        }
        this.ctx.restore();

        // 5. Center Glowing Plasma Core
        const coreRadius = (radius * 0.28) * (this.state === 'SPEAKING' ? (1 + this.audioReactLevel * 0.25) : this.pulse);
        const grad = this.ctx.createRadialGradient(cx, cy, 0, cx, cy, coreRadius);
        grad.addColorStop(0, '#ffffff');
        grad.addColorStop(0.4, glowColor + '0.9)');
        grad.addColorStop(0.8, glowColor + '0.4)');
        grad.addColorStop(1, 'transparent');

        this.ctx.beginPath();
        this.ctx.arc(cx, cy, coreRadius, 0, Math.PI * 2);
        this.ctx.fillStyle = grad;
        this.ctx.shadowColor = mainColor;
        this.ctx.shadowBlur = 20;
        this.ctx.fill();

        // 6. Particle Energy Streams
        this.particles.forEach(p => {
            p.life++;
            if (p.life > p.maxLife) {
                p.life = 0;
                p.angle = Math.random() * Math.PI * 2;
                p.speed = 0.4 + Math.random() * 1.2 * speedMult;
            }
            const curR = (p.life / p.maxLife) * radius;
            const px = cx + Math.cos(p.angle) * curR;
            const py = cy + Math.sin(p.angle) * curR;
            const alpha = (1 - (p.life / p.maxLife)) * 0.8;

            this.ctx.beginPath();
            this.ctx.arc(px, py, p.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = glowColor + alpha + ')';
            this.ctx.shadowBlur = 4;
            this.ctx.fill();
        });

        requestAnimationFrame(this.animate);
    }
}

window.initArcReactor = (id) => new ArcReactorHUD(id);
