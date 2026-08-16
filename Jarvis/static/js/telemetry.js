/**
 * J.A.R.V.I.S. Real-time Telemetry & Hardware Vitals Animator
 */

class JarvisTelemetry {
    constructor() {
        this.pollInterval = 2500;
        this.timer = null;
        this.init();
    }

    init() {
        this.connectWebSocket();
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/telemetry`);

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.renderVitals(data);
            } catch (e) {
                console.warn("Telemetry parsing error:", e);
            }
        };

        this.ws.onclose = () => {
            console.log("Telemetry WS disconnected. Reconnecting in 3s...");
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (e) => {
            console.warn("Telemetry WS error:", e);
            this.ws.close();
        };
    }

    _updateGauge(id, percent) {
        const circle = document.getElementById(id + 'Progress');
        const text = document.getElementById(id + 'Val');
        if (!circle || !text) return;

        const maxOffset = 188.5; // 2 * PI * 30
        const val = Math.min(Math.max(percent, 0), 100);
        const offset = maxOffset - (val / 100) * maxOffset;

        circle.style.strokeDashoffset = offset;
        text.innerText = Math.round(val) + '%';

        // Color coding
        circle.classList.remove('warn', 'alert');
        if (val > 85) {
            circle.classList.add('alert');
        } else if (val > 65) {
            circle.classList.add('warn');
        }
    }

    renderVitals(data) {
        if (!data || data.status === 'ERROR') return;

        // CPU
        const cpuP = data.cpu?.percent || 0;
        this._updateGauge('cpu', cpuP);

        // RAM
        const memP = data.memory?.percent || 0;
        this._updateGauge('ram', memP);

        // Disk
        const diskP = data.disk?.percent || 0;
        this._updateGauge('disk', diskP);

        // Battery
        const batP = data.battery?.percent || 100;
        this._updateGauge('bat', batP);

        // Specs & Details
        const hostEl = document.getElementById('specHost');
        if (hostEl && data.hostname) hostEl.innerText = data.hostname;

        const osEl = document.getElementById('specOS');
        if (osEl && data.os) osEl.innerText = data.os;

        const uptimeEl = document.getElementById('specUptime');
        if (uptimeEl && data.uptime_seconds) {
            const hrs = Math.floor(data.uptime_seconds / 3600);
            const mins = Math.floor((data.uptime_seconds % 3600) / 60);
            uptimeEl.innerText = `${hrs}h ${mins}m`;
        }

        const netEl = document.getElementById('specNet');
        if (netEl && data.network) {
            netEl.innerText = `↑${data.network.bytes_sent_mb}M ↓${data.network.bytes_recv_mb}M`;
        }

        // Top processes in HUD tooltip or mini-list if present
        const procListEl = document.getElementById('topProcessesList');
        if (procListEl && data.top_processes) {
            procListEl.innerHTML = data.top_processes.map(p => `
                <div class="spec-row" style="font-size:0.7rem; padding: 1px 0;">
                    <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:110px;">${p.name}</span>
                    <span class="spec-val">${p.cpu}% CPU / ${p.mem}% RAM</span>
                </div>
            `).join('');
        }
    }
}

window.jarvisTelemetry = new JarvisTelemetry();
