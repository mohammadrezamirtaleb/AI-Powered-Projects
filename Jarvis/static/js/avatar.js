/**
 * J.A.R.V.I.S. Holographic Avatar Engine
 * Renders a 3D particle face (from OBJ) that forms from powder and follows the mouse.
 */

class AvatarEngine {
    constructor() {
        this.container = document.getElementById('hologramContainer');
        if (!this.container) return;

        this.scene = new THREE.Scene();
        // Add ambient glow behind the avatar
        const pointLight = new THREE.PointLight(0xffaa00, 2, 200);
        pointLight.position.set(0, 10, 10);
        this.scene.add(pointLight);
        
        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 2000);
        // Adjust camera position based on the scale of WaltHead
        this.camera.position.z = 150; 
        this.camera.position.y = 20;

        this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        this.particleSystem = null;
        this.isForming = false;
        
        this.mouseX = 0;
        this.mouseY = 0;
        this.targetRotationX = 0;
        this.targetRotationY = 0;

        this.loadModel();
        this.bindEvents();
        this.animate();

        this.container.style.opacity = '0';
        this.container.style.pointerEvents = 'none';
        this.container.style.transition = 'opacity 2s ease';
    }

    loadModel() {
        if (typeof THREE.OBJLoader === 'undefined') {
            console.error('THREE.OBJLoader is not loaded');
            return;
        }

        const loader = new THREE.OBJLoader();
        loader.load('/models/WaltHead.obj', (object) => {
            let geometry = null;
            object.traverse((child) => {
                if (child.isMesh && child.geometry) {
                    geometry = child.geometry;
                }
            });

            if (geometry) {
                // If it's a BufferGeometry, extract the position array
                let positions = geometry.attributes.position.array;
                
                const targetPositions = new Float32Array(positions.length);
                const startPositions = new Float32Array(positions.length);
                const colors = new Float32Array(positions.length);

                const colorCyan = new THREE.Color(0x00f0ff);
                const colorOrange = new THREE.Color(0xff8c00);

                for (let i = 0; i < positions.length; i += 3) {
                    let x = positions[i];
                    let y = positions[i + 1];
                    let z = positions[i + 2];
                    
                    // WaltHead is small, scale it up
                    const scale = 1.2;
                    x *= scale;
                    y *= scale;
                    z *= scale;

                    targetPositions[i] = x;
                    targetPositions[i + 1] = y;
                    targetPositions[i + 2] = z;

                    // Start positions (scattered widely like powder)
                    startPositions[i] = (Math.random() - 0.5) * 1200;
                    startPositions[i + 1] = (Math.random() - 0.5) * 1200;
                    startPositions[i + 2] = (Math.random() - 0.5) * 1200;
                    
                    // Color gradient: Center/Face is orange, edges/back are cyan
                    // The face is typically at +z, center x, center y
                    const distToCore = Math.sqrt(x*x + (y-10)*(y-10) + (z-15)*(z-15));
                    let c = colorCyan.clone();
                    if (distToCore < 22) {
                        c.lerp(colorOrange, 1.0 - (distToCore/22));
                    }
                    
                    colors[i] = c.r;
                    colors[i + 1] = c.g;
                    colors[i + 2] = c.b;
                }

                const particleGeometry = new THREE.BufferGeometry();
                particleGeometry.setAttribute('position', new THREE.BufferAttribute(startPositions, 3));
                particleGeometry.setAttribute('targetPosition', new THREE.BufferAttribute(targetPositions, 3));
                particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

                const material = new THREE.PointsMaterial({
                    size: 0.6,
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.85,
                    blending: THREE.AdditiveBlending,
                    depthWrite: false
                });

                this.particleSystem = new THREE.Points(particleGeometry, material);
                
                // Adjust position lower to align with center
                this.particleSystem.position.y = -20;
                
                this.scene.add(this.particleSystem);
            }
        });
    }

    bindEvents() {
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });

        document.addEventListener('mousemove', (e) => {
            this.mouseX = (e.clientX / window.innerWidth) * 2 - 1;
            this.mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
        });
    }

    formAvatar() {
        if (this.isForming) return;
        this.container.style.opacity = '1';
        this.isForming = true;
        
        if (window.jarvisAudio) window.jarvisAudio.playBoot();
    }

    hideAvatar() {
        if (!this.isForming) return;
        this.container.style.opacity = '0';
        this.isForming = false;
        
        if (this.particleSystem) {
            const positions = this.particleSystem.geometry.attributes.position.array;
            for (let i = 0; i < positions.length; i++) {
                positions[i] = (Math.random() - 0.5) * 1200;
            }
            this.particleSystem.geometry.attributes.position.needsUpdate = true;
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        this.targetRotationY = this.mouseX * 0.4; 
        this.targetRotationX = -this.mouseY * 0.2; 
        
        if (this.particleSystem) {
            this.particleSystem.rotation.y += (this.targetRotationY - this.particleSystem.rotation.y) * 0.05;
            this.particleSystem.rotation.x += (this.targetRotationX - this.particleSystem.rotation.x) * 0.05;

            if (this.isForming) {
                const positions = this.particleSystem.geometry.attributes.position.array;
                const targets = this.particleSystem.geometry.attributes.targetPosition.array;
                
                let doneCount = 0;
                for (let i = 0; i < positions.length; i++) {
                    const diff = targets[i] - positions[i];
                    if (Math.abs(diff) > 0.1) {
                        positions[i] += diff * (0.015 + Math.random() * 0.03); 
                    } else {
                        positions[i] = targets[i];
                        doneCount++;
                    }
                }
                
                this.particleSystem.geometry.attributes.position.needsUpdate = true;
                
                if (doneCount > positions.length * 0.9) {
                    const time = Date.now() * 0.001;
                    const scale = 1 + Math.sin(time * 2) * 0.01;
                    this.particleSystem.scale.set(scale, scale, scale);
                }
            }
        }

        this.renderer.render(this.scene, this.camera);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.jarvisAvatar = new AvatarEngine();
    
    window.showHolographicAvatar = () => {
        if (window.jarvisAvatar) window.jarvisAvatar.formAvatar();
    };
    window.hideHolographicAvatar = () => {
        if (window.jarvisAvatar) window.jarvisAvatar.hideAvatar();
    };
});
