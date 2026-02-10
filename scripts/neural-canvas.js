/* ============================================================
   NEURAL CANVAS — Living neural network background
   ============================================================ */

export class NeuralCanvas {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.nodes = [];
    this.mouse = { x: -9999, y: -9999 };
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.frame = 0;
    this.running = true;

    this._onResize = this._resize.bind(this);
    this._onMouse = this._trackMouse.bind(this);

    window.addEventListener('resize', this._onResize);
    window.addEventListener('mousemove', this._onMouse, { passive: true });

    this._resize();
    this._populate();
    this._loop();
  }

  /* ---- Setup ---- */

  _resize() {
    this.w = window.innerWidth;
    this.h = window.innerHeight;
    this.canvas.width = this.w * this.dpr;
    this.canvas.height = this.h * this.dpr;
    this.canvas.style.width = this.w + 'px';
    this.canvas.style.height = this.h + 'px';
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  _populate() {
    const area = this.w * this.h;
    const count = Math.min(70, Math.max(25, Math.floor(area / 18000)));
    this.nodes = [];

    for (let i = 0; i < count; i++) {
      this.nodes.push({
        x: Math.random() * this.w,
        y: Math.random() * this.h,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        r: 1.2 + Math.random() * 1.8,
        baseAlpha: 0.12 + Math.random() * 0.2,
        pulse: 0,          // current pulse brightness (0-1)
        hue: Math.random() > 0.75 ? 'violet' : 'cyan',
      });
    }
  }

  _trackMouse(e) {
    this.mouse.x = e.clientX;
    this.mouse.y = e.clientY;
  }


  /* ---- Animation loop ---- */

  _loop() {
    if (!this.running) return;
    this.frame++;
    this._update();
    this._draw();
    requestAnimationFrame(() => this._loop());
  }

  _update() {
    const { nodes, w, h, mouse } = this;

    // Auto-pulse wave every ~480 frames (~8s at 60fps)
    if (this.frame % 480 === 0) {
      this._triggerPulse();
    }

    for (const n of nodes) {
      // Drift
      n.x += n.vx;
      n.y += n.vy;

      // Wrap edges
      if (n.x < -10) n.x = w + 10;
      if (n.x > w + 10) n.x = -10;
      if (n.y < -10) n.y = h + 10;
      if (n.y > h + 10) n.y = -10;

      // Subtle random drift change
      if (Math.random() < 0.005) {
        n.vx += (Math.random() - 0.5) * 0.05;
        n.vy += (Math.random() - 0.5) * 0.05;
        n.vx = Math.max(-0.4, Math.min(0.4, n.vx));
        n.vy = Math.max(-0.4, Math.min(0.4, n.vy));
      }

      // Mouse proximity glow
      const dx = n.x - mouse.x;
      const dy = n.y - mouse.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      n.mouseGlow = dist < 200 ? (1 - dist / 200) * 0.3 : 0;

      // Decay pulse
      if (n.pulse > 0) {
        n.pulse *= 0.96;
        if (n.pulse < 0.01) n.pulse = 0;
      }
    }
  }

  _draw() {
    const { ctx, w, h, nodes } = this;
    ctx.clearRect(0, 0, w, h);

    // Draw grid underlay
    this._drawGrid();

    // Draw connections
    const maxDist = 130;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < maxDist) {
          const alpha = (1 - dist / maxDist) * 0.06;
          const pulseBoost = (a.pulse + b.pulse) * 0.08;

          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(76, 201, 240, ${alpha + pulseBoost})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    for (const n of nodes) {
      const alpha = n.baseAlpha + n.mouseGlow + n.pulse * 0.5;
      const r = n.r + n.pulse * 2;

      // Glow
      if (alpha > 0.15) {
        const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 4);
        const color = n.hue === 'violet'
          ? `rgba(123, 47, 247, ${alpha * 0.3})`
          : `rgba(76, 201, 240, ${alpha * 0.3})`;
        grad.addColorStop(0, color);
        grad.addColorStop(1, 'transparent');
        ctx.beginPath();
        ctx.arc(n.x, n.y, r * 4, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
      }

      // Core
      const coreColor = n.hue === 'violet'
        ? `rgba(167, 139, 250, ${alpha})`
        : `rgba(76, 201, 240, ${alpha})`;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = coreColor;
      ctx.fill();
    }
  }

  _drawGrid() {
    const { ctx, w, h } = this;
    const spacing = 80;
    ctx.strokeStyle = 'rgba(76, 201, 240, 0.018)';
    ctx.lineWidth = 0.5;

    for (let x = 0; x < w; x += spacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += spacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  }


  /* ---- Pulse ---- */

  _triggerPulse() {
    if (this.nodes.length === 0) return;
    const seed = this.nodes[Math.floor(Math.random() * this.nodes.length)];
    this._spreadPulse(seed, 1.0, new Set());
  }

  _spreadPulse(node, strength, visited) {
    if (strength < 0.1 || visited.has(node)) return;
    visited.add(node);
    node.pulse = Math.max(node.pulse, strength);

    // Spread to nearby nodes with decay
    for (const n of this.nodes) {
      if (visited.has(n)) continue;
      const dx = node.x - n.x;
      const dy = node.y - n.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 150) {
        const decay = (1 - dist / 150) * 0.6;
        setTimeout(() => {
          this._spreadPulse(n, strength * decay, visited);
        }, dist * 2);
      }
    }
  }

  /** Trigger a pulse from a random position (for interaction) */
  pulse() {
    this._triggerPulse();
  }

  destroy() {
    this.running = false;
    window.removeEventListener('resize', this._onResize);
    window.removeEventListener('mousemove', this._onMouse);
  }
}
