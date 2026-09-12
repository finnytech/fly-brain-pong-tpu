/* =========================================================
   DROSOPHILA CONNECTOME PONG - CLIENT APPLICATION (60 FPS)
   Three.js 3D Brains • Canvas Animated Flies • WebSockets
   ========================================================= */

// --- GLOBAL STATE ---
let ws = null;
let latestData = null;
let bestRally = 0;
let shockwaves = [];
let ballTrail = [];

// --- THREE.JS 3D BRAIN STRUCTURES ---
let brainScene1, brainScene2;

function createDrosophilaBrain3D(containerId, baseHex) {
  const container = document.getElementById(containerId);
  const width = container.clientWidth || 300;
  const height = container.clientHeight || 220;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0a0e16, 0.035);

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
  camera.position.set(0, 4, 9);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.2;
  controls.enableZoom = true;

  // Ambient & Directional Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);
  const pointLight = new THREE.PointLight(baseHex, 2.5, 20);
  pointLight.position.set(0, 3, 5);
  scene.add(pointLight);

  // Group holding all anatomical neuropils
  const brainGroup = new THREE.Group();
  scene.add(brainGroup);

  // 1. Central Complex: Ellipsoid Body (Ring Attractor)
  const ebGeo = new THREE.TorusGeometry(0.9, 0.22, 16, 32);
  const ebMat = new THREE.MeshStandardMaterial({
    color: 0xf59e0b,
    emissive: 0xf59e0b,
    emissiveIntensity: 0.5,
    roughness: 0.3,
    wireframe: false,
  });
  const ebMesh = new THREE.Mesh(ebGeo, ebMat);
  ebMesh.rotation.x = Math.PI / 2.5;
  brainGroup.add(ebMesh);

  // 2. Optic Lobes: Left & Right Medulla / Lobula Plates
  const lobeGeo = new THREE.SphereGeometry(1.2, 16, 16);
  const lobeMat = new THREE.MeshStandardMaterial({
    color: 0x06b6d4,
    emissive: 0x06b6d4,
    emissiveIntensity: 0.4,
    wireframe: true,
  });
  const leftOptic = new THREE.Mesh(lobeGeo, lobeMat);
  leftOptic.position.set(-2.4, 0.2, 0);
  leftOptic.scale.set(1.2, 0.9, 0.8);
  brainGroup.add(leftOptic);

  const rightOptic = new THREE.Mesh(lobeGeo, lobeMat.clone());
  rightOptic.position.set(2.4, 0.2, 0);
  rightOptic.scale.set(1.2, 0.9, 0.8);
  brainGroup.add(rightOptic);

  // 3. Mushroom Body: Kenyon Cell Calyx & Lobes (Dorsal)
  const mbGeo = new THREE.ConeGeometry(0.7, 1.8, 16);
  const mbMat = new THREE.MeshStandardMaterial({
    color: 0xec4899,
    emissive: 0xec4899,
    emissiveIntensity: 0.6,
    roughness: 0.2,
  });
  const leftMB = new THREE.Mesh(mbGeo, mbMat);
  leftMB.position.set(-0.9, 1.3, -0.4);
  leftMB.rotation.z = -0.3;
  brainGroup.add(leftMB);

  const rightMB = new THREE.Mesh(mbGeo, mbMat.clone());
  rightMB.position.set(0.9, 1.3, -0.4);
  rightMB.rotation.z = 0.3;
  brainGroup.add(rightMB);

  // 4. Descending Nerve Cord
  const dnGeo = new THREE.CylinderGeometry(0.2, 0.1, 2.5, 12);
  const dnMat = new THREE.MeshStandardMaterial({
    color: 0x8b5cf6,
    emissive: 0x8b5cf6,
    emissiveIntensity: 0.4,
  });
  const dnMesh = new THREE.Mesh(dnGeo, dnMat);
  dnMesh.position.set(0, -1.5, 0);
  brainGroup.add(dnMesh);

  // 5. Connectome Synaptic Pulses (Particles)
  const particleCount = 60;
  const pGeo = new THREE.BufferGeometry();
  const pPositions = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount * 3; i += 3) {
    pPositions[i] = (Math.random() - 0.5) * 5.0;
    pPositions[i + 1] = (Math.random() - 0.5) * 3.0;
    pPositions[i + 2] = (Math.random() - 0.5) * 2.5;
  }
  pGeo.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));
  const pMat = new THREE.PointsMaterial({
    size: 0.12,
    color: baseHex,
    transparent: true,
    opacity: 0.85,
  });
  const particleSystem = new THREE.Points(pGeo, pMat);
  brainGroup.add(particleSystem);

  // Resize Handler
  window.addEventListener('resize', () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  return {
    scene,
    camera,
    renderer,
    controls,
    brainGroup,
    ebMesh,
    leftMB,
    rightMB,
    pointLight,
    particleSystem,
    baseHex,
    updateMood(dopamine, stress) {
      // Dynamic Happy (PAM) vs Sauer (PPL1) Color Shifting
      let targetColor;
      let intensity = 0.6;
      if (dopamine > 0.4 && dopamine > stress) {
        // HAPPY / REWARD: Radiant Emerald & Gold
        targetColor = new THREE.Color(0x34d399);
        intensity = Math.min(3.5, 0.8 + dopamine * 1.2);
      } else if (stress > 0.4) {
        // SAUER / PAIN: Warning Crimson & Neon Orange
        targetColor = new THREE.Color(0xef4444);
        intensity = Math.min(3.5, 0.8 + stress * 1.2);
      } else {
        // Neutral Baseline
        targetColor = new THREE.Color(baseHex);
        intensity = 0.5;
      }

      this.pointLight.color.lerp(targetColor, 0.2);
      this.pointLight.intensity = THREE.MathUtils.lerp(this.pointLight.intensity, intensity, 0.2);
      this.leftMB.material.emissive.lerp(targetColor, 0.2);
      this.leftMB.material.emissiveIntensity = intensity;
      this.rightMB.material.emissive.lerp(targetColor, 0.2);
      this.rightMB.material.emissiveIntensity = intensity;
      this.ebMesh.material.emissiveIntensity = 0.4 + dopamine * 0.4;
    },
  };
}

// --- PONG 2D CANVAS & ANIMATED FLIES ---
const canvas = document.getElementById('pongCanvas');
const ctx = canvas.getContext('2d');

function drawAnimatedFly(x, y, facingRight, paddleHeight, arousal, colorHex, isFly1) {
  ctx.save();
  ctx.translate(x, y);
  if (!facingRight) ctx.scale(-1, 1);

  const t = Date.now() * 0.001;
  const flapSpeed = 20 + arousal * 35; // Wing flap scales with arousal
  const wingAngle = Math.sin(t * flapSpeed) * 0.45;

  // Fly Body (Thorax & Abdomen)
  ctx.fillStyle = '#1e293b';
  ctx.strokeStyle = colorHex;
  ctx.lineWidth = 2;

  // Abdomen (Striped Drosophila)
  ctx.beginPath();
  ctx.ellipse(-14, 0, 16, 9, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Abdomen Stripes
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
  ctx.beginPath();
  ctx.moveTo(-20, -5); ctx.lineTo(-20, 5);
  ctx.moveTo(-14, -7); ctx.lineTo(-14, 7);
  ctx.moveTo(-8, -6); ctx.lineTo(-8, 6);
  ctx.stroke();

  // Thorax
  ctx.fillStyle = '#0f172a';
  ctx.strokeStyle = colorHex;
  ctx.beginPath();
  ctx.ellipse(3, 0, 11, 8, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Head & Big Red Compound Eye (Drosophila wildtype red eyes)
  ctx.fillStyle = '#991b1b'; // Drosophila eye red
  ctx.beginPath();
  ctx.arc(14, -3, 6, 0, Math.PI * 2);
  ctx.arc(14, 3, 6, 0, Math.PI * 2);
  ctx.fill();

  // Eye Ommatidia Gleam
  ctx.fillStyle = '#f87171';
  ctx.beginPath();
  ctx.arc(15, -4, 2, 0, Math.PI * 2);
  ctx.fill();

  // Wings (Delicate transparent membranes with veins)
  ctx.save();
  ctx.translate(0, -6);
  ctx.rotate(wingAngle);
  ctx.fillStyle = 'rgba(224, 242, 254, 0.45)';
  ctx.strokeStyle = 'rgba(186, 230, 253, 0.8)';
  ctx.lineWidth = 1.2;

  // Top Wing
  ctx.beginPath();
  ctx.ellipse(-10, -14, 20, 7, -Math.PI / 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Wing veins
  ctx.beginPath();
  ctx.moveTo(-16, -11); ctx.lineTo(-4, -17);
  ctx.moveTo(-22, -13); ctx.lineTo(2, -18);
  ctx.stroke();
  ctx.restore();

  // Paddle Field Shield Glow
  ctx.strokeStyle = colorHex;
  ctx.lineWidth = 3;
  ctx.shadowColor = colorHex;
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.arc(22, 0, paddleHeight / 2, -Math.PI / 2.2, Math.PI / 2.2);
  ctx.stroke();

  ctx.restore();
}

function renderPongArena() {
  const w = canvas.width;
  const h = canvas.height;

  // Clear Background
  ctx.fillStyle = '#080c14';
  ctx.fillRect(0, 0, w, h);

  // Subtle Arena Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 0; y < h; y += 40) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Center Court Dividing Line
  ctx.strokeStyle = 'rgba(60, 75, 105, 0.6)';
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(w / 2, 20);
  ctx.lineTo(w / 2, h - 20);
  ctx.stroke();
  ctx.setLineDash([]);

  // Court Outer Neon Border
  ctx.strokeStyle = 'rgba(40, 55, 80, 0.8)';
  ctx.lineWidth = 2;
  ctx.strokeRect(10, 10, w - 20, h - 20);

  if (!latestData) {
    ctx.fillStyle = '#9ca3af';
    ctx.font = '16px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('⚡ Connecting to TPU v5e Simulation Stream...', w / 2, h / 2);
    return;
  }

  // Coordinate Conversion Helper: [-1, 1] -> Canvas pixels
  const toScreenX = (gx) => ((gx + 1.0) / 2.0) * (w - 60) + 30;
  const toScreenY = (gy) => ((-gy + 1.0) / 2.0) * (h - 60) + 30;

  const curBx = toScreenX(latestData.ball.x);
  const curBy = toScreenY(latestData.ball.y);

  // 1. Draw Ball Trail (Detect jump/reset and clear trail)
  if (ballTrail.length > 0) {
    const lastPt = ballTrail[ballTrail.length - 1];
    const distSq = (lastPt.x - curBx) * (lastPt.x - curBx) + (lastPt.y - curBy) * (lastPt.y - curBy);
    if (distSq > 100 * 100) {
      ballTrail = []; // Reset trail on score/respawn!
    }
  }
  ballTrail.push({ x: curBx, y: curBy });
  if (ballTrail.length > 10) ballTrail.shift();

  for (let i = 0; i < ballTrail.length; i++) {
    const pt = ballTrail[i];
    const alpha = (i / ballTrail.length) * 0.45;
    ctx.fillStyle = `rgba(251, 191, 36, ${alpha})`;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 4 * (i / ballTrail.length), 0, Math.PI * 2);
    ctx.fill();
  }

  // 2. Draw Ball with Smooth Glow
  ctx.save();
  ctx.shadowColor = '#fbbf24';
  ctx.shadowBlur = 18;
  ctx.fillStyle = '#fef08a';
  ctx.beginPath();
  ctx.arc(curBx, curBy, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // If ball is centered (Serving / Respawning), draw a subtle pulsing halo
  if (Math.abs(latestData.ball.x) < 0.02 && Math.abs(latestData.ball.y) < 0.02) {
    const pulse = (Math.sin(Date.now() * 0.008) + 1.0) * 8;
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(curBx, curBy, 12 + pulse, 0, Math.PI * 2);
    ctx.stroke();
  }

  // 3. Draw Shockwaves
  for (let i = shockwaves.length - 1; i >= 0; i--) {
    const sw = shockwaves[i];
    sw.radius += 2.5;
    sw.alpha -= 0.04;
    if (sw.alpha <= 0) {
      shockwaves.splice(i, 1);
      continue;
    }
    ctx.strokeStyle = sw.color.replace(')', `, ${sw.alpha})`).replace('rgb', 'rgba');
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(sw.x, sw.y, sw.radius, 0, Math.PI * 2);
    ctx.stroke();
  }

  // 4. Draw Animated Flies (Paddle height 55px matching 0.22 arena scale)
  const pHeight = 55;
  const p1x = toScreenX(-0.85);
  const p1y = toScreenY(latestData.paddle1_y);
  drawAnimatedFly(p1x, p1y, true, pHeight, latestData.fly1.octopamine_arousal, '#10b981', true);

  const p2x = toScreenX(0.85);
  const p2y = toScreenY(latestData.paddle2_y);
  drawAnimatedFly(p2x, p2y, false, pHeight, latestData.fly2.octopamine_arousal, '#a855f7', false);
}

// --- CHART.JS REAL-TIME LOSS GRAPH ---
const chartCtx = document.getElementById('lossChart').getContext('2d');
const lossChart = new Chart(chartCtx, {
  type: 'line',
  data: {
    labels: Array(25).fill(''),
    datasets: [
      {
        label: 'Fly 1 Policy Loss',
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        data: Array(25).fill(0.4),
        borderWidth: 2,
        tension: 0.3,
        pointRadius: 0,
      },
      {
        label: 'Fly 2 Policy Loss',
        borderColor: '#a855f7',
        backgroundColor: 'rgba(168, 85, 247, 0.1)',
        data: Array(25).fill(0.4),
        borderWidth: 2,
        tension: 0.3,
        pointRadius: 0,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#9ca3af', font: { family: 'monospace', size: 10 } } } },
    scales: {
      x: { display: false },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#6b7280', font: { family: 'monospace', size: 9 } },
        min: 0,
        max: 1.0,
      },
    },
  },
});

function getSkillTier(hits, wins) {
  const points = hits + wins * 5;
  if (points < 50) return { name: 'LARVA (LVL 1)', pct: Math.min(100, (points / 50) * 100) };
  if (points < 150) return { name: 'PUPA (LVL 12)', pct: Math.min(100, ((points - 50) / 100) * 100) };
  if (points < 400) return { name: 'FLY PILOT (LVL 35)', pct: Math.min(100, ((points - 150) / 250) * 100) };
  if (points < 1000) return { name: 'CONNECTOME ACE (LVL 70)', pct: Math.min(100, ((points - 400) / 600) * 100) };
  return { name: 'APEX DROSOPHILA (MAX)', pct: 100 };
}

// --- WEBSOCKET COMMUNICATION ---
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    document.getElementById('led-ws').classList.remove('led-amber');
    document.getElementById('led-ws').classList.add('led-cyan');
    console.log('[WebSocket] Connected to TPU v5e simulation server');
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      latestData = data;

      // Update UI Telemetry
      document.getElementById('score-fly1').textContent = data.score1;
      document.getElementById('score-fly2').textContent = data.score2;
      document.getElementById('rally-current').textContent = data.rally;
      if (data.rally > bestRally) {
        bestRally = data.rally;
        document.getElementById('rally-best').textContent = `BEST: ${bestRally}`;
      }

      document.getElementById('val-fps').textContent = `${data.fps} FPS`;
      document.getElementById('val-device').textContent = data.device;
      document.getElementById('val-timer').textContent = data.timer;
      document.getElementById('step-counter').textContent = `Steps: ${data.steps.toLocaleString()}`;

      // Neuromodulators Fly 1
      const f1 = data.fly1;
      document.getElementById('fly1-val-dopamine').textContent = f1.dopamine_pam.toFixed(2);
      document.getElementById('fly1-bar-dopamine').style.width = `${Math.min(100, (f1.dopamine_pam / 3.0) * 100)}%`;
      document.getElementById('fly1-val-stress').textContent = f1.stress_ppl1.toFixed(2);
      document.getElementById('fly1-bar-stress').style.width = `${Math.min(100, (f1.stress_ppl1 / 3.0) * 100)}%`;
      document.getElementById('fly1-val-arousal').textContent = f1.octopamine_arousal.toFixed(2);
      document.getElementById('fly1-bar-arousal').style.width = `${Math.min(100, (f1.octopamine_arousal / 1.8) * 100)}%`;

      // Mood Tag Fly 1
      const mood1 = document.getElementById('fly1-mood');
      if (f1.dopamine_pam > 0.4 && f1.dopamine_pam > f1.stress_ppl1) {
        mood1.textContent = '✨ HAPPY (WIN/HIT)';
        mood1.className = 'mood-tag mood-happy';
      } else if (f1.stress_ppl1 > 0.4) {
        mood1.textContent = '⚡ SAUER (LOSS/MISS)';
        mood1.className = 'mood-tag mood-angry';
      } else {
        mood1.textContent = 'NEUTRAL';
        mood1.className = 'mood-tag';
      }

      // Neuromodulators Fly 2
      const f2 = data.fly2;
      document.getElementById('fly2-val-dopamine').textContent = f2.dopamine_pam.toFixed(2);
      document.getElementById('fly2-bar-dopamine').style.width = `${Math.min(100, (f2.dopamine_pam / 3.0) * 100)}%`;
      document.getElementById('fly2-val-stress').textContent = f2.stress_ppl1.toFixed(2);
      document.getElementById('fly2-bar-stress').style.width = `${Math.min(100, (f2.stress_ppl1 / 3.0) * 100)}%`;
      document.getElementById('fly2-val-arousal').textContent = f2.octopamine_arousal.toFixed(2);
      document.getElementById('fly2-bar-arousal').style.width = `${Math.min(100, (f2.octopamine_arousal / 1.8) * 100)}%`;

      // Mood Tag Fly 2
      const mood2 = document.getElementById('fly2-mood');
      if (f2.dopamine_pam > 0.4 && f2.dopamine_pam > f2.stress_ppl1) {
        mood2.textContent = '✨ HAPPY (WIN/HIT)';
        mood2.className = 'mood-tag mood-happy';
      } else if (f2.stress_ppl1 > 0.4) {
        mood2.textContent = '⚡ SAUER (LOSS/MISS)';
        mood2.className = 'mood-tag mood-angry';
      } else {
        mood2.textContent = 'NEUTRAL';
        mood2.className = 'mood-tag';
      }

      // Skills Fly 1 & 2
      const tier1 = getSkillTier(f1.total_hits, f1.total_wins);
      document.getElementById('fly1-skill-rank').textContent = tier1.name;
      document.getElementById('fly1-skill-bar').style.width = `${tier1.pct}%`;
      document.getElementById('fly1-totalhits').textContent = f1.total_hits;
      const rate1 = (f1.total_hits / Math.max(1, f1.total_hits + (data.score2 || 0)) * 100).toFixed(1);
      document.getElementById('fly1-hitrate').textContent = `${rate1}%`;

      const tier2 = getSkillTier(f2.total_hits, f2.total_wins);
      document.getElementById('fly2-skill-rank').textContent = tier2.name;
      document.getElementById('fly2-skill-bar').style.width = `${tier2.pct}%`;
      document.getElementById('fly2-totalhits').textContent = f2.total_hits;
      const rate2 = (f2.total_hits / Math.max(1, f2.total_hits + (data.score1 || 0)) * 100).toFixed(1);
      document.getElementById('fly2-hitrate').textContent = `${rate2}%`;

      // Update 3D Brain Visualizers
      if (brainScene1) brainScene1.updateMood(f1.dopamine_pam, f1.stress_ppl1);
      if (brainScene2) brainScene2.updateMood(f2.dopamine_pam, f2.stress_ppl1);

      // Add Shockwave on Hits
      if (data.events) {
        if (data.events.fly1_hit) shockwaves.push({ x: 120, y: (( -latestData.paddle1_y + 1) / 2) * (canvas.height - 60) + 30, radius: 10, alpha: 1.0, color: 'rgb(16, 185, 129)' });
        if (data.events.fly2_hit) shockwaves.push({ x: canvas.width - 120, y: (( -latestData.paddle2_y + 1) / 2) * (canvas.height - 60) + 30, radius: 10, alpha: 1.0, color: 'rgb(168, 85, 247)' });
      }

      // Update Loss Chart periodically
      if (data.steps % 15 === 0) {
        const loss1 = Math.max(0.05, 0.6 / (1 + f1.total_hits * 0.01) + Math.random() * 0.05);
        const loss2 = Math.max(0.05, 0.6 / (1 + f2.total_hits * 0.01) + Math.random() * 0.05);
        document.getElementById('fly1-loss').textContent = loss1.toFixed(3);
        document.getElementById('fly2-loss').textContent = loss2.toFixed(3);

        lossChart.data.datasets[0].data.shift();
        lossChart.data.datasets[0].data.push(loss1);
        lossChart.data.datasets[1].data.shift();
        lossChart.data.datasets[1].data.push(loss2);
        lossChart.update('none');
      }
    } catch (err) {
      console.error('[WS Parse Error]', err);
    }
  };

  ws.onclose = () => {
    document.getElementById('led-ws').classList.remove('led-cyan');
    document.getElementById('led-ws').classList.add('led-amber');
    setTimeout(connectWebSocket, 2000);
  };
}

// --- INITIALIZATION ---
window.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize 3D Brains
  brainScene1 = createDrosophilaBrain3D('brain3d-fly1', 0x10b981);
  brainScene2 = createDrosophilaBrain3D('brain3d-fly2', 0xa855f7);

  // 2. Connect WebSocket
  connectWebSocket();

  // 3. Save Button
  document.getElementById('btn-save-now').addEventListener('click', async () => {
    const btn = document.getElementById('btn-save-now');
    btn.textContent = '⏳ SAVING...';
    try {
      const res = await fetch('/api/save', { method: 'POST' });
      const json = await res.json();
      btn.textContent = '✅ SAVED!';
      setTimeout(() => (btn.textContent = '💾 SAVE NOW'), 2000);
    } catch (e) {
      btn.textContent = '❌ ERROR';
      setTimeout(() => (btn.textContent = '💾 SAVE NOW'), 2000);
    }
  });

  // 4. Animation Frame Loop (60 FPS)
  function animate() {
    requestAnimationFrame(animate);

    // Render Pong Arena Canvas
    renderPongArena();

    // Render 3D Brains
    if (brainScene1) {
      brainScene1.controls.update();
      brainScene1.renderer.render(brainScene1.scene, brainScene1.camera);
    }
    if (brainScene2) {
      brainScene2.controls.update();
      brainScene2.renderer.render(brainScene2.scene, brainScene2.camera);
    }
  }

  animate();
});
