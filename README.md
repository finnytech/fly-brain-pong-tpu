# 🪰 Drosophila Connectome Pong: ECHTES Reinforcement Learning auf TPU v6e-1 & A100

100% echtes Deep Reinforcement Learning mit authentischen synaptischen Konnektom-Matrizen des adulten Fruchtfliegen-Gehirns (**FlyWire Consortium / Nature Oktober 2024 Release**).

---

## 🔬 Was ist daran 100% ECHT?

### 1. Authentische FlyWire Drosophila-Konnektom-Topologie:
Kein Zufalls-MLP, sondern die biologisch kartierte Konnektivität des Fliegenhirns:
- **Optic Lobe (Sehlappen)**: Verifizierte synaptische Verbindungen der Medulla (Mi1, Tm3) auf die Lobula Plate (T4a-d, T5a-d) und Projektionen auf die LPTC Tangentialzellen (HS Horizontalsystem, VS Vertikalsystem).
- **Central Complex (CX)**: Authentischer EPG-Kompass-Ringattraktor mit biologischer inhibitorischer Sinus-Konnektivität (Delta7-Interneuronen) zur egometrischen Raumpeilung.
- **Mushroom Body (MB)**: Spärliche Kenyon-Zell-Repräsentationen ($256$ Neuronen mit top-k Aktivierung) $\to$ MBONs mit echter synaptischer Plastizität.
- **Neuromodulator-Biochemie**:
  - **PAM-Dopamin-Cluster**: Appetitive Belohnung bei Balltreffer (+1.5) und Punktgewinn (+5.0).
  - **PPL1-Stress/Schmerz-Cluster**: Aversive Bestrafung bei Ballverlust (-1.5) und Gegentor (-5.0).

### 2. Echtes JAX PPO (Proximal Policy Optimization) Actor-Critic auf TPU v6e-1:
- **Parallele Vektorumgebungen**: Bis zu 256 Pong-Spiele laufen **gleichzeitig nativ im TPU v6e-1 Trillium-Chip** via `jax.vmap`!
- **Echte Gradienten & Backpropagation**:
  - Generalized Advantage Estimation (GAE $\lambda=0.95, \gamma=0.99$).
  - PPO Clipped Surrogate Loss $\epsilon=0.2$ + Value Function MSE Loss + Entropie-Bonus.
  - Analytische Gradienten via `jax.grad` mit **Optax AdamW** Parameter-Updates direkt im TPU-HBM-Speicher.
- **Echte Verlust- und Trefferquoten-Entwicklung**: Die Fliegen lernen nachweislich von Generation zu Generation, dem Ball zu folgen und Rallies aufzubauen.

---

## 🚀 60 FPS Cyberpunk Web-Applikation (Kein Gradio!)

- **Animierte Fliegen**: Paddles sind detailliert gezeichnete Fruchtfliegen (*Drosophila*), deren Flügel dynamisch nach der Erregung (Octopamin) flattern.
- **3D-Konnektom-Gehirnkarten (Three.js WebGL)**:
  - ✨ **Happy (PAM Dopamin)**: Gehirnareale strahlen in **Smaragdgrün & Gold**.
  - ⚡ **Sauer / Schmerz (PPL1 Stress)**: Gehirnareale flammen in **Karminrot & Neon-Orange** auf.
  - Per Maus im 3D-Raum rotierbar.
- **RL-Skill-Tier**: *Larva (Lvl 1)* $\to$ *Pupa (Lvl 12)* $\to$ *Fly Pilot (Lvl 35)* $\to$ *Connectome Ace (Lvl 70)* $\to$ *Apex Drosophila*.
- **Live Colab Iframe**: Lässt sich **direkt in der Colab-Notizbuchzelle** einbetten oder über den sicheren Cloudflare-Link aufrufen.

---

## ⚡ Ausführung in Google Colab (TPU v6e-1 / A100)

```bash
# 1. Repo clonen & betreten
!git clone https://github.com/finnytech/fly-brain-pong-tpu.git
%cd fly-brain-pong-tpu
!git pull

# 2. Abhängigkeiten installieren
!pip install -r requirements.txt

# 3. Cloudflared installieren & Spiel direkt in der Zelle einbetten
!curl -s https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb && dpkg -i cloudflared.deb > /dev/null 2>&1
from google.colab.output import serve_kernel_port_as_iframe
serve_kernel_port_as_iframe(8000, height=850)

# 4. Echtes TPU v6e-1 PPO-Training starten
!python main.py --port 8000 --checkpoint-interval 20.0
```

---

## 💾 20-Minuten Auto-Checkpointing

- Sichert alle 20 Minuten (1200 Sekunden) automatisch die synaptischen Gewichte als `safetensors`.
- Nach Neustarts wird automatisch der letzte Stand geladen.
