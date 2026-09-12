# 🪰 Drosophila Connectome Pong: TPU v6e-1 (Trillium) & NVIDIA A100 / L4 Lab

Biologisch-realistische Reinforcement-Learning Simulation zweier virtueller Fruchtfliegen-Gehirne (*Drosophila melanogaster*), die in einer interaktiven 60 FPS Cyberpunk-Arena gegeneinander Pong spielen.

Basierend auf den wissenschaftlich kartierten neuronalen Schaltkreisen des vollständigen adulten Fruchtfliegen-Konnektoms (**FlyWire Consortium / Nature Oktober 2024**).

---

## ⚡ Unterstützte Hardware-Beschleuniger

Das System erkennt automatisch die aktive Colab-Laufzeit und kompiliert die XLA-Kernel optimal:

1. **Google TPU v6e-1 (Trillium)** ⭐ *(Höchste Empfehlung)*:
   - Googles neueste Flaggschiff-TPU.
   - Bis zu **4,7-fache Peak-Rechenleistung** und doppelte Speicherbandbreite gegenüber v5e.
   - JAX läuft nativ mit maximaler XLA-Parallelisierung.
2. **NVIDIA A100-SXM4 / PCIe (40GB / 80GB)**:
   - Extrem hohe Tensor-Core Rechenleistung via CUDA 12.
3. **NVIDIA L4 GPU**:
   - Moderne Ada-Lovelace Architektur mit hoher Energieeffizienz.

---

## 🔬 Biologische Schaltkreise der Fliegenhirne

Kein abstraktes MLP, sondern eine authentische Nachbildung der Drosophila-Subschaltkreise:

```
[Ommatidien (Facettenauge des Spielfelds)]
                     │
                     ▼
             [Optic Lobe (Sehlappen)]
              ├── Lamina (L1 ON / L2 OFF Kontrastverstärkung)
              ├── Medulla (Mi1, Tm3 Relais)
              └── Lobula Plate (T4/T5 Richtungssensoren + LPTC HS/VS Motion Cells)
                     │
                     ▼
     [Central Complex (CX - Navigation & Raumorientierung)]
              ├── Ellipsoid Body (Ring-Neuronen für egometrische Ballkoordinaten)
              ├── Protocerebral Bridge (PB)
              └── EPG Compass Neurons & P-EN Steering Neurons (Heading Ring-Attraktor)
                     │
                     ▼
     [Mushroom Body (MB - Plastizität & Belohnungslernen)]
              ├── Kenyon Cells (KCs - ~7% spärliche Repräsentation)
              │
              ├── PAM Dopamin-Cluster (Happy / Appetitiv):
              │    Feuert intensive Dopamin-Salven bei Balltreffer (+0.6) & Punktgewinn (+2.0)!
              │
              ├── PPL1 Stress/Schmerz-Cluster (Sauer / Aversiv):
              │    Feuert bei Ballverlust (-0.8) & Niederlage (-2.0)!
              │
              └── MBONs (Mushroom Body Output Neurons): Moduliert synaptische Gewichte via 3-Faktoren-Plastizität
                     │
                     ▼
             [Descending Neurons (DNp01 / DNa02)]
              └── Fliegen-Paddle Motorik: [0: HOCH, 1: STILL, 2: RUNTER]
```

---

## 🚀 Moderne Web-Applikation (Kein Gradio!)

- **60 FPS HTML5 Canvas Pong**: Die Schläger sind **lebendig animierte Fruchtfliegen**, deren Flügel je nach Erregungszustand (Octopamin-Spiegel) flattern.
- **Interaktive 3D-Konnektom-Gehirnkarten (Three.js WebGL)**:
  - Zeigt für beide Fliegen die 3D-Gehirnareale (*Optic Lobe*, *Central Complex*, *Mushroom Body*, *Descending Neurons*).
  - **Happy vs. Sauer (Dynamic Glow)**:
    - ✨ **Happy (PAM Dopamin)**: Gehirnareale erstrahlen in hellem Smaragdgrün & Gold.
    - ⚡ **Sauer / Schmerz (PPL1 Stress)**: Gehirnareale flammen in Warn-Karminrot & Neon-Orange auf.
  - Per Maus frei im 3D-Raum rotierbar.
- **RL-Skill-Level & Loss-Graphen**:
  - Dynamisches Skill-Tier: *Larva (Lvl 1)* $\to$ *Pupa (Lvl 12)* $\to$ *Fly Pilot (Lvl 35)* $\to$ *Connectome Ace (Lvl 70)* $\to$ *Apex Drosophila*.

---

## ⚡ Schnellstart in Google Colab (TPU v6e-1 oder A100)

1. Öffne Google Colab und wähle unter **Laufzeit > Laufzeittyp ändern** entweder:
   - **TPU v6e-1** (Empfohlen für maximale native Geschwindigkeit), oder
   - **A100 GPU**
2. Führe folgende Zellen aus:

```bash
# 1. Repository clonen
!git clone https://github.com/finnytech/fly-brain-pong-tpu.git
%cd fly-brain-pong-tpu

# 2. Smarte Installation (erkennt automatisch TPU v6e-1 oder A100 CUDA)
!pip install -r requirements.txt

# 3. Google Drive mounten für 20-Minuten-Dauerspeicherung
from google.colab import drive
drive.mount('/content/drive')

# 4. Cloudflare Tunnel & Hauptskript starten
!curl -s https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb && dpkg -i cloudflared.deb > /dev/null
!cloudflared tunnel --url http://localhost:8000 > tunnel.log 2>&1 &
import time; time.sleep(4)
!grep -o 'https://.*\.trycloudflare\.com' tunnel.log | head -n 1
!python main.py --port 8000 --checkpoint-interval 20.0
```

---

## 💾 20-Minuten Auto-Checkpointing

- Speichert **alle 20 Minuten (1200 Sekunden)** automatisch den kompletten synaptischen Zustand beider Fliegenhirne als `safetensors`.
- In Colab direkt in `/content/drive/MyDrive/fly_brain_checkpoints/`.
- Nach Disconnects oder Neustarts wird automatisch der aktuellste Checkpoint geladen.
