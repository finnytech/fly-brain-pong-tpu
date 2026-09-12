# 🪰 Drosophila Connectome Pong: TPU v5e Neuro-Arena

Biologisch-realistische Reinforcement-Learning Simulation zweier virtueller Fruchtfliegen-Gehirne (*Drosophila melanogaster*), die in einer interaktiven 60 FPS Cyberpunk-Arena gegeneinander Pong spielen.

Basierend auf den wissenschaftlich kartierten neuronalen Schaltkreisen des vollständigen adulten Fruchtfliegen-Konnektoms (**FlyWire Consortium / Nature Oktober 2024**).

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

Das System nutzt eine eigens entwickelte, hardwarebeschleunigte Web-Oberfläche:
- **60 FPS HTML5 Canvas Pong**: Die Schläger sind **lebendig animierte Fruchtfliegen**, deren Flügel je nach Erregungszustand (Octopamin-Spiegel) realistisch flattern. Mit Ballschweif und Kollisionsfunken.
- **Interaktive 3D-Konnektom-Gehirnkarten (Three.js WebGL)**:
  - Zeigt für beide Fliegen die 3D-Gehirnareale (*Optic Lobe*, *Central Complex*, *Mushroom Body*, *Descending Neurons*).
  - **Happy vs. Sauer (Dynamic Glow)**:
    - ✨ **Happy (PAM Dopamin)**: Gehirnareale erstrahlen in hellem Smaragdgrün & Gold.
    - ⚡ **Sauer / Schmerz (PPL1 Stress)**: Gehirnareale flammen in Warn-Karminrot & Neon-Orange auf.
  - Per Maus frei im 3D-Raum rotierbar.
- **RL-Skill-Level & Loss-Graphen**:
  - Dynamisches Skill-Tier: *Larva (Lvl 1)* $\to$ *Pupa (Lvl 12)* $\to$ *Fly Pilot (Lvl 35)* $\to$ *Connectome Ace (Lvl 70)* $\to$ *Apex Drosophila*.
  - Trefferquoten, Gesamt-Hits und Echtzeit-Verlauf des RL Policy Loss via Chart.js.

---

## ⚡ Schnellstart in Google Colab (TPU v5e)

1. Öffne [Google Colab](https://colab.research.google.com/) und wähle unter **Laufzeit > Laufzeittyp ändern** als Beschleuniger **TPU v5e**.
2. Lade das Notebook [`colab_fly_brain_pong.ipynb`](colab_fly_brain_pong.ipynb) hoch.
3. Starte die Zellen nacheinander:
   - Mountet Google Drive für das 20-Minuten-Autosave.
   - Installiert die Pakete (`fastapi`, `uvicorn`, `websockets`, `jax[tpu]`, etc.).
   - Startet den WebSocket-Server und den sicheren, kostenlosen öffentlichen Tunnel.

---

## 💾 20-Minuten Auto-Checkpointing

- Speichert **alle 20 Minuten (1200 Sekunden)** automatisch den kompletten synaptischen Zustand beider Fliegenhirne als `safetensors`.
- In Colab direkt in `/content/drive/MyDrive/fly_brain_checkpoints/`.
- Nach Disconnects oder Neustarts wird automatisch der aktuellste Checkpoint geladen.
- Ein manueller Save-Button im Dashboard erlaubt jederzeit sofortiges Sichern.

---

## 💻 Lokale Ausführung

```powershell
# Abhängigkeiten installieren
pip install -r requirements.txt

# Web-App & Server starten (Port 8000)
python main.py --port 8000

# Browser öffnen: http://localhost:8000
```

---

## 🚀 Push in dein privates GitHub-Repository

```powershell
cd D:\fly_brain_pong_tpu
git remote add origin https://github.com/finnytech/<dein-privates-repo-name>.git
git push -u origin main
```
