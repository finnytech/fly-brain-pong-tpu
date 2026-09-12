"""
Auto-Checkpointer for Virtual Drosophila Brains
Saves trained fly connectomes every 20 minutes to Google Drive or local storage.
"""
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

try:
    from safetensors.numpy import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

class FlyBrainCheckpointer:
    """
    Manages automated periodic checkpoints every 20 minutes (1200s).
    Preserves trained synaptic weights across Google Colab sessions.
    """
    def __init__(self, 
                 checkpoint_dir: str = "./checkpoints", 
                 gdrive_dir: str = "/content/drive/MyDrive/fly_brain_checkpoints",
                 interval_minutes: float = 20.0):
        self.interval_seconds = interval_minutes * 60.0
        self.last_save_time = time.time()
        
        # Detect if Google Drive is available in Colab
        if os.path.exists("/content/drive/MyDrive"):
            self.save_dir = Path(gdrive_dir)
            print(f"[Checkpointer] Google Drive detected! Saving checkpoints to: {self.save_dir}")
        else:
            self.save_dir = Path(checkpoint_dir)
            print(f"[Checkpointer] Local storage mode: {self.save_dir}")
            
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_count = 0
        
    def save_checkpoint(self, fly1, fly2, step: int, score1: int, score2: int) -> str:
        """Saves current state of both fly brains."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"fly_brains_step_{step}_{timestamp}"
        
        weights1 = fly1.get_weights()
        weights2 = fly2.get_weights()
        
        # Flatten into unified dictionary with prefixes
        unified_weights = {}
        for k, v in weights1.items():
            unified_weights[f"fly1_{k}"] = np.ascontiguousarray(v)
        for k, v in weights2.items():
            unified_weights[f"fly2_{k}"] = np.ascontiguousarray(v)
            
        metadata = {
            "step": step,
            "timestamp": timestamp,
            "score1": score1,
            "score2": score2,
            "fly1_total_hits": fly1.total_hits,
            "fly1_total_wins": fly1.total_wins,
            "fly2_total_hits": fly2.total_hits,
            "fly2_total_wins": fly2.total_wins,
        }
        
        if HAS_SAFETENSORS:
            ckpt_path = self.save_dir / f"{prefix}.safetensors"
            save_file(unified_weights, str(ckpt_path), metadata={k: str(v) for k, v in metadata.items()})
            # Also update "latest.safetensors"
            latest_path = self.save_dir / "latest_fly_brains.safetensors"
            save_file(unified_weights, str(latest_path), metadata={k: str(v) for k, v in metadata.items()})
        else:
            ckpt_path = self.save_dir / f"{prefix}.npz"
            np.savez_compressed(str(ckpt_path), **unified_weights)
            latest_path = self.save_dir / "latest_fly_brains.npz"
            np.savez_compressed(str(latest_path), **unified_weights)
            
        # Write metadata json
        meta_path = self.save_dir / f"{prefix}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        self.last_save_time = time.time()
        self.checkpoint_count += 1
        print(f"\n[Checkpointer] Saved 20-minute checkpoint #{self.checkpoint_count} to: {ckpt_path}")
        return str(ckpt_path)

    def check_and_save(self, fly1, fly2, step: int, score1: int, score2: int, force: bool = False) -> Optional[str]:
        """
        Periodically checks elapsed time; saves if >= 20 minutes or if forced.
        """
        elapsed = time.time() - self.last_save_time
        if force or (elapsed >= self.interval_seconds):
            return self.save_checkpoint(fly1, fly2, step, score1, score2)
        return None

    def load_latest(self, fly1, fly2) -> bool:
        """Loads the most recent checkpoint if available."""
        target = None
        is_safetensors = False
        
        st_file = self.save_dir / "latest_fly_brains.safetensors"
        npz_file = self.save_dir / "latest_fly_brains.npz"
        
        if st_file.exists() and HAS_SAFETENSORS:
            target = st_file
            is_safetensors = True
        elif npz_file.exists():
            target = npz_file
            is_safetensors = False
            
        if not target:
            # Check for any numbered files
            st_files = sorted(self.save_dir.glob("fly_brains_step_*.safetensors"))
            npz_files = sorted(self.save_dir.glob("fly_brains_step_*.npz"))
            if st_files and HAS_SAFETENSORS:
                target = st_files[-1]
                is_safetensors = True
            elif npz_files:
                target = npz_files[-1]
                is_safetensors = False
                
        if not target:
            print("[Checkpointer] No previous checkpoints found. Starting with fresh Drosophila brains.")
            return False
            
        print(f"[Checkpointer] Loading fly brains from: {target}")
        if is_safetensors:
            data = load_file(str(target))
        else:
            data = dict(np.load(str(target)))
            
        weights1 = {k[5:]: v for k, v in data.items() if k.startswith("fly1_")}
        weights2 = {k[5:]: v for k, v in data.items() if k.startswith("fly2_")}
        
        fly1.load_weights(weights1)
        fly2.load_weights(weights2)
        print("[Checkpointer] Successfully loaded trained fly brain synapses!")
        return True
