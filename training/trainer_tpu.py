"""
TPU v5e High-Speed RL Training Loop for Drosophila Connectome Pong
Accelerates the 2-fly match, applies PAM/PPL1 neuromodulation,
and coordinates with the 20-minute checkpointer.
"""
import time
import threading
from typing import Dict, Any, Optional
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from fly_brain.fly_agent import FlyAgent
from environment.pong_vectorized import TwoFlyPongEnv
from training.checkpointer import FlyBrainCheckpointer

class TpuFlyPongTrainer:
    """
    High-speed trainer coordinating 2 Drosophila connectome agents,
    hardware acceleration (TPU v5e / GPU / CPU), and live diagnostic streaming.
    """
    def __init__(self, 
                 checkpoint_interval_minutes: float = 20.0,
                 learning_rate: float = 0.02,
                 fps_target: int = 60):
        self.learning_rate = learning_rate
        self.fps_target = fps_target
        
        # Hardware inspection
        self.device_type = self._detect_hardware()
        
        # Instantiate 2 distinct virtual fruit fly brains
        self.fly1 = FlyAgent(fly_id="Fly-1_Left (Green)", seed=101)
        self.fly2 = FlyAgent(fly_id="Fly-2_Right (Purple)", seed=202)
        
        # Pong Environment
        self.env = TwoFlyPongEnv(visual_res=16)
        
        # Checkpointer (20 minutes)
        self.checkpointer = FlyBrainCheckpointer(interval_minutes=checkpoint_interval_minutes)
        self.checkpointer.load_latest(self.fly1, self.fly2)
        
        # Training state & telemetry
        self.is_running = False
        self.step_counter = 0
        self.fps = 0.0
        self.latest_frame = None
        self.lock = threading.Lock()
        
    def _detect_hardware(self) -> str:
        if HAS_JAX:
            try:
                devices = jax.devices()
                dev_kind = devices[0].platform.lower()
                kind_str = str(devices[0].device_kind).lower()
                
                if "tpu" in dev_kind or "tpu" in kind_str:
                    if "v6e" in kind_str or "v6" in kind_str:
                        dev_desc = f"Google TPU v6e-1 (Trillium {len(devices)} chip) - Max Speed"
                    elif "v5e" in kind_str:
                        dev_desc = f"Google TPU v5e ({len(devices)} chip)"
                    else:
                        dev_desc = f"Google TPU ({devices[0].device_kind})"
                elif "gpu" in dev_kind or "cuda" in dev_kind:
                    if "a100" in kind_str:
                        dev_desc = f"NVIDIA A100 Tensor Core GPU ({devices[0].device_kind})"
                    elif "l4" in kind_str:
                        dev_desc = f"NVIDIA L4 Ada Lovelace GPU ({devices[0].device_kind})"
                    else:
                        dev_desc = f"NVIDIA GPU ({devices[0].device_kind})"
                else:
                    dev_desc = f"JAX {dev_kind.upper()} ({devices[0].device_kind})"
                    
                print(f"\n[Hardware-Engine] >>> ACCELERATOR DETECTED: {dev_desc} <<<")
                return dev_desc
            except Exception as e:
                return f"JAX CPU (Fallback: {e})"
        return "NumPy CPU Engine"

    def step_once(self) -> Dict[str, Any]:
        """Executes a single high-speed biological step."""
        # 1. Capture visual ommatidial fields
        vision1 = self.env.render_ommatidia(flip_horizontal=False)
        vision2 = self.env.render_ommatidia(flip_horizontal=True)
        
        # 2. Both fly connectomes process vision and choose motor drive
        a1 = self.fly1.act(vision1, self.env.paddle1_y)
        a2 = self.fly2.act(vision2, self.env.paddle2_y)
        
        # 3. Environment advances
        _, _, events = self.env.step(a1, a2)
        
        # 4. Neuromodulation updates (PAM Dopamine & PPL1 Stress/Pain)
        self.fly1.on_event(
            ball_hit=events.fly1_hit,
            point_won=events.fly1_point_won,
            ball_missed=events.fly1_miss,
            point_lost=events.fly1_point_lost,
            lr=self.learning_rate
        )
        self.fly2.on_event(
            ball_hit=events.fly2_hit,
            point_won=events.fly2_point_won,
            ball_missed=events.fly2_miss,
            point_lost=events.fly2_point_lost,
            lr=self.learning_rate
        )
        
        self.step_counter += 1
        
        # 5. Check 20-minute auto-save
        self.checkpointer.check_and_save(
            self.fly1, self.fly2, 
            step=self.step_counter, 
            score1=self.env.score1, 
            score2=self.env.score2
        )
        
        return {
            "events": events,
            "score1": self.env.score1,
            "score2": self.env.score2,
            "rally": self.env.rally,
        }

    def run_training_loop(self, total_steps: Optional[int] = None):
        """Continuous execution loop with FPS tracking."""
        self.is_running = True
        t0 = time.time()
        steps_in_sec = 0
        
        print(f"[TPU-Engine] Starting virtual Drosophila Pong RL match on {self.device_type}...")
        while self.is_running:
            self.step_once()
            steps_in_sec += 1
            
            # Recalculate FPS every 50 steps
            if steps_in_sec >= 50:
                elapsed = time.time() - t0
                if elapsed > 0:
                    self.fps = steps_in_sec / elapsed
                t0 = time.time()
                steps_in_sec = 0
                
            if total_steps and self.step_counter >= total_steps:
                break
                
        self.is_running = False

    def start_background(self):
        """Runs the training in a non-blocking background thread."""
        thread = threading.Thread(target=self.run_training_loop, daemon=True)
        thread.start()
        return thread

    def get_live_metrics(self) -> Dict[str, Any]:
        """Provides snapshot metrics for the live public dashboard."""
        status1 = self.fly1.get_status()
        status2 = self.fly2.get_status()
        time_to_next_save = max(0, int(self.checkpointer.interval_seconds - (time.time() - self.checkpointer.last_save_time)))
        mins, secs = divmod(time_to_next_save, 60)
        
        return {
            "device": self.device_type,
            "steps": self.step_counter,
            "fps": round(self.fps, 1),
            "score1": self.env.score1,
            "score2": self.env.score2,
            "rally": self.env.rally,
            "fly1": status1,
            "fly2": status2,
            "checkpoint_timer": f"{mins:02d}:{secs:02d}",
            "checkpoint_count": self.checkpointer.checkpoint_count,
        }
