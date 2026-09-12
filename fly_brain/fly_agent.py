"""
FlyAgent: Autonomous Virtual Fruit Fly Brain (Drosophila melanogaster)
Integrates biological connectome circuits, dopamine/stress neuromodulation,
and reinforcement learning action selection.
"""
import numpy as np
from typing import Dict, Any, Tuple
from fly_brain.connectome_circuits import DrosophilaConnectome
from fly_brain.neuromodulation import DrosophilaNeuromodulator

class FlyAgent:
    """
    Autonomous agent controlled entirely by a virtual Drosophila connectome.
    """
    def __init__(self, fly_id: str = "Fly-1", visual_dim: int = 16, num_kcs: int = 256, seed: int = 42):
        self.fly_id = fly_id
        self.connectome = DrosophilaConnectome(visual_dim=visual_dim, num_kcs=num_kcs, seed=seed)
        self.neuromodulator = DrosophilaNeuromodulator()
        self.total_matches = 0
        self.total_wins = 0
        self.total_hits = 0
        self.recent_actions = []
        self.last_kc = None
        self.last_mbon = None
        
    def act(self, visual_frame: np.ndarray, paddle_pos: float, temperature: float = 0.5) -> int:
        """
        Processes visual input through the fly brain and selects an action.
        Actions:
        0 = PADDLE UP (-1 in game coordinate system)
        1 = PADDLE STILL (0)
        2 = PADDLE DOWN (+1 in game coordinate system)
        """
        proprioception = np.array([paddle_pos, 0.0])
        logits, kc_spikes, mbon_activity = self.connectome.forward(visual_frame, proprioception)
        
        self.last_kc = kc_spikes
        self.last_mbon = mbon_activity
        
        # Softmax probability with temperature exploration
        exp_logits = np.exp((logits - np.max(logits)) / max(temperature, 1e-2))
        probs = exp_logits / np.sum(exp_logits)
        
        action = int(np.random.choice(3, p=probs))
        self.recent_actions.append(action)
        if len(self.recent_actions) > 50:
            self.recent_actions.pop(0)
            
        return action
        
    def on_event(self, ball_hit: bool, point_won: bool, ball_missed: bool, point_lost: bool, lr: float = 0.02):
        """
        Processes biological reward or punishment event.
        Releases Dopamine (PAM) on win/hit, releases Stress/Pain (PPL1) on loss/miss.
        Updates plastic Kenyon Cell -> MBON synapses accordingly.
        """
        if ball_hit:
            self.total_hits += 1
        if point_won:
            self.total_wins += 1
            
        # Neuromodulator pulse
        nm_state = self.neuromodulator.step(
            ball_hit=ball_hit,
            point_won=point_won,
            ball_missed=ball_missed,
            point_lost=point_lost
        )
        
        # Apply three-factor dopamine/stress plasticity
        net_reinforcement = self.neuromodulator.get_effective_reinforcement()
        if self.last_kc is not None and self.last_mbon is not None:
            self.connectome.apply_dopamine_plasticity(
                kc_spikes=self.last_kc,
                mbon_activity=self.last_mbon,
                net_reinforcement=net_reinforcement,
                lr=lr
            )
            
        return nm_state
        
    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic stats for web visualization."""
        nm = self.neuromodulator.state
        records = self.connectome.latest_spike_records
        return {
            "fly_id": self.fly_id,
            "dopamine_pam": float(nm.pam_dopamine),
            "stress_ppl1": float(nm.ppl1_stress_pain),
            "octopamine_arousal": float(nm.octopamine_arousal),
            "total_hits": self.total_hits,
            "total_wins": self.total_wins,
            "lamina_activity": records.get("lamina_mean", 0.0),
            "t4t5_motion": records.get("t4t5_activity", 0.0),
            "lptc_vertical": records.get("lptc_vs_vertical", 0.0),
            "lptc_horizontal": records.get("lptc_hs_horizontal", 0.0),
            "epg_compass_heading": records.get("epg_bump_heading", 0.0),
            "active_kenyon_cells": records.get("kc_active_count", 0),
            "mbon_drive": records.get("mbon_mean", 0.0),
        }
        
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Serializes synaptic weight matrices for saving."""
        return {
            "w_ommatidia_to_lamina_on": self.connectome.w_ommatidia_to_lamina_on,
            "w_ommatidia_to_lamina_off": self.connectome.w_ommatidia_to_lamina_off,
            "w_lamina_to_medulla": self.connectome.w_lamina_to_medulla,
            "w_medulla_to_t4t5": self.connectome.w_medulla_to_t4t5,
            "w_t4t5_to_lptc": self.connectome.w_t4t5_to_lptc,
            "w_lptc_to_ring": self.connectome.w_lptc_to_ring,
            "w_ring_to_epg": self.connectome.w_ring_to_epg,
            "w_epg_recurrent": self.connectome.w_epg_recurrent,
            "w_epg_to_pen": self.connectome.w_epg_to_pen,
            "w_cx_to_kc": self.connectome.w_cx_to_kc,
            "w_kc_to_mbon": self.connectome.w_kc_to_mbon,
            "w_mbon_to_dn": self.connectome.w_mbon_to_dn,
            "w_cx_to_dn": self.connectome.w_cx_to_dn,
            "total_hits": np.array([self.total_hits]),
            "total_wins": np.array([self.total_wins]),
        }
        
    def load_weights(self, weights: Dict[str, np.ndarray]):
        """Restores synaptic weight matrices from checkpoint."""
        for k, v in weights.items():
            if hasattr(self.connectome, k):
                setattr(self.connectome, k, v)
        if "total_hits" in weights:
            self.total_hits = int(weights["total_hits"][0])
        if "total_wins" in weights:
            self.total_wins = int(weights["total_wins"][0])
