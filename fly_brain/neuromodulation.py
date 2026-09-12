"""
Neuromodulatory System of Drosophila Melanogaster:
PAM (Dopamine - Appetitive / Reward) vs PPL1 (Dopamine/Stress - Aversive / Punishment / Pain)
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class NeuromodulatorState:
    """State of the fly's neuromodulatory centers."""
    pam_dopamine: float = 0.0      # Appetitive reward signal (PAM cluster -> gamma5, beta2 lobes)
    ppl1_stress_pain: float = 0.0  # Aversive stress/pain signal (PPL1 cluster -> gamma1, alpha3 lobes)
    octopamine_arousal: float = 0.5 # Baseline arousal & motor readiness (VUMa2 neurons)
    cumulative_reward: float = 0.0
    cumulative_stress: float = 0.0

class DrosophilaNeuromodulator:
    """
    Simulates the biological neuromodulatory system of the fruit fly.
    - PAM Cluster (Protocerebral Anterior Medial): Fires intense dopamine bursts upon
      hitting the ball (+0.5) or winning a point (+2.0).
    - PPL1 Cluster (Protocerebral Posterior Lateral 1): Fires intense stress/pain/shock bursts
      upon missing the ball (-0.5) or losing a point / opponent goal (-2.0).
    """
    def __init__(self, decay_rate: float = 0.85, baseline_arousal: float = 0.5):
        self.decay_rate = decay_rate
        self.baseline_arousal = baseline_arousal
        self.state = NeuromodulatorState(octopamine_arousal=baseline_arousal)
        
    def step(self, ball_hit: bool, point_won: bool, ball_missed: bool, point_lost: bool) -> NeuromodulatorState:
        # Natural clearance / degradation of neurotransmitters in the synaptic cleft
        self.state.pam_dopamine *= self.decay_rate
        self.state.ppl1_stress_pain *= self.decay_rate
        
        # Appetitive Reward (Dopamine burst from PAM neurons)
        if point_won:
            self.state.pam_dopamine += 2.0
            self.state.cumulative_reward += 2.0
        elif ball_hit:
            self.state.pam_dopamine += 0.6
            self.state.cumulative_reward += 0.6
            
        # Aversive Punishment / Stress / Pain (PPL1 cluster activation)
        if point_lost:
            self.state.ppl1_stress_pain += 2.0
            self.state.cumulative_stress += 2.0
        elif ball_missed:
            self.state.ppl1_stress_pain += 0.8
            self.state.cumulative_stress += 0.8
            
        # Dynamic arousal modulation
        current_stimulus = self.state.pam_dopamine + self.state.ppl1_stress_pain
        self.state.octopamine_arousal = float(np.clip(
            0.4 * self.state.octopamine_arousal + 0.6 * (self.baseline_arousal + 0.3 * current_stimulus),
            0.1, 2.0
        ))
        
        return self.state
        
    def get_effective_reinforcement(self) -> float:
        """
        Net reinforcement signal driving synaptic plasticity in Kenyon Cell -> MBON synapses:
        Positive = Long-Term Potentiation (LTP) towards approach/action.
        Negative = Long-Term Depression (LTD) / avoidance.
        """
        return float(self.state.pam_dopamine - self.state.ppl1_stress_pain)
        
    def reset(self):
        self.state = NeuromodulatorState(octopamine_arousal=self.baseline_arousal)
