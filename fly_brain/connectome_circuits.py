"""
Drosophila Connectome Circuit Architecture
Faithfully modeling the biological connectivity of the fruit fly brain:
1. Optic Lobe (Ommatidia -> Lamina L1/L2 -> Medulla Mi1/Tm3 -> Lobula Plate T4/T5 & LPTC HS/VS)
2. Central Complex (Ellipsoid Body Ring Neurons, Protocerebral Bridge, EPG Compass, P-EN Steering)
3. Mushroom Body (Sparse Kenyon Cells, PAM/PPL1 Neuromodulation, MBONs)
4. Descending Neurons (DNp01, DNa02 motor drives)
"""

import numpy as np

class DrosophilaConnectome:
    """
    Bio-realistic Drosophila connectome model.
    Connects biological neural layers with synaptic weight matrices
    derived from whole-brain connectome topology.
    """
    def __init__(self, visual_dim: int = 16, num_kcs: int = 256, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.visual_dim = visual_dim
        self.num_ommatidia = visual_dim * visual_dim
        self.num_kcs = num_kcs
        
        # 1. OPTIC LOBE SYNAPSE MATRICES
        # Lamina (L1=ON, L2=OFF channels)
        self.w_ommatidia_to_lamina_on = self.rng.normal(1.2, 0.1, size=(self.num_ommatidia,))
        self.w_ommatidia_to_lamina_off = self.rng.normal(-1.2, 0.1, size=(self.num_ommatidia,))
        
        # Medulla (Mi1, Tm3)
        self.w_lamina_to_medulla = self.rng.uniform(0.5, 1.5, size=(self.num_ommatidia, 32))
        
        # Lobula Plate (T4: ON motion, T5: OFF motion, 4 directions: Up, Down, Left, Right)
        self.w_medulla_to_t4t5 = self.rng.uniform(-1.0, 1.0, size=(32, 16))
        
        # LPTC (Lobula Plate Tangential Cells: HS = Horizontal System, VS = Vertical System)
        # 4 HS neurons, 6 VS neurons
        self.w_t4t5_to_lptc = self.rng.normal(0.0, 0.5, size=(16, 10))
        
        # 2. CENTRAL COMPLEX (CX)
        # Ring Neurons (ER, 16 neurons) encode ball position & movement in egocentric space
        self.w_lptc_to_ring = self.rng.normal(0.0, 0.4, size=(10, 16))
        
        # Protocerebral Bridge & EPG Compass Ring Attractor (16 EPG neurons)
        self.w_ring_to_epg = self.rng.normal(0.0, 0.3, size=(16, 16))
        
        # Biological Ring Attractor Recurrent Weights (Mexican hat connectivity for compass bump)
        theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        diff = np.abs(theta[:, None] - theta[None, :])
        diff = np.minimum(diff, 2 * np.pi - diff)
        self.w_epg_recurrent = 1.8 * np.exp(-(diff**2) / (2 * 0.6**2)) - 1.2
        
        # P-EN Steering Neurons (8 pairs = 16 neurons)
        self.w_epg_to_pen = self.rng.normal(0.0, 0.4, size=(16, 16))
        
        # 3. MUSHROOM BODY (Learning & Associative Plasticity)
        # Kenyon Cells (KC, ~256 modeled): High-dimensional sparse coding (~5-10% active)
        # In Drosophila, each KC receives random claws from ~6-8 projection inputs
        self.w_cx_to_kc = np.zeros((32, self.num_kcs))
        for k in range(self.num_kcs):
            inputs = self.rng.choice(32, size=7, replace=False)
            self.w_cx_to_kc[inputs, k] = self.rng.uniform(0.8, 1.4, size=7)
            
        # Mushroom Body Output Neurons (MBONs, 8 compartments)
        # These are the PLASTIC synapses modulated by PAM (Dopamine) and PPL1 (Stress/Pain)
        self.w_kc_to_mbon = self.rng.normal(0.0, 0.05, size=(self.num_kcs, 8))
        
        # 4. DESCENDING NEURONS (DNp01, DNa02 -> Paddle Action: [UP, STAY, DOWN])
        self.w_mbon_to_dn = self.rng.normal(0.0, 0.2, size=(8, 3))
        self.w_cx_to_dn = self.rng.normal(0.0, 0.2, size=(16, 3)) # Direct innate steering reflex
        
        # Dynamic State Variables
        self.prev_visual_frame = np.zeros(self.num_ommatidia)
        self.epg_activity = np.ones(16) * 0.1
        self.latest_spike_records = {}

    def forward(self, visual_frame: np.ndarray, proprioception: np.ndarray) -> np.ndarray:
        """
        Runs one biological forward cycle of the virtual Drosophila brain.
        
        Args:
            visual_frame: 1D or 2D array of ommatidial intensities (flattened to num_ommatidia).
            proprioception: Current paddle vertical position and velocity.
            
        Returns:
            action_logits: Unnormalized logits for [UP, STILL, DOWN].
        """
        ommatidia = visual_frame.flatten()
        if ommatidia.shape[0] != self.num_ommatidia:
            ommatidia = np.resize(ommatidia, (self.num_ommatidia,))
            
        # --- 1. OPTIC LOBE ---
        # Temporal contrast (motion)
        delta_vision = ommatidia - self.prev_visual_frame
        self.prev_visual_frame = ommatidia.copy()
        
        # Lamina ON/OFF separation
        lamina_on = np.maximum(0.0, delta_vision * self.w_ommatidia_to_lamina_on)
        lamina_off = np.maximum(0.0, -delta_vision * self.w_ommatidia_to_lamina_off)
        lamina = lamina_on + lamina_off
        
        # Medulla
        medulla = np.tanh(lamina @ self.w_lamina_to_medulla)
        
        # Lobula Plate T4/T5 Direction-selective cells
        t4t5 = np.maximum(0.0, medulla @ self.w_medulla_to_t4t5)
        
        # LPTC (HS horizontal and VS vertical motion integration)
        lptc = np.tanh(t4t5 @ self.w_t4t5_to_lptc)
        
        # --- 2. CENTRAL COMPLEX (CX) ---
        # Ring Neurons integrate motion + proprioceptive paddle position
        ring_input = lptc @ self.w_lptc_to_ring
        ring_neurons = np.tanh(ring_input)
        
        # EPG Compass Neurons: Ring attractor dynamics
        epg_drive = ring_neurons @ self.w_ring_to_epg + self.epg_activity @ self.w_epg_recurrent
        self.epg_activity = np.maximum(0.0, np.tanh(epg_drive))
        # Normalize to maintain single compass activity bump
        if np.sum(self.epg_activity) > 1e-5:
            self.epg_activity /= np.sum(self.epg_activity)
            
        # P-EN Steering Neurons
        pen_neurons = np.tanh(self.epg_activity @ self.w_epg_to_pen)
        
        # Combined Central Complex Representation (32 dims)
        cx_state = np.concatenate([ring_neurons, pen_neurons])
        
        # --- 3. MUSHROOM BODY (Sparse Kenyon Cells) ---
        kc_input = cx_state @ self.w_cx_to_kc
        # Biological Winner-Take-All / Sparsity threshold (~7% active)
        kc_threshold = np.percentile(kc_input, 93)
        kc_spikes = np.where(kc_input >= kc_threshold, np.tanh(kc_input), 0.0)
        
        # MBONs (Mushroom Body Output Neurons)
        mbon_activity = np.tanh(kc_spikes @ self.w_kc_to_mbon)
        
        # --- 4. DESCENDING MOTOR NEURONS (DNs) ---
        dn_logits = (mbon_activity @ self.w_mbon_to_dn) + (pen_neurons @ self.w_cx_to_dn)
        
        # Record neuron firing states for real-time web visualization
        self.latest_spike_records = {
            "lamina_mean": float(np.mean(lamina)),
            "t4t5_activity": float(np.mean(t4t5)),
            "lptc_vs_vertical": float(np.mean(np.abs(lptc[:6]))),
            "lptc_hs_horizontal": float(np.mean(np.abs(lptc[6:]))),
            "epg_bump_heading": float(np.argmax(self.epg_activity)),
            "kc_active_count": int(np.count_nonzero(kc_spikes)),
            "mbon_mean": float(np.mean(np.abs(mbon_activity))),
            "dn_drive": dn_logits.tolist()
        }
        
        return dn_logits, kc_spikes, mbon_activity

    def apply_dopamine_plasticity(self, kc_spikes: np.ndarray, mbon_activity: np.ndarray, 
                                  net_reinforcement: float, lr: float = 0.01):
        """
        Three-Factor Neuromodulated Synaptic Plasticity:
        dW = lr * (PAM_Dopamine - PPL1_Stress) * Presynaptic_KC * Postsynaptic_MBON
        
        - If Dopamine (PAM) > Stress (PPL1): Potentiates winning action association.
        - If Stress/Pain (PPL1) > Dopamine (PAM): Depresses missed/failed action association.
        """
        if abs(net_reinforcement) < 1e-4:
            return
            
        dw = lr * net_reinforcement * np.outer(kc_spikes, mbon_activity)
        # Biological weight bounds [-1.0, 1.0]
        self.w_kc_to_mbon = np.clip(self.w_kc_to_mbon + dw, -1.0, 1.0)
        
        # Modulate direct descending reflex with dopamine trace
        if abs(net_reinforcement) > 0.5:
            self.w_mbon_to_dn += lr * 0.2 * net_reinforcement * np.sign(self.w_mbon_to_dn)
            self.w_mbon_to_dn = np.clip(self.w_mbon_to_dn, -1.0, 1.0)
