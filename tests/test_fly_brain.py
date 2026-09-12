"""
Verification script for virtual Drosophila connectome Pong RL
Tests connectome forward pass, dopamine/stress plasticity, Pong physics, and checkpointing.
"""
import sys
import os
sys.path.insert(0, r"D:\fly_brain_pong_tpu")

from fly_brain.fly_agent import FlyAgent
from environment.pong_vectorized import TwoFlyPongEnv
from training.checkpointer import FlyBrainCheckpointer
from training.trainer_tpu import TpuFlyPongTrainer

def test_all():
    print("[1/4] Testing FlyAgent and Drosophila Connectome...")
    fly1 = FlyAgent("Fly1_Test")
    fly2 = FlyAgent("Fly2_Test")
    
    # Test random visual frame
    dummy_vision = (os.urandom(16 * 16))
    dummy_frame = (list(dummy_vision))
    import numpy as np
    frame = np.array(dummy_frame, dtype=np.float32).reshape(16, 16) / 255.0
    
    action1 = fly1.act(frame, paddle_pos=0.0)
    action2 = fly2.act(frame, paddle_pos=0.0)
    assert action1 in [0, 1, 2], f"Invalid action1: {action1}"
    assert action2 in [0, 1, 2], f"Invalid action2: {action2}"
    print(f"  -> Forward pass OK! Chosen actions: Fly1={action1}, Fly2={action2}")
    
    print("[2/4] Testing Dopamine (PAM) and Stress (PPL1) Plasticity...")
    # Test win event for Fly 1 (Dopamine PAM) and loss for Fly 2 (Stress PPL1)
    nm1 = fly1.on_event(ball_hit=True, point_won=True, ball_missed=False, point_lost=False)
    nm2 = fly2.on_event(ball_hit=False, point_won=False, ball_missed=True, point_lost=True)
    assert nm1.pam_dopamine > 1.5, f"Expected dopamine burst in Fly1, got {nm1.pam_dopamine}"
    assert nm2.ppl1_stress_pain > 1.5, f"Expected stress/pain burst in Fly2, got {nm2.ppl1_stress_pain}"
    print(f"  -> Neuromodulation OK! Fly1 PAM Dopamine={nm1.pam_dopamine:.2f}, Fly2 PPL1 Stress={nm2.ppl1_stress_pain:.2f}")
    
    print("[3/4] Testing TwoFlyPongEnv...")
    env = TwoFlyPongEnv(visual_res=16)
    v1, v2, events = env.step(0, 2)
    assert v1.shape == (16, 16), f"Wrong vision shape: {v1.shape}"
    assert v2.shape == (16, 16), f"Wrong vision shape: {v2.shape}"
    display_img = env.render_display_frame(320, 200)
    assert display_img.size == (320, 200), f"Wrong image size: {display_img.size}"
    print("  -> Environment and ommatidia rendering OK!")
    
    print("[4/4] Testing Checkpointer & Trainer...")
    trainer = TpuFlyPongTrainer(checkpoint_interval_minutes=0.01) # short interval for test
    for _ in range(25):
        trainer.step_once()
    metrics = trainer.get_live_metrics()
    print(f"  -> Executed 25 steps successfully! Telemetry: Score {metrics['score1']}:{metrics['score2']}, Rally {metrics['rally']}")
    
    # Test forced save
    save_path = trainer.checkpointer.save_checkpoint(trainer.fly1, trainer.fly2, 25, metrics['score1'], metrics['score2'])
    assert os.path.exists(save_path), f"Checkpoint not found at {save_path}"
    print(f"  -> Checkpoint successfully created at: {save_path}")
    
    # Test reload
    reload_ok = trainer.checkpointer.load_latest(trainer.fly1, trainer.fly2)
    assert reload_ok, "Failed to reload checkpoint"
    print("  -> Checkpoint reload verified successfully!")
    
    print("\n[OK] ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_all()
