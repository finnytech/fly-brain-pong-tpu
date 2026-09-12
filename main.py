"""
Main Runner: Virtual Drosophila Connectome Pong RL on TPU v6e-1 / GPU
Authentic JAX PPO Deep Reinforcement Learning with FlyWire Connectome Matrix
"""
import sys
import argparse

from training.tpu_ppo_trainer import RealTpuPpoTrainer
from web.server import run_server

def main():
    parser = argparse.ArgumentParser(description="Drosophila Connectome Pong on TPU v6e-1")
    parser.add_argument("--port", type=int, default=8000, 
                        help="Port for web server (default: 8000)")
    parser.add_argument("--public", action="store_true", default=True, 
                        help="Enable safe short-use public tunnel (default: True)")
    parser.add_argument("--checkpoint-interval", type=float, default=20.0, 
                        help="Autosave interval in minutes (default: 20.0)")
    parser.add_argument("--lr", type=float, default=3e-4, 
                        help="PPO AdamW learning rate (default: 3e-4)")
    parser.add_argument("--headless", action="store_true", default=False, 
                        help="Run in headless benchmark mode without web interface")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🪰 AUTHENTIC DROSOPHILA CONNECTOME PONG - TPU v6e-1 / GPU ENGINE")
    print("Connectome Data: FlyWire Consortium (Nature 2024 Whole-Brain Connectome)")
    print("Algorithm: Pure JAX Actor-Critic PPO with GAE & Optax AdamW")
    print("Neuromodulation: PAM Dopamine (Appetitive) vs PPL1 Stress & Pain (Aversive)")
    print(f"Autosave: Every {args.checkpoint_interval} mins to safetensors")
    print("=" * 70)
    
    # Initialize authentic PPO Trainer
    trainer = RealTpuPpoTrainer(
        num_envs=64,
        rollout_len=32,
        lr=args.lr,
        checkpoint_interval_minutes=args.checkpoint_interval
    )
    
    # Start high-speed parallel PPO training loop in background thread
    trainer.start_background()
    
    if args.headless:
        print("[Mode] Running headless PPO benchmark on TPU/GPU...")
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Stopped] Gracefully exiting...")
    else:
        # Start web server streaming 60 FPS live reference match & real PPO metrics
        run_server(trainer, host="0.0.0.0", port=args.port, public=args.public)

if __name__ == "__main__":
    main()
