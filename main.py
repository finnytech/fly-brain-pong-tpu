"""
Main Runner: Virtual Drosophila Connectome Pong RL on TPU v5e
FastAPI + WebSocket 60 FPS Cyberpunk Web-App with 3D Brains & Animated Flies
"""
import sys
import argparse

from training.trainer_tpu import TpuFlyPongTrainer
from web.server import run_server

def main():
    parser = argparse.ArgumentParser(description="Drosophila Connectome Pong on TPU v5e")
    parser.add_argument("--port", type=int, default=8000, 
                        help="Port for web server (default: 8000)")
    parser.add_argument("--public", action="store_true", default=True, 
                        help="Enable safe short-use public tunnel (default: True)")
    parser.add_argument("--checkpoint-interval", type=float, default=20.0, 
                        help="Autosave interval in minutes (default: 20.0)")
    parser.add_argument("--lr", type=float, default=0.02, 
                        help="Plasticity learning rate (default: 0.02)")
    parser.add_argument("--headless", action="store_true", default=False, 
                        help="Run in headless benchmark mode without web interface")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🪰 VIRTUAL DROSOPHILA CONNECTOME PONG - TPU v5e ARENA")
    print("Hardware Target: Google Colab TPU v5e (with GPU / CPU fallback)")
    print(f"20-Minute Autosave: Every {args.checkpoint_interval} mins")
    print("Biochemistry: PAM Dopamine (Happy/Win) vs PPL1 Stress & Pain (Sauer/Loss)")
    print("Frontend: Custom HTML5/CSS3/JS with Three.js 3D Brains & Animated Flies")
    print("=" * 70)
    
    trainer = TpuFlyPongTrainer(
        checkpoint_interval_minutes=args.checkpoint_interval,
        learning_rate=args.lr
    )
    
    if args.headless:
        print("[Mode] Running headless benchmark...")
        try:
            trainer.run_training_loop()
        except KeyboardInterrupt:
            print("\n[Stopped] Saving state before exit...")
            trainer.checkpointer.save_checkpoint(
                trainer.fly1, trainer.fly2, 
                step=trainer.step_counter, 
                score1=trainer.env.score1, 
                score2=trainer.env.score2
            )
    else:
        # Start web server with WebSocket 60 FPS live feed
        run_server(trainer, host="0.0.0.0", port=args.port, public=args.public)

if __name__ == "__main__":
    main()
