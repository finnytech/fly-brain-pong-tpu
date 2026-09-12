"""
Live Public Web Host & Telemetry Dashboard for Drosophila Connectome Pong
Uses Gradio with ephemeral public link (share=True) for safe, short-use streaming
of live matches, dopamine/stress gauges, and biological neural firing.
"""
import time
import gradio as gr
from PIL import Image
from typing import Tuple, Dict, Any

from training.trainer_tpu import TpuFlyPongTrainer

def create_dashboard(trainer: TpuFlyPongTrainer):
    """Constructs the Gradio live streaming UI."""
    
    with gr.Blocks(title="Drosophila Connectome Pong - TPU v5e Arena", theme=gr.themes.Monochrome()) as demo:
        gr.Markdown(
            """
            # 🪰 Drosophila Connectome: Two Fly Brains Playing Pong on TPU v5e
            ### Real-Time Biological Neural Circuit Simulation (Optic Lobe • Central Complex • Mushroom Body • PAM Dopamine vs PPL1 Stress)
            *Public short-use ephemeral live stream for Google Colab & Local Execution.*
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                court_image = gr.Image(label="Live Arena (Fly-1 Green vs Fly-2 Purple)", interactive=False)
                with gr.Row():
                    score_display = gr.Markdown("### 🏆 Score: **Fly 1: 0** | **Fly 2: 0** | Rally: 0")
                    fps_display = gr.Markdown("⚡ Speed: **0 FPS** | Device: Initializing...")
            
            with gr.Column(scale=2):
                gr.Markdown("### 🧪 Fly 1 Neuromodulators (Left - Green)")
                dopamine1 = gr.Slider(minimum=0.0, maximum=5.0, value=0.0, label="PAM Dopamine (Appetitive Reward / Win)", interactive=False)
                stress1 = gr.Slider(minimum=0.0, maximum=5.0, value=0.0, label="PPL1 Stress & Pain (Aversive / Loss)", interactive=False)
                arousal1 = gr.Slider(minimum=0.0, maximum=2.0, value=0.5, label="Octopamine (Arousal & Motor Readiness)", interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("### 🧪 Fly 2 Neuromodulators (Right - Purple)")
                dopamine2 = gr.Slider(minimum=0.0, maximum=5.0, value=0.0, label="PAM Dopamine (Appetitive Reward / Win)", interactive=False)
                stress2 = gr.Slider(minimum=0.0, maximum=5.0, value=0.0, label="PPL1 Stress & Pain (Aversive / Loss)", interactive=False)
                arousal2 = gr.Slider(minimum=0.0, maximum=2.0, value=0.5, label="Octopamine (Arousal & Motor Readiness)", interactive=False)

        with gr.Row():
            with gr.Accordion("🔬 Biological Connectome Spike Telemetry", open=True):
                with gr.Row():
                    telemetry1 = gr.JSON(label="Fly 1 Neural Firings (Optic Lobe, EPG Compass, Kenyon Cells, MBONs)")
                    telemetry2 = gr.JSON(label="Fly 2 Neural Firings (Optic Lobe, EPG Compass, Kenyon Cells, MBONs)")
                    
        with gr.Row():
            checkpoint_status = gr.Markdown("💾 20-Min Checkpoint Timer: **20:00** | Saved: 0")
            save_now_btn = gr.Button("💾 Force Save Checkpoint Now", variant="secondary")
            save_output = gr.Textbox(label="Save Log", interactive=False, max_lines=1)

        # Refresh function called by Gradio Timer
        def update_stream():
            # Step simulation if trainer loop is not run in background thread
            if not trainer.is_running:
                for _ in range(3):
                    trainer.step_once()
                    
            metrics = trainer.get_live_metrics()
            frame = trainer.env.render_display_frame()
            
            score_text = f"### 🏆 Score: **Fly 1: {metrics['score1']}** | **Fly 2: {metrics['score2']}** | Current Rally: **{metrics['rally']}**"
            speed_text = f"⚡ Speed: **{metrics['fps']} FPS** | Steps: {metrics['steps']:,} | Accelerator: `{metrics['device']}`"
            ckpt_text = f"💾 20-Min Checkpoint Timer: **{metrics['checkpoint_timer']}** | Total Saves: **{metrics['checkpoint_count']}**"
            
            f1 = metrics["fly1"]
            f2 = metrics["fly2"]
            
            return (
                frame,
                score_text,
                speed_text,
                round(f1["dopamine_pam"], 2),
                round(f1["stress_ppl1"], 2),
                round(f1["octopamine_arousal"], 2),
                round(f2["dopamine_pam"], 2),
                round(f2["stress_ppl1"], 2),
                round(f2["octopamine_arousal"], 2),
                f1,
                f2,
                ckpt_text
            )
            
        def force_save():
            res = trainer.checkpointer.save_checkpoint(
                trainer.fly1, trainer.fly2, 
                step=trainer.step_counter, 
                score1=trainer.env.score1, 
                score2=trainer.env.score2
            )
            return f"Saved manually to: {res}"

        save_now_btn.click(fn=force_save, outputs=save_output)

        # Timer updates UI smoothly
        timer = gr.Timer(value=0.1) # 100ms updates
        timer.tick(
            fn=update_stream,
            outputs=[
                court_image, score_display, fps_display,
                dopamine1, stress1, arousal1,
                dopamine2, stress2, arousal2,
                telemetry1, telemetry2,
                checkpoint_status
            ]
        )
        
    return demo

def launch_dashboard(trainer: TpuFlyPongTrainer, share: bool = True, port: int = 7860):
    """
    Launches the live dashboard with a public URL for short-use viewing.
    """
    demo = create_dashboard(trainer)
    print(f"\n[Web Host] Launching Gradio Web Host (share={share})...")
    url, local_url, share_url = demo.launch(
        server_name="0.0.0.0", 
        server_port=port, 
        share=share,
        inline=False,
        quiet=False
    )
    if share_url:
        print(f"\n=======================================================")
        print(f"🌍 PUBLIC LIVE WEB HOST (Safe / Short Use):")
        print(f"👉 {share_url}")
        print(f"=======================================================\n")
    return demo
