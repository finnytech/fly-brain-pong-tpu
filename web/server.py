"""
FastAPI & High-Speed WebSocket Server for Drosophila Connectome Pong
Streams 60 FPS game physics, 3D connectome spikes, dopamine/stress states,
and provides public short-use tunnel capabilities.
"""
import os
import asyncio
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from training.trainer_tpu import TpuFlyPongTrainer

app = FastAPI(title="Drosophila Connectome Pong Lab")

# References set on launch
trainer_instance: Optional[TpuFlyPongTrainer] = None
STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def get_index():
    """Serves the main HTML5/Three.js Cyberpunk Dashboard."""
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/api/save")
async def manual_save():
    """Triggers an immediate checkpoint save."""
    global trainer_instance
    if not trainer_instance:
        return JSONResponse({"status": "error", "message": "Trainer not initialized"}, status_code=500)
    
    path = trainer_instance.checkpointer.save_checkpoint(
        trainer_instance.fly1, trainer_instance.fly2,
        step=trainer_instance.step_counter,
        score1=trainer_instance.env.score1,
        score2=trainer_instance.env.score2
    )
    return {"status": "ok", "saved_path": path}

@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket):
    """
    Ultra-low-latency 60 FPS WebSocket stream.
    Delivers continuous physics frames, fly wing speeds, 3D brain activations,
    PAM dopamine and PPL1 stress/pain telemetry.
    """
    await websocket.accept()
    global trainer_instance
    if not trainer_instance:
        await websocket.close()
        return

    try:
        while True:
            # Step environment forward if not running in a separate thread
            if not trainer_instance.is_running:
                step_res = trainer_instance.step_once()
                events = step_res["events"]
            else:
                events = None

            metrics = trainer_instance.get_live_metrics()
            env = trainer_instance.env

            payload = {
                "ball": {
                    "x": env.ball_x,
                    "y": env.ball_y,
                    "vx": env.ball_vx,
                    "vy": env.ball_vy,
                },
                "paddle1_y": env.paddle1_y,
                "paddle2_y": env.paddle2_y,
                "score1": env.score1,
                "score2": env.score2,
                "rally": env.rally,
                "steps": metrics["steps"],
                "fps": metrics["fps"],
                "device": metrics["device"],
                "timer": metrics["checkpoint_timer"],
                "fly1": metrics["fly1"],
                "fly2": metrics["fly2"],
                "events": {
                    "fly1_hit": events.fly1_hit if events else False,
                    "fly2_hit": events.fly2_hit if events else False,
                    "fly1_point_won": events.fly1_point_won if events else False,
                    "fly2_point_won": events.fly2_point_won if events else False,
                }
            }

            await websocket.send_json(payload)
            # ~60 FPS target (16 ms)
            await asyncio.sleep(0.016)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Error] {e}")

def start_public_tunnel(port: int = 8000) -> Optional[subprocess.Popen]:
    """
    Launches a safe, short-use public tunnel (Localtunnel / Cloudflared).
    Useful for Colab or quick remote inspection.
    """
    print(f"\n[Tunnel] Initializing safe ephemeral public web host for port {port}...")
    
    # 1. Try Cloudflared
    if shutil.which("cloudflared"):
        proc = subprocess.Popen(["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print("⚡ Cloudflare Tunnel started! Watch terminal for https://xxxx.trycloudflare.com URL.")
        return proc
        
    # 2. Try Localtunnel via npx
    if shutil.which("npx"):
        proc = subprocess.Popen(["npx", "-y", "localtunnel", "--port", str(port)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print("⚡ Localtunnel started via npx! Watch terminal for public URL.")
        return proc

    print("ℹ️ Note: For public viewing in Google Colab, use:")
    print(f"   !npx localtunnel --port {port}")
    print(f"   Or: from google.colab.output import serve_kernel_port_as_window; serve_kernel_port_as_window({port})")
    return None

def run_server(trainer: TpuFlyPongTrainer, host: str = "0.0.0.0", port: int = 8000, public: bool = False):
    """Starts the FastAPI web server."""
    global trainer_instance
    trainer_instance = trainer

    if public:
        start_public_tunnel(port)

    print(f"\n==================================================================")
    print(f"🚀 DROSOPHILA PONG WEB DASHBOARD IS LIVE:")
    print(f"👉 Local URL: http://localhost:{port}")
    print(f"👉 Public Tunnel: {'Enabled (see above)' if public else 'Disabled (run with --public to enable)'}")
    print(f"==================================================================\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")
