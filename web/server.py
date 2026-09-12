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
            if hasattr(trainer_instance, "step_visual_match"):
                step_res = trainer_instance.step_visual_match()
                env_s = getattr(trainer_instance, 'env_state', getattr(trainer_instance, 'ref_env', None))
                metrics = trainer_instance.get_live_metrics()
                payload = {
                    "ball": {
                        "x": float(env_s.ball_x),
                        "y": float(env_s.ball_y),
                        "vx": float(env_s.ball_vx),
                        "vy": float(env_s.ball_vy),
                    },
                    "paddle1_y": float(env_s.paddle1_y),
                    "paddle2_y": float(env_s.paddle2_y),
                    "score1": int(env_s.score1),
                    "score2": int(env_s.score2),
                    "rally": int(env_s.rally),
                    "steps": metrics["steps"],
                    "fps": metrics["fps"],
                    "device": metrics["device"],
                    "timer": metrics["checkpoint_timer"],
                    "fly1": metrics["fly1"],
                    "fly2": metrics["fly2"],
                    "events": {
                        "fly1_hit": float(step_res.get("reward1", 0.0)) > 1.0,
                        "fly2_hit": float(step_res.get("reward2", 0.0)) > 1.0,
                        "fly1_point_won": False,
                        "fly2_point_won": False,
                    }
                }
            else:
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
    Reads the real generated public URL and prints it prominently in the console.
    """
    import re
    import sys
    import threading

    def monitor_tunnel_output(proc):
        url_found = False
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            if "trycloudflare.com" in line:
                match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                if match and not url_found:
                    url = match.group(0)
                    url_found = True
                    print("\n" + "=" * 70, flush=True)
                    print(f"🎉 DEIN ECHTER ÖFFENTLICHER LINK IST BEREIT:", flush=True)
                    print(f"👉👉 {url} 👈👈", flush=True)
                    print("=" * 70 + "\n", flush=True)
                    try:
                        with open("live_url.txt", "w") as f:
                            f.write(url)
                    except Exception:
                        pass
            elif "localtunnel.me" in line:
                print(f"\n👉 Localtunnel URL: {line.strip()}", flush=True)

    print(f"\n[Tunnel] Initializing safe ephemeral public web host for port {port}...", flush=True)
    
    # 1. Try Cloudflared
    if shutil.which("cloudflared"):
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        t = threading.Thread(target=monitor_tunnel_output, args=(proc,), daemon=True)
        t.start()
        return proc
        
    # 2. Try Localtunnel via npx
    if shutil.which("npx"):
        proc = subprocess.Popen(
            ["npx", "-y", "localtunnel", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        t = threading.Thread(target=monitor_tunnel_output, args=(proc,), daemon=True)
        t.start()
        return proc

    print("ℹ️ Note: No tunnel tool found (install cloudflared or npx for public links).", flush=True)
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
