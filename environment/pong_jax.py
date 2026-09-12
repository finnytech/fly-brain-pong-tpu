"""
Pure JAX Vectorized Pong Arena for High-Speed TPU v6e-1 / GPU Training
Executes hundreds of parallel matches natively inside TPU XLA kernels via jax.vmap.
"""
from typing import NamedTuple, Tuple
import numpy as np

# Try importing JAX, with fallback to NumPy if not in TPU/GPU environment yet
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np

class PongState(NamedTuple):
    ball_x: any
    ball_y: any
    ball_vx: any
    ball_vy: any
    paddle1_y: any
    paddle2_y: any
    score1: any
    score2: any
    rally: any
    done: any

# Constants
PADDLE_H = 0.35
PADDLE_SPEED = 0.05
BALL_SPEED = 0.045
TOP_WALL = 0.95
BOTTOM_WALL = -0.95
PADDLE1_X = -0.85
PADDLE2_X = 0.85

def reset_single(key) -> Tuple[PongState, any, any]:
    """Resets a single environment instance."""
    if HAS_JAX:
        k1, k2, k3 = random.split(key, 3)
        ball_y = random.uniform(k1, minval=-0.3, maxval=0.3)
        dir_sign = jnp.where(random.bernoulli(k2, p=0.5), 1.0, -1.0)
        angle = random.uniform(k3, minval=-jnp.pi / 4, maxval=jnp.pi / 4)
        vx = dir_sign * BALL_SPEED * jnp.cos(angle)
        vy = BALL_SPEED * jnp.sin(angle)
        
        state = PongState(
            ball_x=jnp.array(0.0, dtype=jnp.float32),
            ball_y=jnp.array(ball_y, dtype=jnp.float32),
            ball_vx=jnp.array(vx, dtype=jnp.float32),
            ball_vy=jnp.array(vy, dtype=jnp.float32),
            paddle1_y=jnp.array(0.0, dtype=jnp.float32),
            paddle2_y=jnp.array(0.0, dtype=jnp.float32),
            score1=jnp.array(0, dtype=jnp.int32),
            score2=jnp.array(0, dtype=jnp.int32),
            rally=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False, dtype=jnp.bool_)
        )
    else:
        ball_y = float(np.random.uniform(-0.3, 0.3))
        dir_sign = float(np.random.choice([1.0, -1.0]))
        angle = float(np.random.uniform(-np.pi / 4, np.pi / 4))
        vx = dir_sign * BALL_SPEED * np.cos(angle)
        vy = BALL_SPEED * np.sin(angle)
        state = PongState(
            ball_x=0.0, ball_y=ball_y, ball_vx=vx, ball_vy=vy,
            paddle1_y=0.0, paddle2_y=0.0, score1=0, score2=0, rally=0, done=False
        )
        
    obs1 = get_obs(state, is_fly1=True)
    obs2 = get_obs(state, is_fly1=False)
    return state, obs1, obs2

def get_obs(state: PongState, is_fly1: bool):
    """
    Extracts egocentric sensory representation for the fly connectome:
    [rel_ball_x, rel_ball_y, ball_vx, ball_vy, own_paddle_y, opp_paddle_y, tracking_error]
    """
    xp = jnp if HAS_JAX else np
    if is_fly1:
        rel_bx = state.ball_x - PADDLE1_X
        rel_by = state.ball_y - state.paddle1_y
        bvx = state.ball_vx
        bvy = state.ball_vy
        own_py = state.paddle1_y
        opp_py = state.paddle2_y
    else:
        # Flip coordinates for Fly 2 (symmetric egocentric view)
        rel_bx = PADDLE2_X - state.ball_x
        rel_by = state.ball_y - state.paddle2_y
        bvx = -state.ball_vx
        bvy = state.ball_vy
        own_py = state.paddle2_y
        opp_py = state.paddle1_y
        
    tracking_dist = xp.abs(rel_by)
    return xp.array([rel_bx, rel_by, bvx, bvy, own_py, opp_py, tracking_dist], dtype=xp.float32)

def step_single(state: PongState, action1: int, action2: int, key) -> Tuple[PongState, any, any, any, any, any]:
    """
    Single environment physics tick compiled natively on TPU/GPU.
    Actions: 0: UP, 1: STILL, 2: DOWN
    Returns: (next_state, obs1, obs2, dopamine_reward1, dopamine_reward2, done)
    """
    xp = jnp if HAS_JAX else np
    
    # 1. Update Paddle 1
    move1 = xp.where(action1 == 0, PADDLE_SPEED, xp.where(action1 == 2, -PADDLE_SPEED, 0.0))
    p1_y = xp.clip(state.paddle1_y + move1, -1.0 + PADDLE_H / 2, 1.0 - PADDLE_H / 2)
    
    # 2. Update Paddle 2
    move2 = xp.where(action2 == 0, PADDLE_SPEED, xp.where(action2 == 2, -PADDLE_SPEED, 0.0))
    p2_y = xp.clip(state.paddle2_y + move2, -1.0 + PADDLE_H / 2, 1.0 - PADDLE_H / 2)
    
    # 3. Update Ball Position
    bx = state.ball_x + state.ball_vx
    by = state.ball_y + state.ball_vy
    bvx = state.ball_vx
    bvy = state.ball_vy
    
    # 4. Top/Bottom Wall Collisions
    hit_top = by >= TOP_WALL
    hit_bottom = by <= BOTTOM_WALL
    by = xp.where(hit_top, TOP_WALL, xp.where(hit_bottom, BOTTOM_WALL, by))
    bvy = xp.where(hit_top | hit_bottom, -bvy, bvy)
    
    # 5. Paddle 1 Collision Check (Left)
    p1_dist = xp.abs(by - p1_y)
    p1_hit = (bx <= PADDLE1_X) & (state.ball_vx < 0) & (p1_dist <= (PADDLE_H / 2 + 0.05))
    bx = xp.where(p1_hit, PADDLE1_X, bx)
    bvx = xp.where(p1_hit, xp.abs(bvx) * 1.02, bvx)
    # Spin angle based on hit location
    bvy = xp.where(p1_hit, bvy + (by - p1_y) * 0.1, bvy)
    
    # 6. Paddle 2 Collision Check (Right)
    p2_dist = xp.abs(by - p2_y)
    p2_hit = (bx >= PADDLE2_X) & (state.ball_vx > 0) & (p2_dist <= (PADDLE_H / 2 + 0.05))
    bx = xp.where(p2_hit, PADDLE2_X, bx)
    bvx = xp.where(p2_hit, -xp.abs(bvx) * 1.02, bvx)
    bvy = xp.where(p2_hit, bvy + (by - p2_y) * 0.1, bvy)
    
    # 7. Goals / Out of bounds
    goal_fly1 = bx > 1.05   # Fly 1 scores
    goal_fly2 = bx < -1.05  # Fly 2 scores
    
    # 8. Genuine Dopamine PAM vs PPL1 Stress Reward Shaping
    # Reward 1:
    # +1.5 on hit (PAM burst), +5.0 on goal won
    # -1.5 on missed return (PPL1 stress), -5.0 on goal lost
    # + dense tracking guidance: +0.02 * (1.0 - tracking_dist)
    r1 = (
        xp.where(p1_hit, 1.5, 0.0) +
        xp.where(goal_fly1, 5.0, 0.0) +
        xp.where(goal_fly2, -5.0, 0.0) +
        0.02 * (1.0 - xp.clip(p1_dist, 0.0, 1.0))
    )
    
    # Reward 2:
    r2 = (
        xp.where(p2_hit, 1.5, 0.0) +
        xp.where(goal_fly2, 5.0, 0.0) +
        xp.where(goal_fly1, -5.0, 0.0) +
        0.02 * (1.0 - xp.clip(p2_dist, 0.0, 1.0))
    )
    
    # Update Scores and Rally
    rally = xp.where(p1_hit | p2_hit, state.rally + 1, state.rally)
    score1 = xp.where(goal_fly1, state.score1 + 1, state.score1)
    score2 = xp.where(goal_fly2, state.score2 + 1, state.score2)
    
    # Reset ball on goal
    point_scored = goal_fly1 | goal_fly2
    bx = xp.where(point_scored, 0.0, bx)
    by = xp.where(point_scored, 0.0, by)
    bvx = xp.where(goal_fly1, -BALL_SPEED, xp.where(goal_fly2, BALL_SPEED, bvx))
    
    next_state = PongState(
        ball_x=bx, ball_y=by, ball_vx=bvx, ball_vy=bvy,
        paddle1_y=p1_y, paddle2_y=p2_y,
        score1=score1, score2=score2,
        rally=xp.where(point_scored, 0, rally),
        done=point_scored
    )
    
    obs1 = get_obs(next_state, is_fly1=True)
    obs2 = get_obs(next_state, is_fly1=False)
    
    return next_state, obs1, obs2, r1, r2, point_scored
