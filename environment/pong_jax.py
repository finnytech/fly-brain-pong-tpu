"""
Pure JAX Vectorized Pong Arena - Anti-Camping & Smooth Physics Edition
Features:
- Anti-Idle Reward: No passive camping rewards. Active interception rewarded (+2.5), idling penalized.
- Wide Angle Trajectories: Requires active vertical tracking; stationary paddles will miss.
- Smooth Respawn Delay: Eliminates instant teleporting/vanishing ball behavior.
"""
from typing import NamedTuple, Tuple
import numpy as np

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
    respawn_timer: any
    done: any

# Competitive Dimensions & Kinematics
PADDLE_H = 0.22         # Reduced from 0.35 to require real aiming & tracking
PADDLE_SPEED = 0.055
BALL_SPEED = 0.042
TOP_WALL = 0.95
BOTTOM_WALL = -0.95
PADDLE1_X = -0.85
PADDLE2_X = 0.85
RESPAWN_DELAY_FRAMES = 15  # Smooth serve pause (no teleporting)

def reset_single(key) -> Tuple[PongState, any, any]:
    """Resets a single environment with randomized serve angle."""
    if HAS_JAX:
        k1, k2, k3 = random.split(key if key is not None else random.PRNGKey(42), 3)
        ball_y = random.uniform(k1, minval=-0.5, maxval=0.5)
        dir_sign = jnp.where(random.bernoulli(k2, p=0.5), 1.0, -1.0)
        # Wide serve angle (-55° to +55°)
        angle = random.uniform(k3, minval=-0.95, maxval=0.95)
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
            respawn_timer=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False, dtype=jnp.bool_)
        )
    else:
        ball_y = float(np.random.uniform(-0.5, 0.5))
        dir_sign = float(np.random.choice([1.0, -1.0]))
        angle = float(np.random.uniform(-0.95, 0.95))
        vx = dir_sign * BALL_SPEED * np.cos(angle)
        vy = BALL_SPEED * np.sin(angle)
        state = PongState(
            ball_x=0.0, ball_y=ball_y, ball_vx=vx, ball_vy=vy,
            paddle1_y=0.0, paddle2_y=0.0, score1=0, score2=0, rally=0,
            respawn_timer=0, done=False
        )
        
    obs1 = get_obs(state, is_fly1=True)
    obs2 = get_obs(state, is_fly1=False)
    return state, obs1, obs2

def get_obs(state: PongState, is_fly1: bool):
    """
    Extracts egocentric sensory representation for the fly connectome:
    [rel_ball_x, rel_ball_y, ball_vx, ball_vy, own_paddle_y, opp_paddle_y, tracking_dist]
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
    Executes single tick.
    Anti-camping: Doing nothing (STILL) earns NO dopamine.
    Smooth respawn: Holds ball at center during respawn delay rather than teleporting.
    """
    xp = jnp if HAS_JAX else np
    
    # Check if currently in respawn hold
    in_respawn = state.respawn_timer > 0
    new_respawn = xp.maximum(0, state.respawn_timer - 1)
    
    # 1. Update Paddle 1
    move1 = xp.where(action1 == 0, PADDLE_SPEED, xp.where(action1 == 2, -PADDLE_SPEED, 0.0))
    p1_y = xp.clip(state.paddle1_y + move1, -1.0 + PADDLE_H / 2, 1.0 - PADDLE_H / 2)
    
    # 2. Update Paddle 2
    move2 = xp.where(action2 == 0, PADDLE_SPEED, xp.where(action2 == 2, -PADDLE_SPEED, 0.0))
    p2_y = xp.clip(state.paddle2_y + move2, -1.0 + PADDLE_H / 2, 1.0 - PADDLE_H / 2)
    
    # 3. Update Ball Position (Frozen during respawn hold)
    bx = xp.where(in_respawn, 0.0, state.ball_x + state.ball_vx)
    by = xp.where(in_respawn, 0.0, state.ball_y + state.ball_vy)
    bvx = state.ball_vx
    bvy = state.ball_vy
    
    # 4. Top/Bottom Wall Collisions
    hit_top = by >= TOP_WALL
    hit_bottom = by <= BOTTOM_WALL
    by = xp.where(hit_top, TOP_WALL, xp.where(hit_bottom, BOTTOM_WALL, by))
    bvy = xp.where(hit_top | hit_bottom, -bvy, bvy)
    
    # 5. Paddle 1 Collision Check (Left)
    p1_dist = xp.abs(by - p1_y)
    p1_hit = (~in_respawn) & (bx <= PADDLE1_X) & (bvx < 0) & (p1_dist <= (PADDLE_H / 2 + 0.04))
    bx = xp.where(p1_hit, PADDLE1_X, bx)
    # Velocity reflection with angular deflection based on hit position
    bvx = xp.where(p1_hit, xp.abs(bvx) * 1.03, bvx)
    bvy = xp.where(p1_hit, bvy + (by - p1_y) * 0.15, bvy)
    
    # 6. Paddle 2 Collision Check (Right)
    p2_dist = xp.abs(by - p2_y)
    p2_hit = (~in_respawn) & (bx >= PADDLE2_X) & (bvx > 0) & (p2_dist <= (PADDLE_H / 2 + 0.04))
    bx = xp.where(p2_hit, PADDLE2_X, bx)
    bvx = xp.where(p2_hit, -xp.abs(bvx) * 1.03, bvx)
    bvy = xp.where(p2_hit, bvy + (by - p2_y) * 0.15, bvy)
    
    # 7. Goals / Out of bounds
    goal_fly1 = (~in_respawn) & (bx > 1.02)   # Fly 1 scores (Fly 2 missed)
    goal_fly2 = (~in_respawn) & (bx < -1.02)  # Fly 2 scores (Fly 1 missed)
    point_scored = goal_fly1 | goal_fly2
    
    # 8. ANTI-CAMPING REINFORCEMENT LEARNING SHAPING
    # Active Movement Bonus: Give reward ONLY if the fly actively moves towards the ball
    ball_above_p1 = by > p1_y
    ball_below_p1 = by < p1_y
    p1_moving_correctly = (bvx < 0) & ((ball_above_p1 & (action1 == 0)) | (ball_below_p1 & (action1 == 2)))
    p1_idle_camping = (bvx < 0) & (action1 == 1)
    
    ball_above_p2 = by > p2_y
    ball_below_p2 = by < p2_y
    p2_moving_correctly = (bvx > 0) & ((ball_above_p2 & (action2 == 0)) | (ball_below_p2 & (action2 == 2)))
    p2_idle_camping = (bvx > 0) & (action2 == 1)
    
    # Reward 1 (Fly 1):
    # - BIG REWARD on successful deflection (+2.5)
    # - HUGE REWARD on point won (+5.0)
    # - SEVERE PAIN/STRESS on goal conceded (-5.0)
    # - Active tracking bonus: +0.03
    # - Camping/Idling penalty: -0.02 (Flies must NOT stay still!)
    r1 = (
        xp.where(p1_hit, 2.5, 0.0) +
        xp.where(goal_fly1, 5.0, 0.0) +
        xp.where(goal_fly2, -5.0, 0.0) +
        xp.where(p1_moving_correctly, 0.03, 0.0) +
        xp.where(p1_idle_camping, -0.02, 0.0)
    )
    
    # Reward 2 (Fly 2):
    r2 = (
        xp.where(p2_hit, 2.5, 0.0) +
        xp.where(goal_fly2, 5.0, 0.0) +
        xp.where(goal_fly1, -5.0, 0.0) +
        xp.where(p2_moving_correctly, 0.03, 0.0) +
        xp.where(p2_idle_camping, -0.02, 0.0)
    )
    
    # Update Scores and Rally
    rally = xp.where(p1_hit | p2_hit, state.rally + 1, state.rally)
    score1 = xp.where(goal_fly1, state.score1 + 1, state.score1)
    score2 = xp.where(goal_fly2, state.score2 + 1, state.score2)
    
    # Handle smooth serve respawn (freeze ball in center for delay frames, then launch)
    timer_after_goal = xp.where(point_scored, RESPAWN_DELAY_FRAMES, new_respawn)
    bx = xp.where(point_scored, 0.0, bx)
    by = xp.where(point_scored, 0.0, by)
    
    # New serve angle when respawn timer finishes
    serve_dir = xp.where(goal_fly1, -1.0, 1.0)
    bvx = xp.where(point_scored, serve_dir * BALL_SPEED, bvx)
    bvy = xp.where(point_scored, 0.02, bvy)
    
    next_state = PongState(
        ball_x=bx, ball_y=by, ball_vx=bvx, ball_vy=bvy,
        paddle1_y=p1_y, paddle2_y=p2_y,
        score1=score1, score2=score2,
        rally=xp.where(point_scored, 0, rally),
        respawn_timer=timer_after_goal,
        done=point_scored
    )
    
    obs1 = get_obs(next_state, is_fly1=True)
    obs2 = get_obs(next_state, is_fly1=False)
    
    return next_state, obs1, obs2, r1, r2, point_scored
