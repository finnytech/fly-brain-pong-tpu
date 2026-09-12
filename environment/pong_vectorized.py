"""
Vectorized Two-Fly Pong Environment (Drosophila Table Tennis Arena)
Simulates realistic ball physics, paddle kinematics, ommatidial compound eye projections,
and score / event tracking for reinforcement learning.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

@dataclass
class PongEvents:
    fly1_hit: bool = False
    fly1_miss: bool = False
    fly1_point_won: bool = False
    fly1_point_lost: bool = False
    
    fly2_hit: bool = False
    fly2_miss: bool = False
    fly2_point_won: bool = False
    fly2_point_lost: bool = False
    
    rally_length: int = 0

class TwoFlyPongEnv:
    """
    Continuous 2-player Pong match between Fly 1 (Left) and Fly 2 (Right).
    Coordinates: x in [-1.0, 1.0], y in [-1.0, 1.0].
    """
    def __init__(self, visual_res: int = 16, paddle_height: float = 0.22, paddle_speed: float = 0.055):
        self.visual_res = visual_res
        self.paddle_height = paddle_height
        self.paddle_speed = paddle_speed
        
        # State variables
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.03
        self.ball_vy = 0.02
        
        self.paddle1_y = 0.0
        self.paddle2_y = 0.0
        
        self.score1 = 0
        self.score2 = 0
        self.rally = 0
        self.total_rallies = 0
        self.step_count = 0
        
        self.reset_ball()
        
    def reset_ball(self, serve_to_fly1: bool = None):
        """Resets ball to center with randomized velocity angle."""
        self.ball_x = 0.0
        self.ball_y = float(np.random.uniform(-0.4, 0.4))
        
        direction = -1.0 if (serve_to_fly1 if serve_to_fly1 is not None else np.random.choice([True, False])) else 1.0
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        speed = 0.035
        self.ball_vx = float(direction * speed * np.cos(angle))
        self.ball_vy = float(speed * np.sin(angle))
        self.rally = 0

    def step(self, action1: int, action2: int) -> Tuple[np.ndarray, np.ndarray, PongEvents]:
        """
        Advances the simulation by 1 physics frame.
        Actions:
            0: UP   (y += paddle_speed)
            1: STAY (y unchanged)
            2: DOWN (y -= paddle_speed)
        """
        self.step_count += 1
        events = PongEvents()
        
        # 1. Update Paddle 1
        if action1 == 0:
            self.paddle1_y += self.paddle_speed
        elif action1 == 2:
            self.paddle1_y -= self.paddle_speed
        self.paddle1_y = float(np.clip(self.paddle1_y, -1.0 + self.paddle_height / 2, 1.0 - self.paddle_height / 2))
        
        # 2. Update Paddle 2
        if action2 == 0:
            self.paddle2_y += self.paddle_speed
        elif action2 == 2:
            self.paddle2_y -= self.paddle_speed
        self.paddle2_y = float(np.clip(self.paddle2_y, -1.0 + self.paddle_height / 2, 1.0 - self.paddle_height / 2))
        
        # 3. Update Ball Position
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # 4. Top and Bottom Wall Collisions
        if self.ball_y >= 0.95:
            self.ball_y = 0.95
            self.ball_vy = -abs(self.ball_vy)
        elif self.ball_y <= -0.95:
            self.ball_y = -0.95
            self.ball_vy = abs(self.ball_vy)
            
        # 5. Paddle 1 (Left, x = -0.85) Collision Check
        if self.ball_x <= -0.85 and self.ball_vx < 0:
            if abs(self.ball_y - self.paddle1_y) <= self.paddle_height / 2 + 0.05:
                # Successful return!
                self.ball_x = -0.85
                offset = (self.ball_y - self.paddle1_y) / (self.paddle_height / 2)
                speed = min(0.065, float(np.sqrt(self.ball_vx**2 + self.ball_vy**2) * 1.05))
                angle = offset * (np.pi / 3)
                self.ball_vx = float(speed * np.cos(angle))
                self.ball_vy = float(speed * np.sin(angle))
                self.rally += 1
                events.fly1_hit = True
            else:
                # Missed ball!
                events.fly1_miss = True
                
        # 6. Paddle 2 (Right, x = +0.85) Collision Check
        if self.ball_x >= 0.85 and self.ball_vx > 0:
            if abs(self.ball_y - self.paddle2_y) <= self.paddle_height / 2 + 0.05:
                # Successful return!
                self.ball_x = 0.85
                offset = (self.ball_y - self.paddle2_y) / (self.paddle_height / 2)
                speed = min(0.065, float(np.sqrt(self.ball_vx**2 + self.ball_vy**2) * 1.05))
                angle = offset * (np.pi / 3)
                self.ball_vx = float(-speed * np.cos(angle))
                self.ball_vy = float(speed * np.sin(angle))
                self.rally += 1
                events.fly2_hit = True
            else:
                # Missed ball!
                events.fly2_miss = True
                
        # 7. Goal / Out-of-Bounds Detection
        if self.ball_x < -1.05:
            # Fly 2 scores, Fly 1 loses point
            self.score2 += 1
            events.fly1_point_lost = True
            events.fly2_point_won = True
            events.rally_length = self.rally
            self.total_rallies += 1
            self.reset_ball(serve_to_fly1=False)
        elif self.ball_x > 1.05:
            # Fly 1 scores, Fly 2 loses point
            self.score1 += 1
            events.fly1_point_won = True
            events.fly2_point_lost = True
            events.rally_length = self.rally
            self.total_rallies += 1
            self.reset_ball(serve_to_fly1=True)
            
        events.rally_length = self.rally
        
        # 8. Render Ommatidial Visual Fields for both fly brains
        vision_fly1 = self.render_ommatidia(flip_horizontal=False)
        vision_fly2 = self.render_ommatidia(flip_horizontal=True)
        
        return vision_fly1, vision_fly2, events
        
    def render_ommatidia(self, flip_horizontal: bool = False) -> np.ndarray:
        """
        Renders a low-res ommatidial compound eye representation (e.g. 16x16 grid).
        Values in [0.0, 1.0] representing photon intensity on each ommatidium.
        """
        grid = np.zeros((self.visual_res, self.visual_res), dtype=np.float32)
        
        # Map normalized coords [-1, 1] to grid coords [0, visual_res - 1]
        bx = int(np.clip((self.ball_x + 1.0) / 2.0 * (self.visual_res - 1), 0, self.visual_res - 1))
        by = int(np.clip((self.ball_y + 1.0) / 2.0 * (self.visual_res - 1), 0, self.visual_res - 1))
        
        # Ball glow
        grid[by, bx] = 1.0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = by + dy, bx + dx
                if 0 <= ny < self.visual_res and 0 <= nx < self.visual_res:
                    grid[ny, nx] = max(grid[ny, nx], 0.5)
                    
        # Left Paddle
        p1_x = int(0.1 * (self.visual_res - 1))
        p1_ymin = int(np.clip((self.paddle1_y - self.paddle_height / 2 + 1.0) / 2.0 * (self.visual_res - 1), 0, self.visual_res - 1))
        p1_ymax = int(np.clip((self.paddle1_y + self.paddle_height / 2 + 1.0) / 2.0 * (self.visual_res - 1), 0, self.visual_res - 1))
        grid[p1_ymin:p1_ymax+1, p1_x] = 0.8
        
        # Right Paddle
        p2_x = int(0.9 * (self.visual_res - 1))
        p2_ymin = int(np.clip((self.paddle2_y - self.paddle_height / 2 + 1.0) / 2.0 * (self.visual_res - 1), 0, self.visual_res - 1))
        p2_ymax = int(np.clip((self.paddle2_y + self.paddle_height / 2 + 1.0) / 2.0 * (self.visual_res - 1), 0, self.visual_res - 1))
        grid[p2_ymin:p2_ymax+1, p2_x] = 0.8
        
        if flip_horizontal:
            grid = np.fliplr(grid)
            
        return grid

    def render_display_frame(self, width: int = 640, height: int = 400) -> Image.Image:
        """
        Renders a rich graphic view of the Pong court with scores and visual styling.
        """
        img = Image.new("RGB", (width, height), color=(15, 20, 28))
        draw = ImageDraw.Draw(img)
        
        # Court center dividing line
        for y in range(20, height - 20, 20):
            draw.line([(width // 2, y), (width // 2, y + 10)], fill=(45, 55, 72), width=2)
            
        # Court boundaries
        draw.rectangle([10, 10, width - 10, height - 10], outline=(40, 50, 70), width=2)
        
        # Map game coords to screen coords
        def to_screen(gx, gy):
            sx = int((gx + 1.0) / 2.0 * (width - 40) + 20)
            sy = int((-gy + 1.0) / 2.0 * (height - 40) + 20)
            return sx, sy
            
        # Paddle 1 (Fly 1, Emerald Green)
        p1_x, p1_y = to_screen(-0.85, self.paddle1_y)
        ph = int(self.paddle_height / 2.0 * (height - 40))
        draw.rectangle([p1_x - 6, p1_y - ph, p1_x + 6, p1_y + ph], fill=(16, 185, 129), outline=(52, 211, 153))
        
        # Paddle 2 (Fly 2, Cyber Purple)
        p2_x, p2_y = to_screen(0.85, self.paddle2_y)
        draw.rectangle([p2_x - 6, p2_y - ph, p2_x + 6, p2_y + ph], fill=(168, 85, 247), outline=(192, 132, 252))
        
        # Ball (Glowing Gold)
        bx, by = to_screen(self.ball_x, self.ball_y)
        draw.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=(250, 204, 21), outline=(254, 240, 138))
        
        # Scores & Labels
        draw.text((width // 4, 25), f"FLY 1: {self.score1}", fill=(16, 185, 129))
        draw.text((3 * width // 4 - 40, 25), f"FLY 2: {self.score2}", fill=(168, 85, 247))
        draw.text((width // 2 - 35, 25), f"Rally: {self.rally}", fill=(156, 163, 175))
        
        return img
