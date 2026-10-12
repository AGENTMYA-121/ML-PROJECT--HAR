#!/usr/bin/env python3
"""
Generate Synthetic Activity Clips for Testing & Demonstration
=============================================================
Creates realistic synthetic action video clips with moving humanoid stick figures
for classes: JumpingJack, PushUps, Squats, WalkingWithDog, Punch.
This allows full training, evaluation, and inference validation immediately out-of-the-box.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import cv2
import numpy as np


ACTIONS = [
    "ApplyEyeMakeup",
    "Biking",
    "JumpingJack",
    "PushUps",
    "Squats",
    "WalkingWithDog",
]


def draw_stick_figure(img: np.ndarray, t: float, action: str):
    """Draw a dynamic stick figure executing the action motion at time phase t."""
    h, w, _ = img.shape
    cx, cy = w // 2, h // 2
    color = (255, 255, 255)
    joint_color = (0, 200, 255)
    thickness = 3

    # Add background noise/subtle texture so CNN doesn't just see a black box
    noise = np.random.randint(15, 35, (h, w, 3), dtype=np.uint8)
    img[:] = cv2.add(img, noise)

    # Floor line
    cv2.line(img, (20, h - 30), (w - 20, h - 30), (70, 70, 80), 2)

    sin_t = math.sin(t * 2 * math.pi)
    cos_t = math.cos(t * 2 * math.pi)

    if action == "JumpingJack":
        # Body jumps up and down slightly
        by = cy + int(10 * sin_t)
        # Head
        cv2.circle(img, (cx, by - 55), 16, joint_color, -1)
        # Spine
        cv2.line(img, (cx, by - 40), (cx, by + 15), color, thickness)
        # Arms sweep overhead
        arm_angle = (sin_t + 1) * 0.5  # 0 to 1
        arm_dx = int(45 * math.cos(math.pi * 0.8 * arm_angle - math.pi * 0.4))
        arm_dy = -int(45 * math.sin(math.pi * 0.8 * arm_angle - math.pi * 0.4))
        # Left arm
        cv2.line(img, (cx, by - 30), (cx - arm_dx, by - 30 + arm_dy), color, thickness)
        cv2.line(img, (cx - arm_dx, by - 30 + arm_dy), (cx - arm_dx - 15, by - 30 + arm_dy - 20), color, thickness)
        # Right arm
        cv2.line(img, (cx, by - 30), (cx + arm_dx, by - 30 + arm_dy), color, thickness)
        cv2.line(img, (cx + arm_dx, by - 30 + arm_dy), (cx + arm_dx + 15, by - 30 + arm_dy - 20), color, thickness)
        # Legs spread out
        leg_spread = int(35 * (sin_t + 1) * 0.5) + 10
        cv2.line(img, (cx, by + 15), (cx - leg_spread, by + 75), color, thickness)
        cv2.line(img, (cx, by + 15), (cx + leg_spread, by + 75), color, thickness)

    elif action == "PushUps":
        # Horizontal figure lowering up and down
        py = cy + 20 + int(20 * (sin_t + 1) * 0.5)
        # Head
        cv2.circle(img, (cx - 60, py - 5), 14, joint_color, -1)
        # Torso
        cv2.line(img, (cx - 45, py), (cx + 50, py + 15), color, thickness)
        # Arms
        cv2.line(img, (cx - 35, py), (cx - 35, h - 35), color, thickness)
        # Legs
        cv2.line(img, (cx + 50, py + 15), (cx + 90, h - 35), color, thickness)

    elif action == "Squats":
        # Torso descends
        squat_depth = int(30 * (sin_t + 1) * 0.5)
        sy = cy - 20 + squat_depth
        cv2.circle(img, (cx, sy - 40), 16, joint_color, -1)
        # Spine
        cv2.line(img, (cx, sy - 25), (cx, sy + 25), color, thickness)
        # Arms stretched forward
        cv2.line(img, (cx, sy - 15), (cx + 40, sy - 15), color, thickness)
        # Legs bending
        knee_x = cx + int(20 * (sin_t + 1) * 0.5)
        cv2.line(img, (cx, sy + 25), (knee_x, sy + 55), color, thickness)
        cv2.line(img, (knee_x, sy + 55), (cx + 10, h - 32), color, thickness)

    elif action == "WalkingWithDog":
        # Figure translating horizontally + leg swing
        wx = (cx - 40 + int(sin_t * 20))
        cv2.circle(img, (wx, cy - 45), 15, joint_color, -1)
        cv2.line(img, (wx, cy - 30), (wx, cy + 20), color, thickness)
        # Legs swinging
        l_offset = int(25 * sin_t)
        cv2.line(img, (wx, cy + 20), (wx - l_offset, cy + 75), color, thickness)
        cv2.line(img, (wx, cy + 20), (wx + l_offset, cy + 75), color, thickness)
        # Leash and small dog
        cv2.line(img, (wx + 15, cy - 10), (wx + 60, cy + 50), (120, 180, 250), 1)
        cv2.rectangle(img, (wx + 55, cy + 45), (wx + 85, cy + 65), (100, 150, 200), -1)

    else:
        # Generic action (waving / makeup / gestures)
        cv2.circle(img, (cx, cy - 45), 16, joint_color, -1)
        cv2.line(img, (cx, cy - 30), (cx, cy + 25), color, thickness)
        hand_y = cy - 40 + int(15 * sin_t)
        cv2.line(img, (cx, cy - 20), (cx + 35, hand_y), color, thickness)
        cv2.line(img, (cx, cy + 25), (cx - 15, cy + 75), color, thickness)
        cv2.line(img, (cx, cy + 25), (cx + 15, cy + 75), color, thickness)


def generate_video(file_path: Path, action: str, num_frames: int = 32, fps: int = 25):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 256, 256
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (w, h))

    for i in range(num_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Action phase
        t = i / float(num_frames)
        draw_stick_figure(img, t, action)
        out.write(img)

    out.release()


def main():
    base_dir = Path("c:/projecct/har-project/data/ucf101")
    clips_per_action = 4
    total_clips = len(ACTIONS) * clips_per_action

    print(f"Generating {total_clips} synthetic action clips for immediate testing...")

    for action in ACTIONS:
        act_dir = base_dir / action
        act_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, clips_per_action + 1):
            vid_name = f"v_{action}_g{i:02d}_c01.mp4"
            vid_path = act_dir / vid_name
            generate_video(vid_path, action, num_frames=32, fps=25)
            print(f"  Created: {vid_path.relative_to(base_dir.parent)}")

    print(f"\nSynthetic test dataset successfully generated at: {base_dir}")


if __name__ == "__main__":
    main()
