"""
Cell Tracker - Synthetic Data Generator
"""

import os
import cv2
import numpy as np
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Cell:
    cell_id: int
    x: float
    y: float
    size: float
    intensity: int
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 2.0
    history: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.history = [(self.x, self.y)]


class SyntheticDataGenerator:
    """Generate synthetic microscopy images with cells."""

    def __init__(self, width=800, height=600, num_cells=100,
                 num_frames=50, seed=None, overlap_density: float = 0.0):
        """
        overlap_density in [0, 1]: fraction of cells spawned deliberately
        on top of another cell (tight clusters). 0 = classic behaviour
        (random non-overlapping scatter), 0.5 = heavy clustering — the
        tracker should still survive this without ID swaps.
        """
        self.width = width
        self.height = height
        self.num_cells = num_cells
        self.num_frames = num_frames
        self.overlap_density = max(0.0, min(1.0, float(overlap_density)))

        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.cells: List[Cell] = []

    def generate_cells(self):
        """Create cell population.

        When overlap_density > 0 a fraction of cells is spawned within
        0.4-0.9 diameters of a previously-placed cell — i.e. bounding
        boxes overlap but centres differ, which is the exact situation
        that produced ID swaps in earlier builds.
        """
        self.cells = []
        margin = 50

        for i in range(self.num_cells):
            size = random.uniform(12, 30)
            intensity = random.randint(140, 220)
            speed = random.uniform(1, 4)
            angle = random.uniform(0, 2 * math.pi)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)

            if self.cells and random.random() < self.overlap_density:
                anchor = random.choice(self.cells)
                offset = random.uniform(0.4, 0.9) * (size + anchor.size) / 2.0
                theta = random.uniform(0, 2 * math.pi)
                x = min(self.width - margin,
                        max(margin, anchor.x + offset * math.cos(theta)))
                y = min(self.height - margin,
                        max(margin, anchor.y + offset * math.sin(theta)))
                # Slightly different intensity so appearance differs a bit.
                intensity = random.randint(130, 230)
            else:
                x = random.uniform(margin, self.width - margin)
                y = random.uniform(margin, self.height - margin)

            self.cells.append(Cell(
                cell_id=i, x=x, y=y, size=size,
                intensity=intensity, vx=vx, vy=vy, speed=speed
            ))

        print(f"Created {len(self.cells)} cells")
    
    def _update_cell(self, cell):
        """Update cell position."""
        # Random direction change
        if random.random() > 0.85:
            angle = random.uniform(0, 2 * math.pi)
            cell.vx = cell.speed * math.cos(angle)
            cell.vy = cell.speed * math.sin(angle)
        
        cell.x += cell.vx
        cell.y += cell.vy
        
        # Bounce off walls
        margin = 20
        if cell.x < margin:
            cell.x = margin
            cell.vx *= -1
        elif cell.x > self.width - margin:
            cell.x = self.width - margin
            cell.vx *= -1
        
        if cell.y < margin:
            cell.y = margin
            cell.vy *= -1
        elif cell.y > self.height - margin:
            cell.y = self.height - margin
            cell.vy *= -1
        
        cell.history.append((cell.x, cell.y))
    
    def _draw_cell(self, image, cell):
        """Draw a cell on the image."""
        x, y = int(cell.x), int(cell.y)
        size = int(cell.size)
        intensity = cell.intensity
        
        # Cell body (bright circle)
        cv2.circle(image, (x, y), size, (intensity, intensity, intensity), -1)
        
        # Membrane (slightly darker edge)
        cv2.circle(image, (x, y), size, (intensity - 30, intensity - 30, intensity - 30), 2)
        
        # Nucleus (dark center)
        nucleus_size = max(3, int(size * 0.35))
        cv2.circle(image, (x, y), nucleus_size, 
                  (intensity - 60, intensity - 60, intensity - 60), -1)
    
    def generate_frame(self, frame_idx):
        """Generate one frame."""
        # Update positions
        for cell in self.cells:
            self._update_cell(cell)
        
        # Create dark background
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 25
        
        # Add noise
        noise = np.random.randint(0, 12, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        # Draw cells
        gt = {}
        for cell in self.cells:
            self._draw_cell(image, cell)
            size = int(cell.size)
            gt[cell.cell_id] = {
                "bbox": (int(cell.x - size), int(cell.y - size), size * 2, size * 2),
                "center": (cell.x, cell.y)
            }
        
        return image, gt
    
    def generate_dataset(self, output_dir):
        """Generate complete dataset."""
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"\nGenerating {self.num_frames} frames with {self.num_cells} cells...")
        
        self.generate_cells()
        
        for i in range(self.num_frames):
            image, _ = self.generate_frame(i)
            path = os.path.join(frames_dir, f"frame_{i:05d}.png")
            cv2.imwrite(path, image)
            
            if (i + 1) % 10 == 0:
                print(f"  Frame {i + 1}/{self.num_frames}")
        
        print(f"Dataset saved to: {output_dir}")
        return output_dir
