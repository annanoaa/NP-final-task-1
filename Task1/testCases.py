import numpy as np
import cv2
import os
from typing import List, Tuple
import random


class TestImageGenerator:
    def __init__(self,
                 image_size: Tuple[int, int] = (800, 600),
                 num_balls: int = 5,
                 ball_radius: int = 20,
                 min_distance: int = 60):
        """
        Initialize the test image generator.

        Args:
            image_size: Tuple of (width, height) for the output image
            num_balls: Number of balls to place in the image
            ball_radius: Radius of each ball in pixels
            min_distance: Minimum distance between ball centers
        """
        self.width, self.height = image_size
        self.num_balls = num_balls
        self.ball_radius = ball_radius
        self.min_distance = min_distance
        self.colors = [
            (0, 165, 255),  # Orange (basketball)
            (255, 255, 255),  # White (volleyball)
            (255, 232, 115),  # Light blue (beach ball)
            (255, 191, 0),  # Deep blue (beach ball)
            (147, 20, 255),  # Pink (beach ball)
            (0, 255, 255)  # Yellow (beach ball)
        ]

    def _is_valid_position(self, x: int, y: int, existing_positions: List[Tuple[int, int]]) -> bool:
        """Check if a new position is valid (not too close to existing balls)."""
        # Check image boundaries
        if (x - self.ball_radius < 0 or
                x + self.ball_radius >= self.width or
                y - self.ball_radius < 0 or
                y + self.ball_radius >= self.height):
            return False

        # Check distance from other balls
        for ex_x, ex_y in existing_positions:
            distance = np.sqrt((x - ex_x) ** 2 + (y - ex_y) ** 2)
            if distance < self.min_distance:
                return False
        return True

    def _generate_positions(self) -> List[Tuple[int, int]]:
        """Generate valid positions for all balls."""
        positions = []
        attempts = 0
        max_attempts = 1000

        while len(positions) < self.num_balls and attempts < max_attempts:
            # Generate random position
            x = random.randint(self.ball_radius, self.width - self.ball_radius)
            y = random.randint(self.ball_radius, self.height - self.ball_radius)

            if self._is_valid_position(x, y, positions):
                positions.append((x, y))
            attempts += 1

        if len(positions) < self.num_balls:
            raise ValueError(f"Could not place all {self.num_balls} balls after {max_attempts} attempts")

        return positions

    def generate_image(self, output_path: str = "test_image.png") -> None:
        """Generate and save the test image."""
        # Create blank image
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # Generate positions for balls
        positions = self._generate_positions()

        # Draw balls
        for i, (x, y) in enumerate(positions):
            color = self.colors[i % len(self.colors)]
            # Draw filled circle
            cv2.circle(image, (x, y), self.ball_radius, color, -1)
            # Draw border
            cv2.circle(image, (x, y), self.ball_radius, (0, 0, 0), 2)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Save image
        cv2.imwrite(output_path, image)
        print(f"Test image generated and saved to: {output_path}")


def main():
    # Create multiple test cases with different configurations
    configs = [
        {"image_size": (800, 600), "num_balls": 3, "ball_radius": 20, "min_distance": 100},
        {"image_size": (1000, 800), "num_balls": 5, "ball_radius": 25, "min_distance": 120},
        {"image_size": (1200, 900), "num_balls": 7, "ball_radius": 30, "min_distance": 150}
    ]

    # Generate test images
    for i, config in enumerate(configs, 1):
        try:
            generator = TestImageGenerator(**config)
            generator.generate_image(f"images/test{i}.png")
        except Exception as e:
            print(f"Error generating test case {i}: {e}")


if __name__ == "__main__":
    main()