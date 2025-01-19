import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import cv2
from typing import List, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Point:
    """Simple class to represent 2D points"""
    x: float
    y: float

class NumericalSolver:
    """Implements numerical methods for solving ODEs"""

    def __init__(self, g: float = 9.81):
        """
        Initialize solver with gravitational constant
        Args:
            g (float): Gravitational acceleration (default: 9.81 m/s²)
        """
        self.g = g

    def runge_kutta_4th(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        Fourth-order Runge-Kutta method implementation
        Args:
            state: Current state [x, y, vx, vy]
            dt: Time step
        Returns:
            Next state after time step
        """
        k1 = self._derivatives(state)
        k2 = self._derivatives(state + 0.5 * dt * k1)
        k3 = self._derivatives(state + 0.5 * dt * k2)
        k4 = self._derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Calculate derivatives for the system of ODEs"""
        _, _, vx, vy = state
        return np.array([vx, vy, 0, -self.g])

class TrajectoryCalculator:
    """Handles trajectory calculations using shooting method"""

    def __init__(self, solver: NumericalSolver):
        self.solver = solver
        self.dt = 0.01

    def calculate_trajectory(self,
                           start: Point,
                           velocity: Tuple[float, float],
                           max_time: float = 10.0) -> Tuple[List[float], List[float]]:
        """
        Calculate complete trajectory given initial conditions
        Args:
            start: Starting point
            velocity: Initial velocity components (vx, vy)
            max_time: Maximum simulation time
        Returns:
            Lists of x and y coordinates of trajectory points
        """
        state = np.array([start.x, start.y, velocity[0], velocity[1]])
        x_points = [start.x]
        y_points = [start.y]

        t = 0
        while t < max_time and state[1] >= 0:
            state = self.solver.runge_kutta_4th(state, self.dt)
            x_points.append(state[0])
            y_points.append(state[1])
            t += self.dt

        return x_points, y_points

    def find_trajectory(self,
                       start: Point,
                       target: Point) -> Tuple[List[float], List[float]]:
        """
        Find trajectory using shooting method
        Args:
            start: Starting point
            target: Target point
        Returns:
            Trajectory coordinates that hit the target
        """
        dx = target.x - start.x
        dy = target.y - start.y
        distance = np.sqrt(dx*dx + dy*dy)

        min_v = np.sqrt(self.solver.g * distance)
        speed_ranges = np.linspace(min_v, min_v * 3, 20)
        angle_ranges = np.linspace(np.arctan2(dy, dx) - np.pi/4, np.arctan2(dy, dx) + np.pi/4, 40)  # Center angles around the direction to the target

        best_trajectory = None
        min_error = float('inf')

        for speed in speed_ranges:
            for angle in angle_ranges:
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)

                x_points, y_points = self.calculate_trajectory(start, (vx, vy))
                error = ((x_points[-1] - target.x)**2 + (y_points[-1] - target.y)**2)**0.5

                if error < min_error:
                    min_error = error
                    best_trajectory = (x_points, y_points)

                if error < 10.0:
                    return x_points, y_points

        if best_trajectory is not None:
            return best_trajectory

        raise ValueError(f"Could not find valid trajectory to target at ({target.x}, {target.y})")

class BallDetector:
    """Handles ball detection in input images"""

    @staticmethod
    def detect_balls(image_path: str) -> List[Point]:
        """
        Detect balls in the input image
        Args:
            image_path: Path to input image
        Returns:
            List of detected ball positions
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )

        if circles is None:
            raise ValueError("No balls detected in image")

        balls = []
        for x, y, _ in circles[0]:
            balls.append(Point(float(x), float(y)))

        return sorted(balls, key=lambda p: p.x)

class TrajectoryAnimator:
    """Handles creation of trajectory animations"""

    def __init__(self, fig_size: Tuple[int, int] = (10, 6), starting_point: Point = Point(0, 0)):
        self.fig, self.ax = plt.subplots(figsize=fig_size)
        self.starting_point = starting_point

    def create_animation(self,
                        trajectories: List[Tuple[List[float], List[float]]],
                        balls: List[Point],
                        output_path: str,
                        fps: int = 30):
        """
        Create and save animation showing trajectories from random starting point
        Args:
            trajectories: List of trajectory coordinates
            balls: List of ball positions
            output_path: Path to save animation
            fps: Frames per second
        """
        if not trajectories:
            raise ValueError("No valid trajectories to animate")

        def init():
            self.ax.clear()
            # Plot starting point
            self.ax.plot(self.starting_point.x, self.starting_point.y, 'gs', markersize=15, label='Starting Point')
            # Plot target balls
            for ball in balls:
                self.ax.plot(ball.x, ball.y, 'ro', markersize=10)
            self.ax.grid(True)
            self.ax.legend()
            return []

        def animate(frame):
            self.ax.clear()
            # Always show starting point
            self.ax.plot(self.starting_point.x, self.starting_point.y, 'gs', markersize=15, label='Starting Point')
            # Show all target balls
            for ball in balls:
                self.ax.plot(ball.x, ball.y, 'ro', markersize=10)

            # Show current trajectory
            traj_idx = frame // 50
            if traj_idx < len(trajectories):
                x_points, y_points = trajectories[traj_idx]
                frame_in_traj = frame % 50
                ratio = frame_in_traj / 50
                points_to_show = int(len(x_points) * ratio)

                if points_to_show > 0:
                    # Draw trajectory path
                    self.ax.plot(x_points[:points_to_show],
                               y_points[:points_to_show],
                               'b-', linewidth=2)
                    # Draw moving ball
                    self.ax.plot(x_points[points_to_show-1],
                               y_points[points_to_show-1],
                               'bo', markersize=8)

                    # Add trajectory number label
                    self.ax.text(0.02, 0.98, f'Trajectory {traj_idx + 1}/{len(trajectories)}',
                               transform=self.ax.transAxes,
                               verticalalignment='top')

            # Set consistent axes limits
            max_x = max(max(ball.x for ball in balls),
                       max(max(x) for x, _ in trajectories)) * 1.2
            max_y = max(max(ball.y for ball in balls),
                       max(max(y) for _, y in trajectories)) * 1.2
            self.ax.set_xlim([-10, max_x])  # Start slightly before (0,0)
            self.ax.set_ylim([-10, max_y])

            # Add labels and grid
            self.ax.grid(True)
            self.ax.set_title('Ball Trajectory Simulation\nAll trajectories start from a random point')
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')
            self.ax.legend(['Starting Point', 'Target Balls', 'Current Trajectory'])

            return []

        # Create animation
        frames = len(trajectories) * 50
        anim = FuncAnimation(self.fig, animate, init_func=init,
                           frames=frames, interval=1000//fps, blit=True)

        # Save animation
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close()

def main():
    """Main function to run the simulation"""
    try:
        # Initialize components
        solver = NumericalSolver()
        calculator = TrajectoryCalculator(solver)
        detector = BallDetector()

        # Create output directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Detect balls in input image
        image_path = "images/test2.png"  # Use the generated test image
        balls = detector.detect_balls(image_path)
        logger.info(f"Detected {len(balls)} balls in image")

        # Generate a random starting point within the image boundaries
        starting_point = Point(random.randint(0, 1000), random.randint(0, 1000))
        logger.info(f"Random starting point: ({starting_point.x}, {starting_point.y})")

        # Initialize animator with the random starting point
        animator = TrajectoryAnimator(starting_point=starting_point)

        # Scale coordinates if needed
        scale_factor = 1.0
        if any(ball.x > 1000 or ball.y > 1000 for ball in balls):
            scale_factor = 100.0
            balls = [Point(b.x/scale_factor, b.y/scale_factor) for b in balls]

        # Calculate trajectories
        trajectories = []

        for target in balls:
            try:
                x_points, y_points = calculator.find_trajectory(starting_point, target)
                trajectories.append((x_points, y_points))
                logger.info(f"Calculated trajectory from starting point to target at ({target.x:.2f}, {target.y:.2f})")
            except ValueError as e:
                logger.warning(f"Failed to find optimal trajectory: {e}")
                continue

        if not trajectories:
            raise ValueError("No valid trajectories were found")

        # Scale back if needed
        if scale_factor != 1.0:
            trajectories = [([x*scale_factor for x in xs], [y*scale_factor for y in ys])
                          for xs, ys in trajectories]
            balls = [Point(b.x*scale_factor, b.y*scale_factor) for b in balls]

        # Create animation
        output_file = os.path.join(output_dir, f"ball_trajectories_{timestamp}.gif")
        animator.create_animation(trajectories, balls, output_file)
        logger.info(f"Animation saved successfully as: {os.path.abspath(output_file)}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()