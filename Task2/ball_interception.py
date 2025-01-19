import cv2
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os


class BallTracker:
    def __init__(self):
        self.g = 9.81  # gravitational acceleration
        self.positions = []  # store detected ball positions
        self.timestamps = []  # store corresponding timestamps

    def detect_ball(self, frame):
        """Detect the ball in a frame using color thresholding and contour detection"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range for ball color (adjust these values based on your ball)
        # These values are for a red ball - adjust based on your ball's color
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([180, 255, 255])

        # Create masks for both ranges and combine
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assumed to be the ball)
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            # Debug visualization
            debug_frame = frame.copy()
            cv2.circle(debug_frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.imshow('Ball Detection', debug_frame)
            cv2.imshow('Mask', mask)
            cv2.waitKey(1)

            return (int(x), int(y)), int(radius)
        return None, None

    def ball_motion_ode(self, state, t):
        """Define the ODE system for ball motion"""
        x, y, vx, vy = state
        return [vx, vy, 0, -self.g]

    def fit_trajectory(self):
        """Fit a trajectory to the detected ball positions using ODE"""
        if len(self.positions) < 2:
            return None

        # Extract x and y coordinates
        x_coords = np.array([p[0] for p in self.positions])
        y_coords = np.array([p[1] for p in self.positions])
        t = np.array(self.timestamps)

        # Estimate initial velocities using finite differences
        v0x = (x_coords[1] - x_coords[0]) / (t[1] - t[0])
        v0y = (y_coords[1] - y_coords[0]) / (t[1] - t[0])

        # Initial state [x0, y0, v0x, v0y]
        initial_state = [x_coords[0], y_coords[0], v0x, v0y]

        # Time points for integration
        t_span = np.linspace(0, max(t) * 2, 100)  # Extend beyond observed time

        # Solve ODE using different methods
        solution_rk = odeint(self.ball_motion_ode, initial_state, t_span)

        return t_span, solution_rk

    def calculate_intercept(self, launch_point, target_trajectory, t_span):
        """Calculate interception point and required initial velocity"""

        # Implement shooting method to find intercept
        def shooting_function(v0):
            # Initial state for interceptor
            state0 = [launch_point[0], launch_point[1], v0[0], v0[1]]

            # Solve ODE for interceptor
            solution = odeint(self.ball_motion_ode, state0, t_span)

            # Find minimum distance to target trajectory
            distances = np.sqrt(
                (solution[:, 0:1] - target_trajectory[:, 0:1].T) ** 2 +
                (solution[:, 1:2] - target_trajectory[:, 1:2].T) ** 2
            )
            return np.min(distances)

        # Use optimization to find initial velocity
        from scipy.optimize import minimize

        # Initial guess for velocity
        v0_guess = [10, 10]

        # Optimize
        result = minimize(shooting_function, v0_guess, method='Nelder-Mead')

        return result.x

    def process_video(self, video_path):
        """Process video and track ball"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        frame_count = 0
        first_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if first_frame is None:
                first_frame = frame

            center, radius = self.detect_ball(frame)
            if center is not None:
                self.positions.append(center)
                self.timestamps.append(frame_count / cap.get(cv2.CAP_PROP_FPS))
                print(f"Frame {frame_count}: Ball detected at position {center}")

            frame_count += 1

        cap.release()

        if len(self.positions) < 2:
            raise ValueError("Not enough ball positions detected. Check the ball color thresholds.")

        # Fit trajectory
        result = self.fit_trajectory()
        if result is None:
            raise ValueError("Failed to fit trajectory to detected positions")

        t_span, trajectory = result

        # Choose random launch point
        launch_x = np.random.randint(50, first_frame.shape[1] - 50)
        launch_y = np.random.randint(50, first_frame.shape[0] - 50)
        launch_point = (launch_x, launch_y)

        # Calculate intercept
        v0 = self.calculate_intercept(launch_point, trajectory, t_span)

        return t_span, trajectory, launch_point, v0

    def create_animation(self, video_path, t_span, trajectory, launch_point, v0):
        """Create animation of original trajectory and intercepting ball"""
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Changed from MP4V to XVID
        out = cv2.VideoWriter('output.avi', fourcc, 30.0,  # Changed output to .avi
                              (int(cap.get(3)), int(cap.get(4))))

        # Calculate interceptor trajectory
        state0 = [launch_point[0], launch_point[1], v0[0], v0[1]]
        interceptor = odeint(self.ball_motion_ode, state0, t_span)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Draw original trajectory
            for i in range(len(trajectory)):
                pos = (int(trajectory[i, 0]), int(trajectory[i, 1]))
                cv2.circle(frame, pos, 5, (0, 255, 0), -1)

            # Draw interceptor
            if frame_idx < len(interceptor):
                pos = (int(interceptor[frame_idx, 0]), int(interceptor[frame_idx, 1]))
                cv2.circle(frame, pos, 5, (0, 0, 255), -1)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()


def main():
    # Initialize tracker
    tracker = BallTracker()

    # Process video
    video_path = "videos/test_1.mp4"  # Replace with your video path
    t_span, trajectory, launch_point, v0 = tracker.process_video(video_path)

    # Create animation
    tracker.create_animation(video_path, t_span, trajectory, launch_point, v0)


if __name__ == "__main__":
    main()