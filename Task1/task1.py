import numpy as np
import cv2
import random

class TargetSimulation:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise ValueError(f"Unable to load image from {img_path}")
        self.targets = []
        self.launcher_pos = None

    def edge_detection(self, image):
        """Manual implementation of Sobel edge detection"""
        # Sobel kernels
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32)

        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]], dtype=np.float32)

        height, width = image.shape
        grad_x = np.zeros((height, width), dtype=np.float32)
        grad_y = np.zeros((height, width), dtype=np.float32)

        # Compute gradients
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                region = image[row - 1:row + 2, col - 1:col + 2].astype(np.float32)
                grad_x[row, col] = np.sum(region * kernel_x)
                grad_y[row, col] = np.sum(region * kernel_y)

        # Compute magnitude
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize to 0-255 range
        if magnitude.max() > 0:
            magnitude = magnitude * (255.0 / magnitude.max())

        return magnitude.astype(np.uint8)

    def locate_targets(self):
        """Enhanced target detection using custom edge detection"""
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_img, (9, 9), 2)
        median_blur_img = cv2.medianBlur(gray_img, 7)

        edges_blur = self.edge_detection(blur_img)
        edges_median = self.edge_detection(median_blur_img)

        combined_edges = cv2.bitwise_or(edges_blur, edges_median)
        _, binary_edges = cv2.threshold(combined_edges, 50, 255, cv2.THRESH_BINARY)

        circles = cv2.HoughCircles(
            median_blur_img,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=100
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                self.targets.append({'center': center, 'radius': radius, 'hit': False})

        contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if 0.6 < circularity < 1.4:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                if radius > 15:
                    self.targets.append({'center': center, 'radius': radius, 'hit': False})

        filtered_targets = []
        for target in self.targets:
            is_duplicate = False
            for filtered_target in filtered_targets:
                dist = np.sqrt((target['center'][0] - filtered_target['center'][0]) ** 2 +
                               (target['center'][1] - filtered_target['center'][1]) ** 2)
                if dist < max(target['radius'], filtered_target['radius']):
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_targets.append(target)

        self.targets = filtered_targets
        print(f"Detected {len(self.targets)} targets")

        cv2.imwrite('edges_debug.png', combined_edges)
        debug_img = self.img.copy()
        for target in self.targets:
            cv2.circle(debug_img, target['center'], target['radius'], (0, 255, 0), 2)
        cv2.imwrite('detected_targets_debug.png', debug_img)

    def generate_path(self, start, end, height_factor=0.5):
        """Generate a smooth trajectory"""
        path_points = []
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        height_factor *= distance / 400

        mid_x = (start[0] + end[0]) / 2
        mid_y = min(start[1], end[1]) - abs(dx * height_factor)

        for t in np.linspace(0, 1, 40):
            x = int((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid_x + t ** 2 * end[0])
            y = int((1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid_y + t ** 2 * end[1])
            path_points.append((x, y))

        return path_points

    def create_video(self, output_file='simulation.avi'):
        if not self.targets:
            self.locate_targets()

        height, width = self.img.shape[:2]
        self.launcher_pos = (width // 2, 50)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_file, fourcc, 30, (width, height))

        for target in self.targets:
            path = self.generate_path(self.launcher_pos, target['center'])

            for i in range(len(path)):
                frame = self.img.copy()

                # Draw the trajectory line
                if i > 0:
                    pts = np.array(path[:i + 1], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, (255, 0, 0), 2)

                # Draw the moving small ball
                small_ball_center = path[i]
                cv2.circle(frame, small_ball_center, 5, (255, 0, 0), -1)  # Changed to vibrant red

                video_writer.write(frame)

            target['hit'] = True

            for _ in range(5):
                frame = self.img.copy()
                # No need to draw circles around the main targets
                video_writer.write(frame)

        video_writer.release()

if __name__ == "__main__":
    image_path = "images/test4.png"
    simulation = TargetSimulation(image_path)
    simulation.create_video()