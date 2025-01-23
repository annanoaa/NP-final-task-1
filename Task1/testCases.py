import unittest
import numpy as np
import cv2
from task1 import TargetSimulation


class TestTargetSimulation(unittest.TestCase):
    def setUp(self):
        """Initialize test cases with known properties"""
        self.test_cases = [
            {
                "path": "images/test1.png",
                "expected_targets": 3,
                "colors": ["cyan", "orange", "white"],
                "description": "Basic three-target configuration"
            },
            {
                "path": "images/test2.png",
                "expected_targets": 5,
                "colors": ["cyan", "cyan", "orange", "white", "pink"],
                "description": "Complex five-target layout"
            },
            {
                "path": "images/test3.png",
                "expected_targets": 7,
                "colors": ["yellow", "cyan", "orange", "white", "orange", "pink"],
                "description": "Scattered six-target pattern"
            },
            {
                "path": "images/test4.png",
                "expected_targets": 7,
                "colors": ["white"],
                "description": "Complex 7-target layout"
            }
        ]
        self.test_image_paths = [case["path"] for case in self.test_cases]

    def test_edge_detection(self):
        """Test edge detection functionality"""
        sim = TargetSimulation(self.test_image_paths[0])
        gray_img = cv2.cvtColor(sim.img, cv2.COLOR_BGR2GRAY)
        edges = sim.edge_detection(gray_img)

        # Verify edge detection output
        self.assertEqual(edges.shape, gray_img.shape)
        self.assertEqual(edges.dtype, np.uint8)
        self.assertTrue(np.any(edges > 0))  # Should detect some edges

    def test_target_detection(self):
        """Test target detection accuracy"""
        for path in self.test_image_paths:
            sim = TargetSimulation(path)
            sim.locate_targets()

            # Verify targets were detected
            self.assertTrue(len(sim.targets) > 0)

            # Verify target properties
            for target in sim.targets:
                self.assertIn('center', target)
                self.assertIn('radius', target)
                self.assertIn('hit', target)
                self.assertFalse(target['hit'])  # Should be False initially

    def test_path_generation(self):
        """Test trajectory generation"""
        sim = TargetSimulation(self.test_image_paths[0])
        start = (0, 0)
        end = (100, 100)

        path = sim.generate_path(start, end)

        # Verify path properties
        self.assertEqual(len(path), 40)  # Should have 40 points
        self.assertEqual(path[0], start)  # Should start at start point
        self.assertTrue(abs(path[-1][0] - end[0]) <= 1)  # Should end near end point
        self.assertTrue(abs(path[-1][1] - end[1]) <= 1)

    def test_video_creation(self):
        """Test video output generation"""
        sim = TargetSimulation(self.test_image_paths[0])
        output_file = 'test_simulation.avi'
        sim.create_video(output_file)

        # Verify video file was created
        import os
        self.assertTrue(os.path.exists(output_file))

        # Clean up
        os.remove(output_file)


if __name__ == '__main__':
    unittest.main()