from __future__ import annotations

import unittest

import cv2
import numpy as np

import road_detector as rd


class RoadAutolocateTests(unittest.TestCase):
    def test_portrait_white_road_board_generates_upper_middle_candidates(self) -> None:
        image = np.zeros((1280, 590, 3), dtype=np.uint8)
        image[:] = (28, 30, 36)
        cv2.rectangle(image, (0, 910), (589, 1160), (245, 245, 245), -1)
        # Simulate a central upper big-road area with red/blue marks.
        for column, bgr in enumerate(((0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0))):
            cx = 205 + column * 20
            cy = 950
            cv2.circle(image, (cx, cy), 7, bgr, 2)
        candidates = rd._autolocate_candidate_rois(image)
        self.assertTrue(candidates)
        self.assertTrue(any(item.get("source") == "bright_board_upper_middle" for item in candidates))
        # The fallback should search the lower road-paper area, not the betting UI/top half.
        self.assertTrue(any(float(item["roi"][1]) >= 0.65 for item in candidates))

    def test_road_only_crop_always_gets_whole_image_candidate(self) -> None:
        image = np.full((116, 676, 3), 245, dtype=np.uint8)
        candidates = rd._autolocate_candidate_rois(image)
        names = {str(item.get("name")) for item in candidates}
        self.assertIn("autolocate_whole_image_6xN", names)
        whole = next(item for item in candidates if item.get("name") == "autolocate_whole_image_6xN")
        self.assertEqual(tuple(whole["roi"]), (0.0, 0.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
