from __future__ import annotations

import unittest

import cv2
import numpy as np

import road_detector as rd


def _synthetic_grid(height: int, width: int, pitch_x: int, pitch_y: int) -> np.ndarray:
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    for x in range(0, width, pitch_x):
        cv2.line(image, (x, 0), (x, height - 1), (190, 190, 190), 1)
    for y in range(0, height, pitch_y):
        cv2.line(image, (0, y), (width - 1, y), (190, 190, 190), 1)
    for column, bgr in enumerate(((0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0))):
        cv2.circle(
            image,
            (pitch_x // 2 + column * pitch_x, pitch_y // 2),
            max(3, min(pitch_x, pitch_y) // 3),
            bgr,
            2,
        )
    return image


class RoadAutolocateTests(unittest.TestCase):
    def test_portrait_white_road_board_generates_upper_middle_candidates(self) -> None:
        image = np.zeros((1280, 590, 3), dtype=np.uint8)
        image[:] = (28, 30, 36)
        cv2.rectangle(image, (0, 910), (589, 1160), (245, 245, 245), -1)
        for column, bgr in enumerate(((0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0))):
            cx = 205 + column * 20
            cy = 950
            cv2.circle(image, (cx, cy), 7, bgr, 2)
        candidates = rd._autolocate_candidate_rois(image)
        self.assertTrue(candidates)
        self.assertTrue(any(item.get("source") == "bright_board_upper_middle" for item in candidates))
        self.assertTrue(any(float(item["roi"][1]) >= 0.65 for item in candidates))

    def test_road_only_crop_always_gets_whole_image_candidate(self) -> None:
        image = np.full((116, 676, 3), 245, dtype=np.uint8)
        candidates = rd._autolocate_candidate_rois(image)
        names = {str(item.get("name")) for item in candidates}
        self.assertIn("autolocate_whole_image_6xN", names)
        whole = next(item for item in candidates if item.get("name") == "autolocate_whole_image_6xN")
        self.assertEqual(tuple(whole["roi"]), (0.0, 0.0, 1.0, 1.0))

    def test_wide_road_crop_can_exceed_legacy_32_column_limit(self) -> None:
        image = _synthetic_grid(116, 676, 19, 19)
        trials = rd._grid_column_trials(image)
        self.assertTrue(trials)
        self.assertEqual(int(trials[0]["grid_columns"]), 36)
        self.assertEqual(str(trials[0]["source"]), "grid_period")
        self.assertAlmostEqual(float(trials[0]["period"]), 19.0, places=1)
        self.assertGreater(float(trials[0]["correlation"]), 0.5)

    def test_rectangular_mobile_grid_uses_real_horizontal_pitch(self) -> None:
        image = _synthetic_grid(107, 343, 25, 18)
        trials = rd._grid_column_trials(image)
        self.assertTrue(trials)
        self.assertEqual(int(trials[0]["grid_columns"]), 14)
        self.assertEqual(str(trials[0]["source"]), "grid_period")
        self.assertAlmostEqual(float(trials[0]["period"]), 25.0, places=1)
        # The old square-cell aspect estimate would be about 19 columns.
        self.assertNotEqual(int(trials[0]["grid_columns"]), 19)


if __name__ == "__main__":
    unittest.main()
