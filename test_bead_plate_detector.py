"""Synthetic bead-plate tests. No user images committed to the repository."""
import unittest
import cv2
import numpy as np
from bead_plate_detector import detect_bead_plate_sequence_detailed

COLORS={"B":(20,24,228),"P":(232,97,25),"T":(45,144,46)}


def make_plate(sequence, *, pitch=33, radius=13, dark=False):
    columns=(len(sequence)+5)//6
    width=columns*pitch+pitch//2
    height=6*pitch+pitch//2
    background=(69,26,105) if dark else (250,250,250)
    image=np.full((height,width,3),background,dtype=np.uint8)
    for i,letter in enumerate(sequence):
        col,row=divmod(i,6)
        center=(int(pitch*.55+col*pitch),int(pitch*.55+row*pitch))
        cv2.circle(image,center,radius,COLORS[letter],-1,cv2.LINE_AA)
        cv2.putText(image,"6",(center[0]-5,center[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1,cv2.LINE_AA)
    return image


class BeadPlateTests(unittest.TestCase):
    def test_white_and_dark_complete_bead_plate_including_ties(self):
        history=list('BPTBBTPPBTTB')+list('PBBTPB')*3+list('BPT')
        for dark,pitch,radius,venue in (
            (False,30,12,'DG'),(False,37,14,'MT'),(True,27,10,'DB')
        ):
            with self.subTest(venue=venue):
                image=make_plate(history,pitch=pitch,radius=radius,dark=dark)
                result=detect_bead_plate_sequence_detailed(image,venue=venue,input_type="road_crop")
                self.assertTrue(result["ok"],result)
                self.assertEqual(result["raw_outcomes"],history)
                self.assertEqual(result["sequence"],[v for v in history if v!="T"])
                self.assertEqual(result["confirmed_round_count"],len(history))
                self.assertEqual(result["method"],"bead_plate_hough_6row_v1")
                self.assertEqual(result["chronology"],"column_major_top_to_bottom")

    def test_reject_missing_middle_cell_without_shifting_history(self):
        history=list('BPTBBT')*3
        image=make_plate(history)
        center=(int(33*.55+33),int(33*.55+33*3))
        cv2.circle(image,center,16,(250,250,250),-1)
        result=detect_bead_plate_sequence_detailed(image,venue="DG",input_type="road_crop")
        self.assertFalse(result["ok"])
        self.assertEqual(result.get("raw_outcomes"),[])

    def test_mt_contrast_enhancement_hough_false_columns_recovered(self):
        """An MT white-panel numeral creates two off-column Hough candidates.

        Do not promote the false Hough x-centers into real columns, and do not
        guess any B/P/T outcomes: re-sample original HSV only on verified grid.
        """
        from unittest.mock import patch
        import bead_plate_detector as detector
        history=list("BBPPTP")*5+list("BPTBP")
        image=make_plate(history,pitch=34,radius=13,dark=False)
        pitch=34
        points=[]
        for index,value in enumerate(history):
            col,row=divmod(index,6)
            cx=34*.55+col*pitch
            cy=34*.55+row*pitch
            if (col,row) in {(2,4),(4,4)}:
                cx-=10
            points.append([cx,cy,13.0])
        mock_circles=np.asarray([points],dtype=np.float32)
        with patch.object(detector.cv2,"HoughCircles",return_value=mock_circles):
            result=detector.detect_bead_plate_sequence_detailed(
                image,venue="MT",input_type="road_crop"
            )
        self.assertTrue(result["ok"],result)
        self.assertEqual(result["raw_outcomes"],history)
        self.assertEqual(result["confirmed_round_count"],len(history))

    def test_mobile_road_crop_session_does_not_disable_mt_db_roi(self):
        """Persisted screenshot input_type must not crop the entire phone UI."""
        import bead_plate_detector as detector
        phone=np.zeros((1280,720,3),dtype=np.uint8)
        for venue in ("MT","DB"):
            candidates=list(detector._rois(venue,phone,"road_crop"))
            self.assertGreaterEqual(len(candidates),5)
            self.assertNotEqual(candidates[0][0],"bead_plate_crop")
            self.assertIn(venue.lower(),candidates[0][0])
        # Keep DG's previously working behavior exactly as before.
        self.assertEqual(list(detector._rois("DG",phone,"road_crop"))[0][0],
                         "bead_plate_crop")

    def test_no_big_road_fallback_on_unrelated_background(self):
        image=np.full((240,480,3),180,dtype=np.uint8)
        result=detect_bead_plate_sequence_detailed(image,venue="DB",input_type="road_crop")
        self.assertFalse(result["ok"])
        self.assertEqual(result.get("sequence"),[])

    def test_screen_pipeline_imports_only_bead_plate_source(self):
        from pathlib import Path
        source=Path("screen_pipeline.py").read_text(encoding="utf-8")
        self.assertIn(
            "from bead_plate_detector import detect_bead_plate_sequence_detailed as detect_road_sequence_detailed",
            source
        )
        self.assertNotIn("from road_detector_mt_ofalive_profile import detect_road_sequence_detailed",source)
        # Core, OCR, LINE/UID, model files are not changed by this feature.
        self.assertIn("def analyze_game_screen(",source)
        self.assertIn("and reconstructed_all",source)


if __name__=="__main__":
    unittest.main()
