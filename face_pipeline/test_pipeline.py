from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from face_pipeline.classifier import DeepfakeClassifier, label_from_real_score, preprocess_for_model
from face_pipeline.crop_sweep import resize_with_padding, should_preserve_original
from face_pipeline.cropper import CropConfig, crop_face, square_crop_box
from face_pipeline.detector import clamp_box, sort_detections
from face_pipeline.pipeline import DeepfakeFacePipeline


class FakeModel:
    def __init__(self, prediction: np.ndarray) -> None:
        self.prediction = prediction

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        return self.prediction


class StubDetector:
    def __init__(self, detections: list[dict]) -> None:
        self.detections = detections

    def detect(self, image):
        return self.detections


class StubClassifier:
    def __init__(self) -> None:
        self.calls = 0

    def predict_image(self, image):
        self.calls += 1
        return {"real_score": 0.72, "fake_score": 0.28, "label": "Likely Real"}


class FacePipelineUnitTests(unittest.TestCase):
    def test_bounding_box_clamping(self) -> None:
        box = clamp_box({"x1": -4, "y1": 3.2, "x2": 101.6, "y2": 80}, image_width=100, image_height=70)
        self.assertEqual(box, {"x1": 0, "y1": 3, "x2": 100, "y2": 70})

    def test_square_crop_calculation(self) -> None:
        crop = square_crop_box({"x1": 40, "y1": 50, "x2": 140, "y2": 170}, 256, 256, margin=0.20)
        self.assertEqual(crop["x2"] - crop["x1"], crop["y2"] - crop["y1"])
        self.assertGreaterEqual(crop["x1"], 0)
        self.assertLessEqual(crop["x2"], 256)

    def test_margin_calculation(self) -> None:
        crop = square_crop_box({"x1": 80, "y1": 80, "x2": 180, "y2": 180}, 300, 300, margin=0.20)
        self.assertEqual(crop["x2"] - crop["x1"], 140)

    def test_tiny_face_rejection(self) -> None:
        image = Image.new("RGB", (50, 50), color="white")
        with self.assertRaises(ValueError):
            crop_face(image, {"x1": 10, "y1": 10, "x2": 20, "y2": 20}, CropConfig(minimum_crop_size=64))

    def test_no_face_response_does_not_classify(self) -> None:
        classifier = StubClassifier()
        pipeline = DeepfakeFacePipeline(detector=StubDetector([]), classifier=classifier)
        result = pipeline.analyze(Image.new("RGB", (128, 128)), image_name="blank.jpg")
        self.assertEqual(result["status"], "no_face_detected")
        self.assertEqual(result["faces_detected"], 0)
        self.assertEqual(classifier.calls, 0)

    def test_multiple_face_ordering(self) -> None:
        detections = [
            {"x1": 90, "y1": 5, "x2": 120, "y2": 40},
            {"x1": 10, "y1": 50, "x2": 40, "y2": 80},
            {"x1": 10, "y1": 5, "x2": 40, "y2": 35},
        ]
        ordered = sort_detections(detections)
        self.assertEqual([(d["face_index"], d["x1"], d["y1"]) for d in ordered], [(0, 10, 5), (1, 10, 50), (2, 90, 5)])

    def test_label_threshold_boundaries(self) -> None:
        self.assertEqual(label_from_real_score(0.399), "Likely Fake")
        self.assertEqual(label_from_real_score(0.400), "Uncertain")
        self.assertEqual(label_from_real_score(0.600), "Uncertain")
        self.assertEqual(label_from_real_score(0.601), "Likely Real")

    def test_score_conversion(self) -> None:
        classifier = DeepfakeClassifier(model=FakeModel(np.array([[0.25]], dtype=np.float32)))
        prediction = classifier.predict_image(Image.new("RGB", (32, 32), color="white"))
        self.assertAlmostEqual(prediction["real_score"], 0.25, places=6)
        self.assertAlmostEqual(prediction["fake_score"], 0.75, places=6)
        self.assertEqual(prediction["label"], "Likely Fake")

    def test_unexpected_model_output(self) -> None:
        classifier = DeepfakeClassifier(model=FakeModel(np.array([[0.2, 0.8]], dtype=np.float32)))
        with self.assertRaises(ValueError):
            classifier.predict_image(Image.new("RGB", (32, 32), color="white"))

    def test_rgb_conversion(self) -> None:
        gray = np.full((10, 10), 127, dtype=np.uint8)
        batch = preprocess_for_model(gray)
        self.assertEqual(batch.shape, (1, 256, 256, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertTrue(np.allclose(batch[:, :, :, 0], batch[:, :, :, 1]))
        self.assertTrue(np.allclose(batch[:, :, :, 1], batch[:, :, :, 2]))

    def test_prepared_face_bypass_requires_geometry_not_just_256(self) -> None:
        centered = [{"x1": 45, "y1": 55, "x2": 215, "y2": 225}]
        decision = should_preserve_original((256, 256), centered)
        self.assertTrue(decision["preserve_original"])
        self.assertIn("image_is_256x256", decision["reasons"])

        tiny_corner = [{"x1": 5, "y1": 5, "x2": 40, "y2": 40}]
        decision = should_preserve_original((256, 256), tiny_corner)
        self.assertFalse(decision["preserve_original"])
        self.assertIn("face_occupancy_below_threshold", decision["blockers"])

    def test_reflected_and_edge_padding_keep_256_square(self) -> None:
        image = Image.new("RGB", (180, 90), color="white")
        reflected, reflected_pad = resize_with_padding(image, "reflect")
        edged, edge_pad = resize_with_padding(image, "edge")
        self.assertEqual(reflected.size, (256, 256))
        self.assertEqual(edged.size, (256, 256))
        self.assertGreater(reflected_pad["top"] + reflected_pad["bottom"], 0)
        self.assertEqual(reflected_pad, edge_pad)


if __name__ == "__main__":
    unittest.main()
