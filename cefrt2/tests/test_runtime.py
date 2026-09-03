import os
import unittest
from pathlib import Path

import numpy as np
import torch

from cefrt2 import host_ops
from cefrt2.geometry import mask_to_quad_ransac, order_quad
from cefrt2.runtime import ChekiEdgeFitRT, PACKAGE, THRESHOLD


class GeometryTests(unittest.TestCase):
    def test_letterbox_roundtrip(self):
        rgb = np.zeros((300, 500, 3), np.uint8)
        image, scale, left, top = host_ops.letterbox_image(rgb, 768)
        self.assertEqual(image.shape, (768, 768, 3))
        original = np.asarray([30, 20, 400, 250], np.float32)
        normalized = (original * np.float32(scale) + np.asarray([left, top, left, top], np.float32)) / np.float32(768)
        np.testing.assert_allclose(host_ops.restore_original_box(normalized, 500, 300), original, atol=4e-5, rtol=0)

    def test_g_rectangle(self):
        mask = np.zeros((128, 128), bool)
        mask[20:108, 25:103] = True
        quad = order_quad(mask_to_quad_ransac(mask))
        self.assertEqual(quad.shape, (4, 2))
        self.assertTrue(np.isfinite(quad).all())

    def test_square20_not_clipped(self):
        roi = host_ops.square20_fp32(np.asarray([0, 0, 100, 50], np.float32))
        np.testing.assert_array_equal(roi, [-20, -45, 120, 95])
        self.assertEqual(THRESHOLD, 0.9267578125)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weights = Path(os.environ.get("CEFRT2_WEIGHTS_DIR", str(PACKAGE / "checkpoints")))
        if not (weights / "cefrt2-rtdetr-r18-epoch24-fp32.pt").is_file():
            raise unittest.SkipTest("Download the release checkpoints first")
        torch.set_num_threads(2)
        cls.model = ChekiEdgeFitRT(weights, device="cpu")

    def test_raw_shapes_and_dtypes(self):
        with torch.inference_mode():
            output = self.model.detector(torch.zeros((1, 3, 768, 768)))
        self.assertEqual(tuple(output["pred_logits"].shape), (1, 300, 1))
        self.assertEqual(tuple(output["pred_boxes"].shape), (1, 300, 4))
        self.assertEqual(output["pred_logits"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(output["pred_boxes"]).all())

    def test_synthetic_box_path(self):
        image = np.full((240, 320, 3), 128, np.float32)
        result = self.model.segment_box(image, [40, 30, 270, 210])
        self.assertTrue(np.isfinite(result["edge_sam_quality_score"]))
        if result["quad"] is not None:
            self.assertEqual(np.asarray(result["quad"]).shape, (4, 2))
            self.assertTrue(np.isfinite(result["quad"]).all())
        else:
            self.assertIsInstance(result["error"], str)


if __name__ == "__main__":
    unittest.main()
