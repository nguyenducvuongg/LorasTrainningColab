import unittest
import numpy as np
from omni_lora.validation.likeness_meter import LikenessMeter

class TestLikenessMeter(unittest.TestCase):
    def test_cosine_similarity(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(LikenessMeter.compute_cosine_similarity(v1, v2), 1.0)

        v3 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(LikenessMeter.compute_cosine_similarity(v1, v3), 0.0)

    def test_likeness_meter_score_calculation(self):
        meter = LikenessMeter()
        v1 = np.random.randn(512).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        sim = meter.compute_cosine_similarity(v1, v1)
        self.assertAlmostEqual(sim, 1.0, places=3)

if __name__ == "__main__":
    unittest.main()
