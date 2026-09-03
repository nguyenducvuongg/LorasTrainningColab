import unittest
from omni_lora.core.hardware import HardwareProfiler, GPUProfile

class TestHardware(unittest.TestCase):
    def test_hardware_profiler(self):
        profile = HardwareProfiler.analyze("flux-dev")
        self.assertIsInstance(profile, GPUProfile)
        self.assertGreaterEqual(profile.recommended_batch_size, 1)
        self.assertIn(profile.hardware_tier, ["T4_FREE", "L4_PRO", "A100_PRO", "CONSUMER", "CPU"])

if __name__ == "__main__":
    unittest.main()
