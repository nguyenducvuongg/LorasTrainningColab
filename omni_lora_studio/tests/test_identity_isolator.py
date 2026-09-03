import unittest
from omni_lora.dataset.captioning.identity_isolator import IdentityIsolator

class TestIdentityIsolator(unittest.TestCase):
    def test_identity_isolator_removes_invariant_facial_traits(self):
        raw_caption = "A smiling woman with blonde hair and blue eyes, wearing a red jacket, standing in a sunny park"
        purified = IdentityIsolator.purify_caption(raw_caption, trigger_word="sks", class_word="woman")

        self.assertNotIn("blonde hair", purified.lower())
        self.assertNotIn("blue eyes", purified.lower())
        self.assertIn("red jacket", purified.lower())
        self.assertIn("park", purified.lower())
        self.assertTrue(purified.startswith("sks woman"))

    def test_identity_isolator_empty_caption(self):
        purified = IdentityIsolator.purify_caption("", trigger_word="mytoken", class_word="girl")
        self.assertEqual(purified, "mytoken girl")

if __name__ == "__main__":
    unittest.main()
