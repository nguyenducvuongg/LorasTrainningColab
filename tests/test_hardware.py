import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lora_colab.core.hardware import HardwareProfiler, GPUProfile

def test_hardware_profiler():
    profile = HardwareProfiler.detect_and_profile("flux")
    assert isinstance(profile, GPUProfile)
    assert profile.tier in ["T4", "L4", "A100", "V100", "CPU", "GENERIC_CUDA"]
    assert profile.recommended_batch_size >= 1
    assert profile.precision in ["fp8", "fp16", "bf16", "fp32"]

def test_hardware_profiler_models():
    for model_name in ["flux", "sdxl", "pony", "sd15", "krea"]:
        profile = HardwareProfiler.detect_and_profile(model_name)
        assert profile is not None
