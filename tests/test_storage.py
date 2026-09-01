import pytest
import os
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lora_colab.storage.gdrive_manager import GDriveWorkspaceManager
from lora_colab.storage.model_downloader import MODEL_REGISTRY
from lora_colab.storage.resume_manager import ResumeManager

def test_workspace_init_and_scan():
    with tempfile.TemporaryDirectory() as tmp_root:
        # 1. First init (creates directories)
        paths = GDriveWorkspaceManager.init_workspace(tmp_root)
        assert os.path.exists(os.path.join(tmp_root, "models", "flux"))
        assert os.path.exists(os.path.join(tmp_root, "datasets", "01_face"))
        assert os.path.exists(os.path.join(tmp_root, "datasets", "02_character"))
        assert os.path.exists(os.path.join(tmp_root, "outputs", "final_loras"))

        # Create dummy model file
        dummy_model = os.path.join(tmp_root, "models", "flux", "test_model.safetensors")
        with open(dummy_model, "wb") as f:
            f.write(b"0" * 1024)

        # 2. Second init (scans without overwriting)
        paths2 = GDriveWorkspaceManager.init_workspace(tmp_root)
        assert os.path.exists(dummy_model)

        # 3. Scan models
        found = GDriveWorkspaceManager.scan_existing_models(tmp_root)
        assert len(found) == 1
        assert found[0]["filename"] == "test_model.safetensors"

def test_model_registry():
    assert "flux-dev" in MODEL_REGISTRY
    assert "flux-kontext" in MODEL_REGISTRY
    assert "krea2-raw" in MODEL_REGISTRY
    assert "pony-v6" in MODEL_REGISTRY
    assert "sd15-base" in MODEL_REGISTRY

def test_resume_manager():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # No checkpoints initially
        res = ResumeManager.get_resume_status(tmp_dir)
        assert not res["can_resume"]

        # Create fake checkpoint
        ckpt = os.path.join(tmp_dir, "my_lora-step00000500.safetensors")
        with open(ckpt, "w") as f:
            f.write("test")

        res = ResumeManager.get_resume_status(tmp_dir)
        assert res["can_resume"]
        assert res["step"] == 500
