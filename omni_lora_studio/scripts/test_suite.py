#!/usr/bin/env python3
import subprocess
import sys

def run_tests():
    print("🧪 [OmniLoRA Studio] Đang chạy toàn bộ kiểm thử đơn vị...")
    res = subprocess.run(["pytest", "tests/", "-v"])
    sys.exit(res.returncode)

if __name__ == "__main__":
    run_tests()
