import subprocess
import sys
from pathlib import Path


def run(script: str) -> None:
    here = Path(__file__).resolve().parent
    print(f"\n===== Running {script} =====")
    subprocess.run([sys.executable, str(here / script)], check=True, cwd=here)


if __name__ == "__main__":
    run("data_processing.py")
    run("train_depthdiff.py")
    run("train_baselines.py")
    run("evaluate.py")
