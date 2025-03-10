import wandb
import yaml
import subprocess
import os
from pathlib import Path

# 현재 스크립트 위치 기준으로 경로 설정
current_dir = Path(__file__).parent.absolute()
sweep_config_path = current_dir / 'conf' / 'sweep_config.yaml'

# sweep 설정 파일 로드
with open(sweep_config_path, 'r') as f:
    sweep_config = yaml.safe_load(f)

# sweep 초기화
sweep_id = wandb.sweep(sweep_config, project="test_sweep")

# sweep 에이전트 실행 함수
def run_sweep():
    # sweep 모드로 설정 변경
    subprocess.run(
        ["python", "main.py", "wandb.sweep=True"],
        cwd=current_dir  # 작업 디렉토리 설정
    )

# sweep 에이전트 실행 (count는 실행할 실험 횟수)
wandb.agent(sweep_id, function=run_sweep, count=10)

if __name__ == "__main__":
    run_sweep()