import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from model.sasrec.train_sasrec import run_sasrec
from model.als.train_als import run_als
from utils.metric import cal_ndcg
from utils.utils import set_seed
import wandb

# 로거 설정
log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    메인 실행 함수
    
    Args:
        cfg: Hydra 설정
    """
    # wandb sweep 모드인 경우 설정 업데이트
    if cfg.wandb.use and cfg.wandb.get('sweep', False):
        run = wandb.init()
        # wandb sweep 파라미터로 설정 업데이트
        wandb_config = wandb.config
        for key, value in wandb_config.items():
            if '.' in key:  # 중첩된 설정 (예: model.als.num_factor)
                parts = key.split('.')
                cfg_part = cfg
                for part in parts[:-1]:
                    cfg_part = cfg_part[part]
                cfg_part[parts[-1]] = value
            else:  # 최상위 설정
                cfg[key] = value
        log.info("Updated config with wandb sweep")
    
    # 재현성을 위해 seed 설정
    set_seed(cfg.common.seed)
    
    # 모델 타입에 따라 적절한 학습 진행 및 추천 결과 생성
    if cfg.model.type.lower() == "sasrec":
        log.info("SASRec 학습 시작...")
        model_specific_config = {
            "model_type": "sasrec",
            "sasrec": OmegaConf.to_container(cfg.model.sasrec, resolve=True),
            "recommendation": OmegaConf.to_container(cfg.recommendation, resolve=True)
        }
        recommendations_df = run_sasrec(cfg)
    elif cfg.model.type.lower() == "als":
        log.info("ALS 학습 시작...")
        model_specific_config = {
            "model_type": "als",
            "als": OmegaConf.to_container(cfg.model.als, resolve=True),
            "recommendation": OmegaConf.to_container(cfg.recommendation, resolve=True)
        }
        recommendations_df = run_als(cfg)  # 전체 설정 객체 전달
    else:
        log.error(f"지원하지 않는 모델: {cfg.model.type}")
        return
    
    # 일반 실행 모드에서 wandb 초기화 (sweep 모드에서는 이미 초기화된 상태)
    if cfg.wandb.use and not cfg.wandb.get('sweep', False):
        wandb.init(
            project=cfg.wandb.project,
            config=model_specific_config,
            dir=cfg.wandb.dir  # 결과파일 저장되는 공간간
        )
    
    # 결과 파일 경로
    submission_path = Path(cfg.common.output_dir) / cfg.common.output_file

    # 결과 저장
    recommendations_df.to_csv(submission_path, index=False)
    log.info(f"추천 결과 저장 완료: {submission_path}, {len(recommendations_df)} 행")

    # 검증 데이터가 있는 경우 평가 지표 계산
    valid_path = os.path.join(cfg.common.data_dir, "valid_ts7.csv")
    
    if valid_path and os.path.exists(valid_path):
        try:
            ndcg = cal_ndcg(recommendations_df, str(valid_path))
            log.info(f"NDCG: {ndcg:.4f}")
            log.info("NDCG: 성지님 Module 사용 완료")
            
            # Wandb에 metric 기록 (sweep 모드와 일반 모드 모두)
            if cfg.wandb.use:
                wandb.log({"ndcg": ndcg})
        except ImportError:
            log.warning("평가 지표 계산을 위한 모듈을 찾을 수 없습니다.")
        except Exception as e:
            log.error(f"평가 지표 계산 중 오류 발생: {e}")
    
    # Wandb 종료 (설정된 경우, sweep 모드가 아니라면)
    if cfg.wandb.use and not cfg.wandb.get('sweep', False):
        wandb.finish()

    log.info("작업 완료!")

if __name__ == "__main__":
    main()
