import pandas as pd
from omegaconf import DictConfig
import logging
from tqdm import tqdm
from collections import defaultdict
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start.quick_start import load_data_and_model
from .data_loader import load_and_preprocess_data

logger = logging.getLogger(__name__)

def run_sasrec(cfg: DictConfig):
    """
    SASRec 모델 실행 함수
    
    Args:
        cfg: 설정 객체
        
    Returns:
        추천 결과가 담긴 DataFrame
    """
    # 데이터 로드 및 전처리
    train_df, mappings = load_and_preprocess_data(cfg)
    
    # RecBole 설정 생성
    recbole_config = create_recbole_config(cfg)
    
    # 데이터셋 생성
    init_seed(recbole_config['seed'], recbole_config['reproducibility'])
    dataset = create_dataset(recbole_config)
    train_data, valid_data, _ = data_preparation(recbole_config, dataset)
    
    # 모델 생성
    model = SASRec(recbole_config, train_data.dataset).to(recbole_config['device'])
    logger.info(f"모델 정보: {model}")
    
    # 학습
    if cfg.model.model_path is None:
        trainer = Trainer(recbole_config, model)
        trainer.fit(train_data, valid_data, saved=True, show_progress=True)
        model_path = trainer.saved_model_file
        logger.info(f"모델 저장 경로: {model_path}")
    else:
        model_path = cfg.model.model_path
        logger.info(f"사전 학습된 모델 로드: {model_path}")
    
    # 추론
    recommendations_df = generate_recommendations(cfg, train_df, mappings, model_path, dataset)
    
    return recommendations_df

def create_recbole_config(cfg: DictConfig):
    """
    RecBole 설정 생성
    
    Args:
        cfg: Hydra 설정 객체
        
    Returns:
        RecBole 설정 객체
    """

    # 설정 딕셔너리 생성
    config_dict = {
        'data_path': './',
        'USER_ID_FIELD': cfg.model.sasrec.data.USER_ID_FIELD,
        'ITEM_ID_FIELD': cfg.model.sasrec.data.ITEM_ID_FIELD,
        'TIME_FIELD': cfg.model.sasrec.data.TIME_FIELD,
        'user_inter_num_interval': cfg.model.sasrec.data.user_inter_num_interval,
        'item_inter_num_interval': cfg.model.sasrec.data.item_inter_num_interval,
        'load_col': cfg.model.sasrec.data.load_col,
        
        'train_batch_size': cfg.model.sasrec.model.train_batch_size,
        'hidden_size': cfg.model.sasrec.model.hidden_size,
        'n_layers': cfg.model.sasrec.model.n_layers,
        'n_heads': cfg.model.sasrec.model.n_heads,
        'inner_size': cfg.model.sasrec.model.inner_size,
        'hidden_dropout_prob': cfg.model.sasrec.model.hidden_dropout_prob,
        'attn_dropout_prob': cfg.model.sasrec.model.attn_dropout_prob,
        'hidden_act': cfg.model.sasrec.model.hidden_act,
        'layer_norm_eps': cfg.model.sasrec.model.layer_norm_eps,
        'initializer_range': cfg.model.sasrec.model.initializer_range,
        'pooling_mode': cfg.model.sasrec.model.pooling_mode,
        'loss_type': cfg.model.sasrec.model.loss_type,
        'fusion_type': cfg.model.sasrec.model.fusion_type,
        'attribute_predictor': cfg.model.sasrec.model.attribute_predictor,
        'epoch': cfg.model.sasrec.model.epoch,
        'stopping_step': cfg.model.sasrec.model.stopping_step,
        
        'MAX_ITEM_LIST_LENGTH': cfg.model.sasrec.model.MAX_ITEM_LIST_LENGTH,
        'eval_args': cfg.model.sasrec.model.eval_args,
        'metrics': cfg.model.sasrec.model.metrics,
        'topk': cfg.model.sasrec.model.topk,
        'valid_metric': cfg.model.sasrec.model.valid_metric,
        'checkpoint_dir': cfg.model.sasrec.model.checkpoint_dir
    }
    
    # RecBole 설정 객체 생성
    return Config(model='SASRec', config_dict=config_dict, dataset='SASRec_dataset')

def generate_recommendations(cfg, train_df, mappings, model_path, dataset):
    """
    추천 생성 및 저장
    
    Args:
        cfg: 설정 객체
        train_df: 학습 데이터
        mappings: ID 매핑 딕셔너리
        model_path: 모델 경로
        dataset: RecBole 데이터셋
        
    Returns:
        추천 결과가 담긴 DataFrame
    """
    logger.info("추천 생성 중...")
    
    # 사용자별 아이템 목록 생성
    train_df = train_df.sort_values(by=['user_session','event_time'])
    users = defaultdict(list)
    for u, i in zip(train_df['user_idx'], train_df['item_idx']):
        users[u].append(i)
    
    # 모델 로드
    config, model, dataset, _, _, test_data = load_data_and_model(model_file=model_path)
    logger.info('데이터 및 모델 로드 완료')
    
    # 인기 아이템 (콜드 스타트 사용자용)
    popular_top_10 = train_df.groupby('item_idx').count().rename(
        columns={"user_idx": "user_counts"}
    ).sort_values(
        by=['user_counts', 'item_idx'], ascending=[False, True]
    )[:cfg.recommendation.top_n].index
    
    # 결과 저장용 리스트
    result = []
    
    # 사용자별 추천 생성
    for uid in tqdm(users, desc="사용자 처리 중"):
        if str(uid) in dataset.field2token_id['user_idx']:
            # 모델 기반 추천
            recbole_id = dataset.token2id(dataset.uid_field, str(uid))
            topk_score, topk_iid_list = full_sort_topk(
                [recbole_id], model, test_data, 
                k=cfg.recommendation.top_n, 
                device=config['device']
            )
            predicted_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            predicted_item_list = predicted_item_list[-1]
            predicted_item_list = list(map(int, predicted_item_list))
        else:
            # 콜드 스타트 사용자는 인기 아이템 추천
            predicted_item_list = list(popular_top_10)
        
        # 결과 저장
        for iid in predicted_item_list:
            result.append((mappings['idx2user'][uid], mappings['idx2item'][iid]))
    
    # DataFrame 생성 및 반환
    recommendations_df = pd.DataFrame(result, columns=["user_id", "item_id"])
    
    logger.info(f"추천 결과 생성 완료: {len(recommendations_df)} 행")
    return recommendations_df