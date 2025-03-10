import pandas as pd
import os
import numpy as np
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)

def load_and_preprocess_data(cfg: DictConfig):
    """
    데이터 로드 및 전처리
    
    Args:
        cfg: 설정 객체
        
    Returns:
        전처리된 데이터프레임, 매핑 딕셔너리
    """
    data_path = os.path.join(cfg.common.data_dir, cfg.common.data_file)
    logger.info(f"데이터 로드 중: {data_path}")
    
    # 데이터 로드
    train_df = pd.read_csv(data_path)
    
    # event_time을 datatime으로 변환
    # train_df['event_time'] = pd.to_datetime(train_df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
    train_df = train_df.sort_values(by=['event_time'])
    train_df = train_df[['user_id','item_id','user_session','event_time']]
    train_df['event_time'] = train_df['event_time'].values.astype(float)
    
    # 사용자(user)와 아이템(item)을 인덱스로 매핑하기 위한 딕셔너리 생성
    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}
    
    # 사용자와 아이템을 인덱스로 변환하여 새로운 열 추가
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)
    train_df = train_df.dropna().reset_index(drop=True)
    
    # RecBole 형식으로 변환
    recbole_df = train_df.rename(columns={
        'user_idx': 'user_idx:token', 
        'item_idx': 'item_idx:token', 
        'event_time': 'event_time:float'
    })
    
    # 데이터셋 디렉토리 생성
    os.makedirs(cfg.common.output_dir, exist_ok=True)
    
    # 데이터 저장
    dataset_path = os.path.join(cfg.common.output_dir, 'SASRec_dataset.inter')
    recbole_df[['user_idx:token', 'item_idx:token', 'event_time:float']].to_csv(
        dataset_path, sep='\t', index=None
    )
    
    logger.info(f"데이터 전처리 완료: {len(train_df)} 행")
    
    return train_df, {
        'user2idx': user2idx,
        'idx2user': idx2user,
        'item2idx': item2idx,
        'idx2item': idx2item
    }