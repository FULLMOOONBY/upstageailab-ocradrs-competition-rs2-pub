import os
import numpy as np
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from utils.utils import set_seed  # 전체 경로 사용

def run_als(cfg):
    """
    ALS 모델 학습 및 추천 결과 생성 함수.
    
    Args:
        cfg: 전체 설정 객체
        
    Returns:
        추천 결과가 담긴 DataFrame
    """
    # 설정 추출
    als_cfg = cfg.model.als
    common_cfg = cfg.common
    
    # 데이터 로드 및 전처리
    train_df = pd.read_csv(os.path.join(common_cfg.data_dir, common_cfg.data_file))
    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}
    
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)
    train_df["label"] = 1
    user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()
    
    sparse_user_item = sparse.csr_matrix(
        (user_item_matrix["label"].values,
         (user_item_matrix["user_idx"].values, user_item_matrix["item_idx"].values)),
        shape=(len(user2idx), len(item2idx)),
        dtype=np.float32
    ).tocsr()
    
    # 모델 학습
    model = AlternatingLeastSquares(
        factors=als_cfg.num_factor,
        regularization=als_cfg.regularization,
        alpha=als_cfg.alpha,
        use_gpu=als_cfg.use_gpu
    )
    model.fit(sparse_user_item)
    
    # 평가 대상 유저에 대한 추천
    sample_df = pd.read_csv(os.path.join(common_cfg.data_dir, common_cfg.sample_submission))
    test_users = sample_df['user_id'].unique()
    test_users_idx = [user2idx[user] for user in test_users if user in user2idx]
    
    public_outputs = model.recommend(
        test_users_idx,
        sparse_user_item[test_users_idx],
        N=cfg.recommendation.top_n,
        filter_already_liked_items=False
    )
    recommend_items = public_outputs[0]
    
    # 추천 결과 DataFrame 생성 및 원래의 item_id 복원
    sub_df = pd.DataFrame({
        'user_id': np.repeat(test_users, cfg.recommendation.top_n),
        'item_id': recommend_items.flatten()
    })
    sub_df['item_id'] = sub_df['item_id'].map(idx2item)
    
    print("ALS training and recommendation completed.")
    return sub_df