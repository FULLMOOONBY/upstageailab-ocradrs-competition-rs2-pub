import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time
import pickle
import multiprocessing
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_data(file_path):
    """데이터 로드"""
    df = pd.read_parquet(file_path)
    print("📂 데이터 로드 완료!")
    print(f"🔹 데이터 크기: {df.shape}")
    return df

def create_index_mappings(df):
    """사용자와 아이템을 인덱스로 변환하는 매핑 생성"""
    user2idx = {v: k for k, v in enumerate(df['user_id'].unique())}
    idx2user = {k: v for v, k in user2idx.items()}  # 역매핑 생성

    item2idx = {v: k for k, v in enumerate(df['item_id'].unique())}
    idx2item = {k: v for v, k in item2idx.items()}  # 역매핑 생성

    # 인덱스 변환
    df['user_idx'] = df['user_id'].map(user2idx)
    df['item_idx'] = df['item_id'].map(item2idx)
    
    return df, user2idx, idx2user, item2idx, idx2item

def prepare_features(df):
    """특성 엔지니어링 및 전처리"""
    df_prep = df.copy()
    
    # 시간 관련 특성
    df_prep['event_time'] = pd.to_datetime(df_prep['event_time'])
    df_prep['hour'] = df_prep['event_time'].dt.hour
    df_prep['day_of_week'] = df_prep['event_time'].dt.dayofweek
    
    # 범주형 변수 인코딩
    le = LabelEncoder()
    categorical_cols = ['category_code', 'brand', 'event_type']
    for col in categorical_cols:
        if col in df_prep.columns:
            df_prep[f'{col}_encoded'] = le.fit_transform(df_prep[col])
    
    # 가격 관련 특성
    if 'price' in df_prep.columns:
        df_prep['price_rank'] = df_prep['price'].rank(method='dense')
        df_prep['price_bin'] = pd.qcut(df_prep['price'], q=10, labels=False)
    
    # 상호작용 집계 특성
    user_counts = df_prep.groupby('user_idx').size().reset_index(name='user_interaction_count')
    item_counts = df_prep.groupby('item_idx').size().reset_index(name='item_interaction_count')
    
    if 'category_code_encoded' in df_prep.columns:
        category_counts = df_prep.groupby('category_code_encoded').size().reset_index(name='category_interaction_count')
        df_prep = df_prep.merge(category_counts, on='category_code_encoded', how='left')
    
    if 'brand_encoded' in df_prep.columns:
        brand_counts = df_prep.groupby('brand_encoded').size().reset_index(name='brand_interaction_count')
        df_prep = df_prep.merge(brand_counts, on='brand_encoded', how='left')
    
    # 병합
    df_prep = df_prep.merge(user_counts, on='user_idx', how='left')
    df_prep = df_prep.merge(item_counts, on='item_idx', how='left')
    
    return df_prep

def split_train_valid_test(df_prep, valid_ratio=0.1, test_ratio=0.1):
    """시간 기반 train/validation/test 분할"""
    df_prep = df_prep.sort_values('event_time', ascending=True)
    
    # 분할 지점 계산
    test_split_idx = int((1 - test_ratio) * len(df_prep))
    valid_split_idx = int((1 - test_ratio - valid_ratio) * len(df_prep))
    
    # 데이터 분할
    train_df = df_prep.iloc[:valid_split_idx]
    valid_df = df_prep.iloc[valid_split_idx:test_split_idx]
    test_df = df_prep.iloc[test_split_idx:]
    
    # 검증/테스트셋은 구매 데이터만 사용
    valid_purchase = valid_df[valid_df['event_type'] == 'purchase']
    test_purchase = test_df[test_df['event_type'] == 'purchase']
    
    valid_dict = valid_purchase.groupby('user_idx')['item_idx'].agg(list).to_dict()
    test_dict = test_purchase.groupby('user_idx')['item_idx'].agg(list).to_dict()
    
    return train_df, valid_df, test_df, valid_dict, test_dict

def create_train_dataset(df_prep, feature_columns=None):
    """학습용 데이터셋 생성"""
    if feature_columns is None:
        # 기본 피처 목록
        feature_columns = [
            'user_idx', 'item_idx',
            'category_code_encoded', 'brand_encoded',
            'price', 'price_rank', 'price_bin',
            'hour', 'day_of_week',
            'user_interaction_count', 'item_interaction_count',
            'category_interaction_count', 'brand_interaction_count'
        ]
        
        # 존재하는 컬럼만 선택
        feature_columns = [col for col in feature_columns if col in df_prep.columns]
    
    X = df_prep[feature_columns]
    y = (df_prep['event_type'] == 'purchase').astype(int)
    
    return X, y, feature_columns

def get_model_params(cfg):
    """LightGBM 모델 파라미터 정의"""
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': cfg.model.lightgbm.boosting_type,
        'learning_rate': cfg.model.lightgbm.learning_rate,
        'num_leaves': cfg.model.lightgbm.num_leaves,
        'max_depth': cfg.model.lightgbm.max_depth,
        'feature_fraction': cfg.model.lightgbm.feature_fraction,
        'bagging_fraction': cfg.model.lightgbm.bagging_fraction,
        'bagging_freq': cfg.model.lightgbm.bagging_freq,
        'verbose': -1,
        'device_type': cfg.model.lightgbm.device_type,
        'n_jobs': cfg.model.lightgbm.n_jobs if cfg.model.lightgbm.n_jobs > 0 else multiprocessing.cpu_count(),
        'early_stopping_rounds': cfg.model.lightgbm.early_stopping_rounds
    }
    return params

def get_feature_importance(model, feature_names):
    """LightGBM 모델의 특성 중요도 출력"""
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importance(importance_type='gain')
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    return importance_df

def train_lightgbm_model(X_train, y_train, X_valid, y_valid, cfg):
    """LightGBM 모델 학습"""
    print("모델 학습 시작...")
    start_time = time.time()
    
    # 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    # 모델 파라미터 설정
    params = get_model_params(cfg)
    
    # 콜백 설정
    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=params['early_stopping_rounds'])
    ]
    
    # 모델 학습
    model = lgb.train(
        params,
        train_data,
        num_boost_round=cfg.model.lightgbm.num_boost_round,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    print(f"\n모델 학습 완료! 소요 시간: {training_time:.2f}초")
    print(f"Best iteration: {model.best_iteration}")
    
    return model

def calculate_ndcg(true_items, pred_items, k=10):
    """NDCG@K 계산 (numpy 연산 최적화)"""
    if not true_items:
        return 0.0

    pred_items = np.array(pred_items[:k])
    true_items = np.array(true_items)

    ranks = np.where(np.isin(pred_items, true_items))[0] + 1
    if len(ranks) == 0:
        return 0.0
    
    dcg = np.sum(1 / np.log2(ranks + 1))
    idcg = np.sum(1 / np.log2(np.arange(1, min(len(true_items), k) + 1) + 1))

    return dcg / idcg if idcg > 0 else 0.0

def generate_recommendations(model, user_idx, df_prep, candidate_items, feature_dict, feature_columns, n_items=10):
    """특정 사용자에 대한 추천 아이템 생성 (벡터 연산 최적화)"""
    # 후보 아이템에 대한 특성 가져오기
    test_data = {
        'user_idx': [user_idx] * len(candidate_items),
        'item_idx': candidate_items
    }

    # 사전 생성된 feature_dict에서 아이템 특성 가져오기
    for col in feature_dict.keys():
        test_data[col] = [feature_dict[col].get(item, 0) for item in candidate_items]

    test_df = pd.DataFrame(test_data)
    
    # 예측 점수 계산
    scores = model.predict(test_df[feature_columns])

    # 점수가 높은 상위 N개 아이템 선택 (성능 최적화)
    top_indices = np.argpartition(scores, -n_items)[-n_items:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    
    return [candidate_items[idx] for idx in top_indices]

def evaluate_user(args):
    """병렬 실행을 위한 사용자 단위 평가"""
    model, user_idx, df_prep, test_dict, candidate_items, feature_dict, feature_columns = args
    if user_idx not in test_dict:
        return None
    
    recommended_items = generate_recommendations(model, user_idx, df_prep, candidate_items, feature_dict, feature_columns)
    true_items = test_dict[user_idx]
    return calculate_ndcg(true_items, recommended_items)

def evaluate_model(model, val_users, df_prep, test_dict, feature_columns, num_workers=16):
    """모델 성능 평가 (멀티스레딩)"""
    print("🚀 모델 평가 시작...")
    start_time = time.time()

    all_items = list(df_prep['item_idx'].unique())

    # feature_columns에서 필요한 열만 선택
    feature_cols_to_use = [col for col in feature_columns if col not in ['user_idx', 'item_idx']]
    
    feature_dict = {
        col: df_prep.set_index('item_idx')[col].to_dict() for col in feature_cols_to_use
        if col in df_prep.columns
    }

    ndcg_scores = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = list(executor.map(
            evaluate_user,
            [(model, user_idx, df_prep, test_dict, all_items, feature_dict, feature_columns) for user_idx in val_users]
        ))

    ndcg_scores = [score for score in futures if score is not None]

    mean_ndcg = np.mean(ndcg_scores)
    eval_time = time.time() - start_time
    print(f"\n✅ 평가 완료! 소요 시간: {eval_time:.2f}초")
    
    return mean_ndcg

def process_user_batch(batch_users, df_prep, feature_columns, idx2user, idx2item, top_k, model):
    """배치 단위로 사용자별 추천 생성 (최적화)"""
    results = []

    # 중복 제거 후 `set_index("item_idx")` 사용
    df_items = df_prep.drop_duplicates(subset=["item_idx"]).set_index("item_idx")

    # 후보 아이템 리스트
    candidate_items = df_items.index.to_numpy()

    # 벡터 연산을 이용하여 batch_users 전체를 한 번에 처리
    test_data = pd.DataFrame({
        "user_idx": np.repeat(batch_users, len(candidate_items)),
        "item_idx": np.tile(candidate_items, len(batch_users))
    })

    # `.map()`을 사용하여 빠르게 feature 추가
    for col in feature_columns:
        if col not in ["user_idx", "item_idx"]:
            if col in df_items.columns:
                test_data[col] = test_data["item_idx"].map(df_items[col])
            else:
                test_data[col] = 0  # 열이 없는 경우 기본값 0 설정

    # NaN 값 처리
    test_data.fillna(0, inplace=True)

    # 모델 예측 (벡터 연산 활용)
    scores = model.predict(test_data[feature_columns])
    
    scores_reshaped = scores.reshape(len(batch_users), -1)
    
    for i, user_idx in enumerate(batch_users):
        user_scores = scores_reshaped[i]
        top_indices = np.argpartition(user_scores, -top_k)[-top_k:]
        sorted_indices = top_indices[np.argsort(user_scores[top_indices])[::-1]]
        recommended_items = [candidate_items[i] for i in sorted_indices]

        results.extend([
            {"user_id": idx2user.get(user_idx, user_idx), 
             "item_id": idx2item.get(item_idx, item_idx)} 
            for item_idx in recommended_items
        ])

    return results

def generate_final_recommendations(test_users_idx, df_prep, idx2user, idx2item, feature_columns, model, cfg):
    """최종 추천 생성 (멀티스레딩 최적화)"""
    print("🚀 최적화된 추천 생성 시작...")
    start_time = time.time()

    # 체크포인트 파일 경로 설정
    checkpoint_file = os.path.join(cfg.common.output_dir, 'checkpoint.pkl')
    
    # 이전 체크포인트가 있으면 로드, 없으면 새로 생성
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "rb") as f:
                completed_users = pickle.load(f)
        except:
            completed_users = set()
    else:
        completed_users = set()

    # 처리할 남은 사용자 목록
    remaining_users = [user for user in test_users_idx if user not in completed_users]
    
    print(f"✅ 남은 사용자: {len(remaining_users)}명")

    # 결과를 저장할 데이터프레임 초기화
    submission_rows = []
    
    # 배치 크기 설정
    save_interval = cfg.memory.get('batch_size', 100)
    
    # 스레드 수 설정
    num_workers = cfg.model.lightgbm.n_jobs if cfg.model.lightgbm.n_jobs > 0 else 16
    
    # 배치별 진행상황 표시를 위한 총 배치 수 계산
    total_batches = len(remaining_users) // save_interval + (len(remaining_users) % save_interval > 0)

    # 멀티스레딩을 사용한 추천 생성
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(
                process_user_batch, 
                remaining_users[i:i + save_interval], 
                df_prep, 
                feature_columns, 
                idx2user, 
                idx2item, 
                cfg.recommendation.top_n, 
                model
            ): remaining_users[i:i + save_interval] 
            for i in range(0, len(remaining_users), save_interval)
        }

        # 완료된 작업 처리
        output_file = os.path.join(cfg.common.output_dir, cfg.common.output_file)
        for i, future in enumerate(tqdm(as_completed(future_to_batch), total=total_batches, desc="📌 진행 중")):
            try:
                result = future.result()
                if result:
                    submission_rows.extend(result)
                    completed_users.update(set(future_to_batch[future]))
            except Exception as e:
                print(f"❌ 추천 생성 실패: {e}")
                continue

            # 배치 단위로 체크포인트 저장 및 결과 파일 추가
            if (i + 1) % (save_interval // 10) == 0 or (i == total_batches - 1):
                # 체크포인트 저장
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(completed_users, f)
                
                # 결과 추가
                pd.DataFrame(submission_rows).to_csv(
                    output_file, 
                    mode="a", 
                    header=not os.path.exists(output_file), 
                    index=False
                )
                submission_rows = []
                gc.collect()  # 가비지 컬렉션 실행

    # 남은 결과 저장
    if submission_rows:
        pd.DataFrame(submission_rows).to_csv(
            output_file, 
            mode="a", 
            header=not os.path.exists(output_file), 
            index=False
        )

    print(f"\n✅ 완료! ⏳ 소요 시간: {time.time() - start_time:.2f}초")
    print(f"📁 저장된 파일: {output_file}")
    
    # 최종 추천 결과를 데이터프레임으로 반환
    return pd.read_csv(output_file)

def run_lightgbm(cfg):
    """
    LightGBM 모델 학습 및 사용자별 추천 결과 생성 함수
    
    Args:
        cfg: 전체 설정 객체 (config.yaml 기반)
    
    Returns:
        추천 결과가 담긴 DataFrame (submission 형식)
    """
    # 데이터 로드
    data_path = os.path.join(cfg.common.data_dir, cfg.common.data_file)
    df = load_data(data_path)
    
    # 인덱스 매핑 생성
    df, user2idx, idx2user, item2idx, idx2item = create_index_mappings(df)
    
    # 특성 엔지니어링
    df_prep = prepare_features(df)
    
    # 데이터셋 분할
    train_df, valid_df, test_df, valid_dict, test_dict = split_train_valid_test(
        df_prep, 
        valid_ratio=cfg.model.lightgbm.valid_ratio, 
        test_ratio=cfg.model.lightgbm.test_ratio
    )
    
    # 학습용 데이터셋 생성
    X_train, y_train, feature_columns = create_train_dataset(train_df)
    X_valid, y_valid, _ = create_train_dataset(valid_df, feature_columns)
    
    print("데이터셋 분할 완료!")
    print(f"학습 데이터 크기: {X_train.shape}")
    print(f"검증 데이터 크기: {X_valid.shape}")
    print(f"검증 사용자 수: {len(valid_dict)}")
    print(f"테스트 사용자 수: {len(test_dict)}")
    
    # 모델 학습
    model = train_lightgbm_model(X_train, y_train, X_valid, y_valid, cfg)
    
    # 특성 중요도 확인
    importance_df = get_feature_importance(model, X_train.columns)
    print("\n=== 특성 중요도 Top 10 ===")
    print(importance_df.head(10))
    
    # 모델 저장
    model_dir = os.path.join(cfg.common.output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_txt_path = os.path.join(model_dir, 'lightgbm_model.txt')
    model.save_model(model_txt_path)
    
    model_pkl_path = os.path.join(model_dir, 'lightgbm_model.pkl')
    with open(model_pkl_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"✅ 모델이 저장되었습니다: {model_txt_path}, {model_pkl_path}")
    
    # 모델 평가
    val_users = list(test_dict.keys())
    ndcg_score = evaluate_model(model, val_users, df_prep, test_dict, feature_columns, num_workers=cfg.model.lightgbm.n_jobs)
    print(f"\n🎯 Validation NDCG@10: {ndcg_score:.4f}")
    
    # 사용자별 추천 생성
    test_users_idx = np.array(df_prep["user_idx"].unique())
    
    # 최종 추천 생성
    final_results = generate_final_recommendations(
        test_users_idx, 
        df_prep, 
        idx2user, 
        idx2item, 
        feature_columns, 
        model,
        cfg
    )
    
    return final_results