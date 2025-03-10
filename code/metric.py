from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd

def cal_ndcg(sub_df, valid_dir):
    
    # sub_df는 최종 제출 형식의 추론 파일, train 데이터로 학습한 후 추론한 파일임
    
    valid_df = pd.read_csv(valid_dir)
    valid_df = valid_df[valid_df['event_type'] == 'purchase']
    sub_df = sub_df[sub_df['user_id'].isin(valid_df['user_id'].unique())]

    # ✅ 실제 ground truth 데이터 (valid_df)를 dictionary 형태로 변환 (set 사용)
    ground_truth = valid_df.groupby("user_id")["item_id"].apply(set)

    # ✅ sub_df도 user_id별로 item_id 리스트로 변환 (Pandas 최적화)
    sub_grouped = sub_df.groupby("user_id")["item_id"].apply(list)

    # ✅ 벡터화된 NDCG 계산
    ndcg_scores = []

    for user, recommended_items in sub_grouped.items():
        true_items = ground_truth.get(user, set())  # 없는 경우 빈 set 반환
        
        # ✅ relevance 벡터화 (벡터 연산으로 속도 향상)
        relevance = np.isin(recommended_items, list(true_items)).astype(int)

        # ✅ NDCG@10 계산
        ndcg = ndcg_score([relevance], [relevance], k=10)  # 리스트 형태로 입력
        ndcg_scores.append(ndcg)

    # ✅ 평균 NDCG@10 값 출력
    average_ndcg = np.mean(ndcg_scores)
    print(f"Average NDCG@10: {average_ndcg:.4f}")
    return average_ndcg