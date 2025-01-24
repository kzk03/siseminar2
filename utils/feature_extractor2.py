def extract_features(pr_data, start_date, end_date):
    """PRデータから特徴量を抽出"""
    metrics_list = []
    objective_list = []
 
    for pr in pr_data:
        # 特徴量の作成
        metrics_list.append([
            pr.get('comment', 0),  # メッセージ数
            pr.get('additions', 0), # 追加行数
            pr.get('deletions', 0)   # 削除行数
        ])
        
        # PR内のメッセージを評価してラベルを決定
        is_positive = False
        if pr.get('review_comments') > 0:
            is_positive = True
        
        objective_list.append(1 if is_positive else 0)
 
    return metrics_list, objective_list