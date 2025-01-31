def extract_features(pr_data, start_date=None, end_date=None):
    """PRデータから特徴量を抽出"""
    metrics_list = []
    objective_list = []

    for pr in pr_data:
        # 📌 `start_date` & `end_date` が指定されている場合のみフィルタリング
        if start_date and end_date:
            pr_date = pr.get("created_at", "2000-01-01")  # デフォルト値を設定
            if not (start_date <= pr_date <= end_date):  # 期間外のデータはスキップ
                continue

        metrics_list.append([
            pr.get('comment', 0),  # メッセージ数
            pr.get('additions', 0), # 追加行数
            pr.get('deletions', 0)   # 削除行数
        ])
        
        # PR内のメッセージを評価してラベルを決定
        is_positive = pr.get('review_comments', 0) > 0
        objective_list.append(1 if is_positive else 0)

    return metrics_list, objective_list
