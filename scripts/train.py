import os
import json
import joblib
import datetime
from collections import Counter
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np

def train_model(metrics_list, objective_list, output_dir):
    """
    Balanced Random Forestを使用してモデルを学習し、保存する
    :param metrics_list: 特徴量のリスト (X)
    :param objective_list: 目的変数のリスト (y)
    :param output_dir: モデルを保存するディレクトリ
    :return: 学習済みモデルの保存パス
    """
    if not metrics_list or not objective_list:
        raise ValueError("Error: Input data (metrics_list or objective_list) is empty.")

    X = np.array(metrics_list)
    y = np.array(objective_list)

    # クラスの分布を確認
    class_counts = Counter(y)
    print(f"Class distribution in y: {class_counts}")

    if len(class_counts) < 2:
        raise ValueError("Error: The target variable y needs at least 2 classes. Found only one.")

    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print("Training Balanced Random Forest Classifier...")

    model = BalancedRandomForestClassifier(random_state=0)
    model.fit(X, y)

    # モデルを保存
    model_path = os.path.join(output_dir, "balanced_random_forest_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    return model_path


def load_pr_data(data_path):
    """PRデータを読み込む"""
    pr_data = []
    for file in os.listdir(data_path):
        if file.endswith(".json"):
            try:
                with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
                    pr_data.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON file {file}")
    return pr_data


def extract_features(pr_data, start_date=None, end_date=None):
    """PRデータから特徴量を抽出"""
    metrics_list = []
    objective_list = []

    for pr in pr_data:
        pull_request = pr.get("pull_request", {})

        # `created_at` の取得と変換
        created_at = pull_request.get("created_at", "2025-02-01T00:00:00Z")
        created_at = datetime.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").date()

        if start_date and end_date and not (start_date <= created_at <= end_date):
            continue

        metrics_list.append([
            pull_request.get('comments', 0),  # コメント数
            pull_request.get('additions', 0), # 追加行数
            pull_request.get('deletions', 0)  # 削除行数
        ])

        # `review_comments` の取得方法修正
        review_comments = pull_request.get("review_comments", 0)
        is_positive = review_comments > 0
        objective_list.append(1 if is_positive else 0)

    # クラス分布のチェック
    class_counts = Counter(objective_list)
    print(f"Class distribution after feature extraction: {class_counts}")

    return metrics_list, objective_list


def main():
    data_path = "pr_data"

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=30)
    end_date = today

    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading PR data for training...")
    pr_data = load_pr_data(data_path)

    print(f"Extracting features from {start_date} to {end_date}...")
    metrics_list, objective_list = extract_features(pr_data, start_date, end_date)

    print("Training model...")
    try:
        model_path = train_model(metrics_list, objective_list, output_dir)
        print("Training complete. Model saved at:", model_path)
    except ValueError as e:
        print(f"Training failed: {e}")


if __name__ == "__main__":
    main()
