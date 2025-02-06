import joblib
import numpy as np
import os
import glob
import os
import datetime
import json
from collections import Counter

def load_pr_data(data_path):
    """PRデータを読み込む"""
    pr_data = []
    for file in os.listdir(data_path):
        if file.endswith(".json"):
            with open(os.path.join(data_path, file), 'r') as f:
                pr_data.append(json.load(f))
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


def get_latest_model(model_dir):
    """
    指定されたディレクトリ内の最新のpklモデルファイルを取得
    :param model_dir: モデルが保存されているディレクトリ
    :return: 最新のモデルファイルのパス（存在しない場合は None）
    """
    model_files = glob.glob(os.path.join(model_dir, "*.pkl"))

    if not model_files:
        print("Error: No model file found in the directory.")
        return None

    # 最も新しいモデルを選択（更新日時が最新のもの）
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Using latest model: {latest_model}")

    return latest_model

def predict_with_model(model_dir, pr_data_path, start_date, end_date):
    """GitHubリポジトリ内のmodelsディレクトリにある最新モデルを用いて予測を実行"""
    
    print(f"Loading PR data from {pr_data_path}...")
    pr_data = load_pr_data(pr_data_path)

    print(f"Extracting features from {start_date} to {end_date}...")
    metrics_list = []
    pr_numbers = []  # PR番号を保存するリスト

    for pr in pr_data:
        pull_request = pr.get("pull_request", {})
        created_at = pull_request.get("created_at", "2025-02-01T00:00:00Z")
        created_at = datetime.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").date()

        if start_date and end_date and not (start_date <= created_at <= end_date):
            continue

        metrics_list.append([
            pull_request.get('comments', 0),
            pull_request.get('additions', 0),
            pull_request.get('deletions', 0)
        ])

        pr_numbers.append(pr.get("number", "N/A"))  # PR番号を取得

    if not metrics_list:
        raise ValueError("Error: No valid features extracted from PR data.")

    model_path = get_latest_model(model_dir)
    if model_path is None:
        raise FileNotFoundError("No valid model file found. Please train a model first.")

    model = joblib.load(model_path)
    print("Model loaded successfully.")

    X = np.array(metrics_list)
    prediction_scores = model.predict_proba(X)[:, 1]  # 1クラス（肯定クラス）の確率のみ取得

    # スコアとPR番号を結合
    results = [f"{score:.2f}:{num}" for score, num in zip(prediction_scores, pr_numbers)]

    #print("Prediction scores:", results)
    return results


if __name__ == "__main__":
    model_directory = "models"
    pr_data_directory = "pr_data"
    today = datetime.date.today()

    # 日付を datetime.date に変換
    start_date = today - datetime.timedelta(days=30)
    end_date = today

    try:
        results = predict_with_model(model_directory, pr_data_directory, start_date, end_date)
        print("Prediction scores:", results)  # 0または1ではなくスコアを出力
    except Exception as e:
        print(f"Error during prediction: {e}")

