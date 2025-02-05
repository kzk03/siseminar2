import os
import json
import joblib
import datetime
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
#from utils.data_loader2 import load_pr_data
#from utils.feature_extractor2 import extract_features
#from utils.model_trainer2 import train_model

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
            with open(os.path.join(data_path, file), 'r') as f:
                pr_data.append(json.load(f))
    return pr_data

def extract_features(pr_data, start_date=None, end_date=None):
    print(start_date)
    
    """PRデータから特徴量を抽出"""
    metrics_list = []
    objective_list = []

    for pr in pr_data:
        # 📌 `start_date` & `end_date` が指定されている場合のみフィルタリング
        if start_date and end_date:
            pr_date = pr.get("created_at", "2025-02-01")  # デフォルト値を設定
            print(pr_date)
            if not (start_date <= pr_date <= end_date):  # 期間外のデータはスキップ
                continue

        metrics_list.append([
            pr.get('comments', 0),  # メッセージ数
            pr.get('additions', 0), # 追加行数
            pr.get('deletions', 0)   # 削除行数
        ])
        
        # PR内のメッセージを評価してラベルを決定
        is_positive = pr.get('review_comments', 0) > 0
        objective_list.append(1 if is_positive else 0)
    
    print(metrics_list)
    return metrics_list, objective_list

def main():
    data_path = "pr_data"

    today = datetime.date.today()
    start_date = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading PR data for training...")
    pr_data = load_pr_data(data_path)

    print(f"Extracting features from {start_date} to {end_date}...")
    metrics_list, objective_list = extract_features(pr_data, start_date, end_date)

    print("Training model...")
    model_path = train_model(metrics_list, objective_list, output_dir)

    print("Training complete. Model saved at:", model_path)

if __name__ == "__main__":
    main()
