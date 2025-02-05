from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
import joblib
import os

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
