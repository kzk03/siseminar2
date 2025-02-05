import joblib
import numpy as np
import os
import glob

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

def predict_with_model(model_dir, metrics_list):
    """
    GitHubリポジトリ内のmodelsディレクトリにある最新モデルを用いて予測を実行
    :param model_dir: モデルが保存されているディレクトリ
    :param metrics_list: 予測に使用する特徴量 (X)
    :return: 予測結果のリスト
    """
    if not metrics_list:
        raise ValueError("Error: Input data (metrics_list) is empty.")

    model_path = get_latest_model(model_dir)
    if model_path is None:
        raise FileNotFoundError("No valid model file found. Please train a model first.")

    # モデルをロード
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # 特徴量を numpy array に変換
    X = np.array(metrics_list)

    # 予測を実行
    predictions = model.predict(X)
    print("Predictions complete.")

    return predictions

# テスト実行（リポジトリの models ディレクトリを指定）
if __name__ == "__main__":
    model_directory = "models"
    test_features = [[5, 120, 30], [2, 50, 10]]  # テスト用の特徴量リスト
    try:
        results = predict_with_model(model_directory, test_features)
        print("Prediction results:", results)
    except Exception as e:
        print(f"Error during prediction: {e}")
