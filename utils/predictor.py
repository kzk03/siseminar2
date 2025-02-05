import joblib
import numpy as np

def predict_with_model(model_path, metrics_list):
    """
    学習済みモデルを用いて予測を実行
    :param model_path: 学習済みモデルのパス
    :param metrics_list: 予測に使用する特徴量 (X)
    :return: 予測結果のリスト
    """
    if not metrics_list:
        raise ValueError("Error: Input data (metrics_list) is empty.")

    # モデルを読み込み
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # 予測を実行
    X = np.array(metrics_list)
    predictions = model.predict(X)
    print("Predictions complete.")

    return predictions
