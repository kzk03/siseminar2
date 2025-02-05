import sys
import os
import datetime

# カレントディレクトリを sys.path に追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader2 import load_pr_data
from utils.feature_extractor2 import extract_features
from utils.model_trainer2 import train_model

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
