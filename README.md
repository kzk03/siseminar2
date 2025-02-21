README - GitHub Actions PR 優先順位推定ツール

概要

本プロジェクトはGitHub Actionsを利用して、Pull Request (PR) の情報を保存し、そのデータをもとにPRのレビュー優先順位を推定するツールです。作成されたPRは保存され、機械学習モデルによって優先度が算出されます。

機能

PRが作成されるたびに、そのデータをJSON形式で保存

過去のPRデータをもとに機械学習モデルを学習

最新のPRデータに対して優先度を算出し、上位10件をリストアップ

モデルは毎日0時に更新される

システム構成

1. PRデータの保存 (.github/workflows/save_pr_data.yml)

GitHub ActionsがPR作成時に自動実行され、PR情報をJSON形式で pr_data/ ディレクトリに保存します。

2. モデルの学習 (train.py)

train.py は過去30日間のPRデータを使用し、機械学習モデルを構築。

BalancedRandomForestClassifier を使用し、PRの特徴量を学習。

学習済みモデルは models/ ディレクトリに保存。

3. 予測 (predict.py)

predict.py は最新のモデルを使用し、新しいPRの優先順位を予測。

予測結果から優先度の高い上位10件を表示。

使用方法

1. リポジトリのセットアップ

git clone https://github.com/your-repo.git
cd your-repo

2. PRデータの収集

GitHub Actionsによって自動実行され、データが pr_data/ に保存されます。

3. モデルの学習

python train.py

4. 予測の実行

python predict.py

必要な環境

Python 3.8 以上

依存ライブラリ (事前にインストールしてください)

pip install -r requirements.txt

カスタマイズ

train.py の extract_features 関数を変更すると、PRの評価基準をカスタマイズ可能。

predict.py の表示形式を変更し、PRの優先順位を異なる形式で表示可能。

エラーハンドリング

JSONの破損データをスキップして処理。

モデルが存在しない場合はエラーメッセージを表示。

予測結果が取得できない場合は警告を出力。

ライセンス

本プロジェクトはMITライセンスのもとで提供されます。

貢献方法

Issueを作成し、バグ報告や機能提案を行う。
