import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model_latest.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(LATEST_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(LATEST_MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(LATEST_MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"


def test_model_performance(train_model):
    """モデルの性能を検証"""

    # 環境変数の初期化
    with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
        print(f"MODEL_IMPROVED=true", file=fh)
        print(f"FIRST_MODEL=true", file=fh)

    """既存モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("既存モデルファイルが存在しないためスキップします")
    else:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            print(f"FIRST_MODEL=false", file=fh)

    """モデルの性能を検証"""
    latest_model, X_test, y_test = train_model

    with open(MODEL_PATH, "rb") as f:
        old_model = pickle.load(f)

    # 推論時間の計測
    start_time = time.time()
    old_pred = old_model.predict(X_test)
    end_time = time.time()
    old_inference_time = end_time - start_time

    start_time = time.time()
    latest_pred = latest_model.predict(X_test)
    end_time = time.time()
    latest_inference_time = end_time - start_time

    time_is_better = latest_inference_time < old_inference_time

    if time_is_better:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            print(f"OLD_MODEL_TIME={old_inference_time}", file=fh)
            print(f"NEW_MODEL_TIME={latest_inference_time}", file=fh)
    else:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            print(f"MODEL_IMPROVED=false", file=fh)

    assert (
        time_is_better
    ), f"新しいモデルの推論時間が古いモデルより遅いです。 latest:{latest_inference_time}秒  old:{old_inference_time}秒"

    # 予測精度の計算
    old_accuracy = accuracy_score(y_test, old_pred)

    latest_accuracy = accuracy_score(y_test, latest_pred)

    acc_is_better = latest_accuracy > old_accuracy

    if acc_is_better:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            print(f"OLD_MODEL_ACCURACY={old_accuracy}", file=fh)
            print(f"NEW_MODEL_ACCURACY={latest_accuracy}", file=fh)
    else:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            print(f"MODEL_IMPROVED=false", file=fh)

    assert (
        acc_is_better
    ), f"新しいモデルの精度が古いモデルより低いです。 latest:{latest_accuracy}  old:{old_accuracy}"

    # モデルのF1スコアを計算
    old_f1 = f1_score(y_test, old_pred)

    latest_f1 = f1_score(y_test, latest_pred)

    f1_is_better = latest_f1 > old_f1

    if f1_is_better:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            print(f"OLD_MODEL_F1={old_f1}", file=fh)
            print(f"NEW_MODEL_F1={latest_f1}", file=fh)
    else:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            print(f"MODEL_IMPROVED=false", file=fh)

    assert (
        f1_is_better
    ), f"新しいモデルのF1スコアが古いモデルより低いです。 latest:{latest_f1}  old:{old_f1}"
