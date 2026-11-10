"""训练字符级 TF-IDF + Logistic Regression 情绪分类基线。"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def text_column(frame):
    """从 DataFrame 输入中取出评论列，供 sklearn Pipeline 调用。"""
    return frame["review"].astype(str)


def main() -> None:
    """训练基线模型，在固定测试集评估并保存模型与指标。"""
    train = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    test = pd.read_csv(DATA / "test.csv", encoding="utf-8-sig")

    # 字符 n-gram 不依赖中文分词，适合作为中文短文本的轻量基线。
    model = Pipeline(
        [
            ("text", FunctionTransformer(text_column, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=300000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", C=4.0, solver="liblinear"
                ),
            ),
        ]
    )
    model.fit(train, train["label"])
    # 测试集只用于最终比较，不参与模型拟合或超参数选择。
    pred = model.predict(test)
    report = classification_report(test["label"], pred, digits=4, zero_division=0)
    accuracy = accuracy_score(test["label"], pred)
    macro_f1 = f1_score(test["label"], pred, average="macro")
    summary = (
        f"Accuracy={accuracy:.4f}\n"
        f"Macro-F1={macro_f1:.4f}\n\n"
        f"{report}\n"
        "Confusion matrix:\n"
        f"{confusion_matrix(test['label'], pred)}\n"
    )
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "tfidf_logreg.joblib")
    (REPORT_DIR / "baseline_metrics.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
