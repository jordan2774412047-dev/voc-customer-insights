"""清洗商品评论并生成情绪分类训练、验证和测试集。"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online_shopping_10_cats.csv"
OUT = ROOT / "data" / "processed"


def main() -> None:
    """校验原始字段、清理评论并按类别与情绪分层切分数据。"""
    if not RAW.exists():
        raise FileNotFoundError(f"找不到数据文件：{RAW}")

    df = pd.read_csv(RAW, encoding="utf-8-sig")
    expected = {"cat", "label", "review"}
    if not expected.issubset(df.columns):
        raise ValueError(f"CSV 列不匹配，实际列：{list(df.columns)}")

    # 仅保留后续训练需要的字段，去掉缺失记录并统一文本/标签类型。
    df = df[["cat", "label", "review"]].dropna()
    df["review"] = df["review"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    df = df[df["review"].str.len().between(2, 2000)]
    df = df.drop_duplicates(subset=["cat", "label", "review"]).reset_index(drop=True)
    # 按“商品类别 × 情绪标签”分层，避免切分后某些类别或极性比例失衡。
    df["stratum"] = df["cat"].astype(str) + "__" + df["label"].astype(str)

    # 先留出 20%，再将留出集均分为验证集和最终测试集（各占总量约 10%）。
    train, holdout = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["stratum"]
    )
    valid, test = train_test_split(
        holdout, test_size=0.5, random_state=42, stratify=holdout["stratum"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        # utf-8-sig 便于 Windows Excel 正确读取中文 CSV。
        part[["cat", "label", "review"]].to_csv(
            OUT / f"{name}.csv", index=False, encoding="utf-8-sig"
        )
        label_counts = part["label"].value_counts().sort_index().to_dict()
        print(f"{name}: {len(part):,} rows; labels={label_counts}")
    print(f"Clean unique rows: {len(df):,}; categories: {df['cat'].nunique()}")


if __name__ == "__main__":
    main()
