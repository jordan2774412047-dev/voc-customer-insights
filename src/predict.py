"""命令行单条中文评论情绪预测入口。"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "macbert"
LABELS = {0: "负向", 1: "正向"}


def main():
    """解析命令行文本，加载本地权重并打印情绪类别与置信度。"""
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        raise SystemExit('用法：python src/predict.py "这台机器用了两周就漏水了"')
    if not (MODEL_DIR / "config.json").exists():
        raise SystemExit("尚未训练 MacBERT。请先运行 python src/train_macbert.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = (
        AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    )
    # 截断长度与整体情绪模型训练配置保持一致。
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(dim=-1)[0]
    idx = int(probs.argmax().item())
    print(
        {
            "text": text,
            "sentiment": LABELS[idx],
            "confidence": round(float(probs[idx]), 4),
        }
    )


if __name__ == "__main__":
    main()
