"""用 MacBERT 微调整体正/负向情绪分类器并评估独立测试集。"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VALID_CSV = ROOT / "data" / "processed" / "valid.csv"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
MODEL_DIR = ROOT / "models" / "macbert"
REPORT_DIR = ROOT / "reports"
DEFAULT_MODEL = "hfl/chinese-macbert-base"


class ReviewDataset(Dataset):
    """将 pandas 评论表封装为 Transformer 可读取的张量数据集。"""

    def __init__(self, frame, tokenizer, max_length):
        """缓存文本和标签；分词器仅在取样时使用，避免预先占用过多内存。"""
        self.texts = frame["review"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """返回数据集中评论条数。"""
        return len(self.labels)

    def __getitem__(self, index):
        """按需分词，返回模型输入和一个整数情绪标签。"""
        enc = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def seed_everything(seed):
    """固定常用随机数生成器，尽量提高训练过程可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    """关闭梯度计算并收集一轮预测；评估阶段不更新模型参数。"""
    model.eval()
    labels, preds = [], []
    for batch in tqdm(loader, desc="evaluate", leave=False, mininterval=5):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    return labels, preds


def main():
    """读取切分数据、训练若干轮、按验证集 Macro-F1 保存模型并测试。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用。先安装 CUDA 版 PyTorch，并确认 torch.cuda.is_available() 为 True。"
        )
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}; VRAM={vram_gb:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_frame = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    valid_frame = pd.read_csv(VALID_CSV, encoding="utf-8-sig")
    test_frame = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
    train_ds = ReviewDataset(train_frame, tokenizer, args.max_length)
    valid_ds = ReviewDataset(valid_frame, tokenizer, args.max_length)
    test_ds = ReviewDataset(test_frame, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)
    # 梯度累积允许在显存有限时使用较小微批次，同时模拟更大的有效批次。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(train_loader) + args.grad_accum - 1) // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(steps_per_epoch * args.epochs * 0.06),
        steps_per_epoch * args.epochs,
    )
    # FP16 自动混合精度降低显存占用；GradScaler 防止半精度梯度下溢。
    scaler = torch.amp.GradScaler("cuda")
    amp = True
    best_f1 = -1.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"epoch {epoch + 1}/{args.epochs}",
            mininterval=5,
        )
        for step, batch in progress:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                ).loss
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            total_loss += loss.item() * args.grad_accum
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{total_loss / (step + 1):.4f}")

        y_true, y_pred = evaluate(model, valid_loader, device, amp)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        val_accuracy = accuracy_score(y_true, y_pred)
        print(
            f"epoch={epoch + 1} val_macro_f1={val_f1:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )
        # 只用验证集挑 checkpoint；测试集留到训练结束后使用一次。
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

    # 从验证集最优权重重新加载，再进行最终测试集评估。
    best = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    y_true, y_pred = evaluate(best, test_loader, device, amp)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "model_name": args.model_name,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "test_rows": len(y_true),
        "warning": "公开数据二分类情绪结果不代表 VOC 多类别业务效果。",
    }
    (REPORT_DIR / "macbert_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: metrics[k] for k in ("accuracy", "macro_f1", "test_rows")},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
