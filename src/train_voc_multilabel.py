"""使用商家授权标注评论训练五类多标签 VOC 分类器。"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from voc_model import BASE_MODEL, VOC_LABELS, VOC_MODEL_DIR, VOCClassifier

LABEL_ALIASES = {
    "quality_issue": "质量问题",
    "delivery_issue": "物流问题",
    "refund_return": "退款退货",
    "feature_request": "改进建议",
    "purchase_intent": "购买意向",
}


class VOCDataset(Dataset):
    """读取 review_text / issue_types，并将标签集合编码成五维 0/1 向量。"""

    def __init__(self, path):
        """加载 CSV 并校验标签，将多标签字符串转成五维目标向量。"""
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if not {"review_text", "issue_types"}.issubset(frame.columns):
            raise ValueError(f"{path} must have review_text and issue_types columns")
        self.texts = frame["review_text"].fillna("").astype(str).tolist()
        self.targets = []
        for value in frame["issue_types"].fillna("").astype(str):
            labels = {
                item.strip()
                for item in value.replace("|", ";").split(";")
                if item.strip()
            }
            # 同时接受模板中的英文别名和模型使用的标准中文类别名。
            labels = {LABEL_ALIASES.get(label, label) for label in labels}
            unknown = labels.difference(VOC_LABELS)
            if unknown:
                raise ValueError(
                    f"Unknown issue_types: {sorted(unknown)}; allowed={VOC_LABELS}"
                )
            self.targets.append([float(label in labels) for label in VOC_LABELS])

    def __len__(self):
        """返回评论条数。"""
        return len(self.texts)

    def __getitem__(self, index):
        """返回一条评论文本及浮点型多标签向量。"""
        return self.texts[index], torch.tensor(self.targets[index], dtype=torch.float32)


def make_collator(tokenizer, max_length):
    """创建动态 padding 的 batch 组装器，返回编码文本和多标签目标矩阵。"""

    def collate(batch):
        """将文本编码并动态补齐，同时堆叠多标签监督目标。"""
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        return enc, torch.stack(labels)

    return collate


def seed_all(seed):
    """固定随机数种子，帮助复现数据打乱与模型训练过程。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def infer(model, loader, device):
    """推理并返回真实标签矩阵及 sigmoid 概率矩阵。"""
    model.eval()
    probs, labels = [], []
    for enc, batch_labels in loader:
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(**enc)
        probs.append(logits.sigmoid().cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.vstack(labels), np.vstack(probs)


def tune_thresholds(y_true, y_prob):
    """逐标签在验证集搜索 F1 最优阈值，缓解所有类共用 0.5 的问题。"""
    thresholds = []
    for i in range(y_true.shape[1]):
        candidates = np.arange(0.15, 0.86, 0.05)
        scores = [
            f1_score(y_true[:, i], y_prob[:, i] >= t, zero_division=0)
            for t in candidates
        ]
        thresholds.append(float(candidates[int(np.argmax(scores))]))
    return thresholds


def metric_dict(y_true, y_prob, thresholds):
    """按传入阈值二值化概率，并计算整体及逐类别指标。"""
    pred = (y_prob >= np.asarray(thresholds)).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "exact_match_accuracy": float((y_true == pred).all(axis=1).mean()),
        "per_label": classification_report(
            y_true, pred, target_names=VOC_LABELS, output_dict=True, zero_division=0
        ),
        "thresholds": {label: float(t) for label, t in zip(VOC_LABELS, thresholds)},
    }


def main():
    """训练多标签 VOC 模型；阈值只在 dev 集调节，test 集用于最终报告。"""
    parser = argparse.ArgumentParser(
        description="Fine-tune MacBERT on manually labeled customer VOC feedback."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--dev-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    datasets = {
        name: VOCDataset(getattr(args, f"{name}_csv"))
        for name in ("train", "dev", "test")
    }
    collate = make_collator(tokenizer, args.max_length)
    loaders = {
        name: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )
        for name, ds in datasets.items()
    }
    train_counts = np.asarray(datasets["train"].targets).sum(axis=0)
    if np.any(train_counts < 20):
        counts_by_label = dict(zip(VOC_LABELS, train_counts.tolist()))
        raise ValueError(
            "Each category needs enough positive examples; "
            f"train counts={counts_by_label}"
        )

    model = VOCClassifier(args.model_name).to(device)
    y_train = np.asarray(datasets["train"].targets)
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    # 正例稀少时提高其损失权重，并裁剪权重避免梯度被少数样本主导。
    pos_weight = torch.tensor(
        np.sqrt(neg / np.maximum(pos, 1)), dtype=torch.float32, device=device
    ).clamp(1.0, 6.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(loaders["train"]) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.06), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    VOC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = -1
    best_thresholds = [0.5] * len(VOC_LABELS)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            enumerate(loaders["train"]),
            total=len(loaders["train"]),
            desc=f"epoch {epoch+1}/{args.epochs}",
            mininterval=5,
        )
        for step, (enc, labels) in progress:
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(**enc)
                loss = criterion(logits.float(), labels) / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}")

        y_dev, p_dev = infer(model, loaders["dev"], device)
        thresholds = tune_thresholds(y_dev, p_dev)
        metrics = metric_dict(y_dev, p_dev, thresholds)
        print(
            f"epoch={epoch + 1} dev_macro_f1={metrics['macro_f1']:.4f} "
            f"dev_micro_f1={metrics['micro_f1']:.4f}"
        )
        # checkpoint 依据 dev Macro-F1 选择；不能依据 test 指标挑模型。
        if metrics["macro_f1"] > best_f1:
            best_f1, best_thresholds = metrics["macro_f1"], thresholds
            torch.save(model.cpu().state_dict(), VOC_MODEL_DIR / "voc_head.pt")
            model.to(device)
            tokenizer.save_pretrained(VOC_MODEL_DIR)
            (VOC_MODEL_DIR / "meta.json").write_text(
                json.dumps(
                    {
                        "base_model": args.model_name,
                        "labels": VOC_LABELS,
                        "thresholds": dict(zip(VOC_LABELS, thresholds)),
                        "max_length": args.max_length,
                        "train_rows": len(datasets["train"]),
                        "dev_rows": len(datasets["dev"]),
                        "dev_metrics": metrics,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    # 最终阈值取自最佳验证轮次，再用该阈值报告独立测试集结果。
    best = VOCClassifier(args.model_name).to(device)
    best.load_state_dict(
        torch.load(
            VOC_MODEL_DIR / "voc_head.pt", map_location=device, weights_only=True
        )
    )
    y_test, p_test = infer(best, loaders["test"], device)
    report = {
        "dataset_scope": "authorized merchant feedback only",
        "train_rows": len(datasets["train"]),
        "dev_rows": len(datasets["dev"]),
        "test_rows": len(datasets["test"]),
        "test_metrics": metric_dict(y_test, p_test, best_thresholds),
    }
    (VOC_MODEL_DIR / "test_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "macro_f1": report["test_metrics"]["macro_f1"],
                "micro_f1": report["test_metrics"]["micro_f1"],
                "per_label": report["test_metrics"]["per_label"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
