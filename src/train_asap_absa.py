"""微调餐饮评论 18 方面情绪模型，并按方面联合 F1 进行评估。"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from asap_model import (
    ASAP_MODEL_DIR,
    ASPECTS,
    AspectClassifier,
    BASE_MODEL,
    ID_TO_POLARITY,
    RAW_LABEL_TO_ID,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "asap_data"
REPORT_DIR = ROOT / "reports"


class ASAPDataset(Dataset):
    """将 ASAP 原始极性矩阵映射为每方面的四分类目标。"""

    def __init__(self, frame):
        """将文本与 18 列原始标签编码成模型使用的四分类 ID。"""
        self.texts = frame["review"].fillna("").astype(str).tolist()
        labels = frame[ASPECTS].to_numpy(dtype=np.int64)
        self.targets = np.vectorize(RAW_LABEL_TO_ID.__getitem__)(labels)

    def __len__(self):
        """返回评论条数。"""
        return len(self.texts)

    def __getitem__(self, index):
        """返回单条评论文本和形状为 (18,) 的类别目标。"""
        return self.texts[index], torch.tensor(self.targets[index], dtype=torch.long)


def make_collator(tokenizer, max_length):
    """创建动态 padding 的批处理函数，按当前 batch 最长文本补齐。"""

    def collate(batch):
        """将文本动态补齐成 batch，并堆叠方面标签矩阵。"""
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


def set_seed(seed):
    """固定 Python、NumPy 和 PyTorch 随机数生成器。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(gold, pred):
    """计算方面提及、方面-极性元组和已提及方面极性的指标。"""
    gold = gold.numpy()
    pred = pred.numpy()
    gold_present = gold != 0
    pred_present = pred != 0
    mention_p = precision_score(
        gold_present.reshape(-1), pred_present.reshape(-1), zero_division=0
    )
    mention_r = recall_score(
        gold_present.reshape(-1), pred_present.reshape(-1), zero_division=0
    )
    mention_f1 = f1_score(
        gold_present.reshape(-1), pred_present.reshape(-1), zero_division=0
    )
    exact_aspect = float(np.all(gold_present == pred_present, axis=1).mean())
    # 联合 F1 要求同一评论、方面及极性三者完全匹配才算正确。
    gold_triples = gold != 0
    pred_triples = pred != 0
    tuple_correct = (gold == pred) & (gold != 0)
    tp = int(tuple_correct.sum())
    fp = int(pred_triples.sum() - tp)
    fn = int(gold_triples.sum() - tp)
    tuple_p = tp / max(tp + fp, 1)
    tuple_r = tp / max(tp + fn, 1)
    tuple_f1 = 2 * tuple_p * tuple_r / max(tuple_p + tuple_r, 1e-12)
    mention_gold = gold[gold != 0]
    mention_pred = pred[gold != 0]
    polarity_f1 = f1_score(
        mention_gold, mention_pred, labels=[1, 2, 3], average="macro", zero_division=0
    )
    return {
        "aspect_mention_precision": float(mention_p),
        "aspect_mention_recall": float(mention_r),
        "aspect_mention_f1": float(mention_f1),
        "aspect_polarity_tuple_precision": float(tuple_p),
        "aspect_polarity_tuple_recall": float(tuple_r),
        "aspect_polarity_tuple_f1": float(tuple_f1),
        "polarity_macro_f1_on_gold_mentions": float(polarity_f1),
        "exact_aspect_set_accuracy": exact_aspect,
    }


@torch.no_grad()
def predict_ids(model, loader, device):
    """对数据集推理，返回目标标签和预测标签张量。"""
    model.eval()
    predictions, targets = [], []
    for enc, labels in tqdm(loader, desc="evaluate", leave=False, mininterval=5):
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**enc)
        predictions.append(logits.argmax(-1).cpu())
        targets.append(labels)
    return torch.cat(targets), torch.cat(predictions)


def main():
    """读取官方划分，训练方面分类头，保存验证集最佳权重并报告测试集。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ASAP fine-tuning needs the existing CUDA PyTorch environment."
        )
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU={gpu_name}; VRAM={vram_gb:.1f} GB")

    frames = {
        name: pd.read_csv(DATA_DIR / f"{file}.csv")
        for name, file in (("train", "train"), ("dev", "dev"), ("test", "test"))
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate = make_collator(tokenizer, args.max_length)
    loaders = {
        name: DataLoader(
            ASAPDataset(frame),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=0,
            pin_memory=True,
            collate_fn=collate,
        )
        for name, frame in frames.items()
    }
    model = AspectClassifier(args.model_name).to(device)

    raw_labels = frames["train"][ASPECTS].to_numpy(dtype=np.int64)
    # 方面与极性类别分布不均衡；基于训练集频数计算温和的逆频率权重。
    class_weights = np.zeros((len(ASPECTS), 4), dtype=np.float32)
    for aspect_i, aspect in enumerate(ASPECTS):
        counts = np.bincount(
            np.vectorize(RAW_LABEL_TO_ID.__getitem__)(raw_labels[:, aspect_i]),
            minlength=4,
        )
        weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
        weights = np.clip(weights / np.mean(weights), 0.5, 4.0)
        class_weights[aspect_i] = weights
    class_weights = torch.tensor(class_weights, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(loaders["train"]) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.06), total_steps
    )
    scaler = torch.amp.GradScaler("cuda")
    best_f1 = -1.0
    ASAP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    global_step = 0

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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**enc)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                per_item = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                selected_weights = (
                    class_weights.unsqueeze(0)
                    .expand(labels.shape[0], -1, -1)
                    .gather(-1, labels.unsqueeze(-1))
                    .squeeze(-1)
                )
                loss = (per_item * selected_weights).mean() / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            progress.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}")

        y_true, y_pred = predict_ids(model, loaders["dev"], device)
        metrics = compute_metrics(y_true, y_pred)
        print(
            f"epoch={epoch + 1} "
            f"dev_tuple_f1={metrics['aspect_polarity_tuple_f1']:.4f} "
            f"dev_mention_f1={metrics['aspect_mention_f1']:.4f}"
        )
        # 用验证集联合 F1 选模型，保留官方测试集用于最后一次无偏评估。
        if metrics["aspect_polarity_tuple_f1"] > best_f1:
            best_f1 = metrics["aspect_polarity_tuple_f1"]
            torch.save(model.cpu().state_dict(), ASAP_MODEL_DIR / "asap_head.pt")
            model.to(device)
            tokenizer.save_pretrained(ASAP_MODEL_DIR)
            meta = {
                "base_model": args.model_name,
                "aspects": ASPECTS,
                "max_length": args.max_length,
                "no_mention_label_id": 0,
                "id_to_polarity": ID_TO_POLARITY,
                "best_dev_metrics": metrics,
                "train_rows": len(frames["train"]),
                "dev_rows": len(frames["dev"]),
                "test_rows": len(frames["test"]),
            }
            (ASAP_MODEL_DIR / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # 重新载入验证集最佳 checkpoint 后，才在 test split 上计算最终指标。
    best = AspectClassifier(args.model_name).to(device)
    best.load_state_dict(
        torch.load(
            ASAP_MODEL_DIR / "asap_head.pt", map_location=device, weights_only=True
        )
    )
    test_true, test_pred = predict_ids(best, loaders["test"], device)
    result = {
        "dataset": "Meituan-Dianping ASAP",
        "train_rows": len(frames["train"]),
        "dev_rows": len(frames["dev"]),
        "test_rows": len(frames["test"]),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "aspects": ASPECTS,
        "test_metrics": compute_metrics(test_true, test_pred),
        "scope_note": (
            "Restaurant aspect/polarity benchmark; it does not label refund, "
            "purchase intent, or product delivery logistics."
        ),
    }
    (REPORT_DIR / "asap_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["test_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
