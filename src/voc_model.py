"""定义商家专属 VOC 多标签模型及权重加载逻辑。"""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
VOC_MODEL_DIR = ROOT / "models" / "voc_multilabel"
BASE_MODEL = "hfl/chinese-macbert-base"
VOC_LABELS = ["质量问题", "物流问题", "退款退货", "改进建议", "购买意向"]


class VOCClassifier(nn.Module):
    """使用 MacBERT 同时判断一条反馈是否属于五种业务问题。"""

    def __init__(self, base_model=BASE_MODEL):
        """建立共享文本编码器与五维多标签分类头。"""
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(VOC_LABELS))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """返回各标签的 logits；调用方使用 sigmoid 转为独立概率。"""
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.classifier(self.dropout(output.last_hidden_state[:, 0]))


def load_voc_model(device):
    """加载商家训练权重和各标签阈值；未训练时返回三个 None。"""
    weights = VOC_MODEL_DIR / "voc_head.pt"
    if not weights.exists():
        return None, None, None
    meta = json.loads((VOC_MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(VOC_MODEL_DIR)
    model = VOCClassifier(meta["base_model"])
    # 保存的元信息包含底座名称和验证集调出的标签阈值。
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.to(device).eval()
    return tokenizer, model, meta
