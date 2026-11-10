"""定义 ASAP 餐饮评论方面级情绪分类模型及其标签映射。"""

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ASAP_MODEL_DIR = ROOT / "models" / "asap_macbert"
BASE_MODEL = "hfl/chinese-macbert-base"
# 方面顺序必须与 ASAP CSV 列顺序和分类头输出顺序保持一致。
ASPECTS = [
    "Location#Transportation",
    "Location#Downtown",
    "Location#Easy_to_find",
    "Service#Queue",
    "Service#Hospitality",
    "Service#Parking",
    "Service#Timely",
    "Price#Level",
    "Price#Cost_effective",
    "Price#Discount",
    "Ambience#Decoration",
    "Ambience#Noise",
    "Ambience#Space",
    "Ambience#Sanitary",
    "Food#Portion",
    "Food#Taste",
    "Food#Appearance",
    "Food#Recommend",
]
ASPECT_ZH = {
    "Location#Transportation": "到店交通",
    "Location#Downtown": "商圈位置",
    "Location#Easy_to_find": "店铺是否好找",
    "Service#Queue": "排队等位",
    "Service#Hospitality": "服务态度",
    "Service#Parking": "停车便利",
    "Service#Timely": "服务效率",
    "Price#Level": "价格水平",
    "Price#Cost_effective": "性价比",
    "Price#Discount": "优惠活动",
    "Ambience#Decoration": "装修",
    "Ambience#Noise": "噪音",
    "Ambience#Space": "空间",
    "Ambience#Sanitary": "卫生",
    "Food#Portion": "分量",
    "Food#Taste": "口味",
    "Food#Appearance": "菜品外观",
    "Food#Recommend": "菜品推荐度",
}
# 原始标签 -2 表示“未提及”，映射为分类头的第 0 类；其余三类为情绪极性。
POLARITIES = {1: "正向", 0: "中性", -1: "负向"}
RAW_LABEL_TO_ID = {-2: 0, -1: 1, 0: 2, 1: 3}
ID_TO_POLARITY = {1: -1, 2: 0, 3: 1}


class AspectClassifier(nn.Module):
    """对每个方面分别预测未提及、负向、中性或正向。"""

    def __init__(self, base_model=BASE_MODEL):
        """加载 MacBERT 编码器，并建立 18 × 4 维分类头。"""
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, len(ASPECTS) * 4)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """返回形状为 (batch, 方面数, 4) 的 logits。"""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = self.dropout(outputs.last_hidden_state[:, 0])
        return self.classifier(cls).view(-1, len(ASPECTS), 4)


def load_asap_model(device):
    """加载已训练权重；权重不存在时返回空值，页面可跳过方面分析。"""
    weights_path = ASAP_MODEL_DIR / "asap_head.pt"
    if not weights_path.exists():
        return None, None
    import json

    meta = json.loads((ASAP_MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(ASAP_MODEL_DIR)
    model = AspectClassifier(meta["base_model"])
    # 先映射到 CPU 加载权重，随后整体迁移到目标设备，兼容不同运行设备。
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.to(device).eval()
    return tokenizer, model
