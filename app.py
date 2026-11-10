"""VOC 本地网页：整合情绪模型、餐饮方面模型、规则提示和人工复核。"""

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.asap_model import ASPECT_ZH, load_asap_model
from src.rule_triage import actions_for_aspects, classify_intents
from src.voc_model import VOC_LABELS, load_voc_model

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "macbert"
LABELS = {0: "负向反馈", 1: "正向反馈"}

st.set_page_config(page_title="VOC 客户声音分析", page_icon="🗣️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding: 2rem 2.2rem 3rem;
    }
    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid #e5edf5;
        border-radius: 18px;
        background: linear-gradient(120deg, #f4f8ff 0%, #f7fbf8 100%);
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        margin: 0 0 .45rem 0;
        color: #102a43;
        letter-spacing: -.03em;
    }
    .hero p {
        margin: 0;
        color: #52677c;
        font-size: 1.02rem;
    }
    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #e8eef4;
        border-radius: 14px;
        padding: .9rem 1rem;
    }
    div[data-testid="stMetricLabel"] {color: #52677c;}
    div[data-testid="stTabs"] button {font-weight: 600;}
    .stAlert {border-radius: 12px;}
    section[data-testid="stFileUploader"] {border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🗣️ VOC 客户声音分析</h1>'
    "<p>从客户原话中发现情绪、具体问题和跟进线索，支持批量处理与人工复核。</p></div>",
    unsafe_allow_html=True,
)
st.markdown("### VOC 工作台")
cap1, cap2, cap3, cap4 = st.columns(4)
cap1.metric("整体情绪", "正向 / 负向", "测试集 Macro-F1 95.1%")
cap2.metric("餐饮方面分析", "18 个方面", "方面-情绪联合 F1 74.0%")
cap3.metric("业务问题提示", "5 类", "规则初筛，需人工确认")
cap4.metric("批量处理", "CSV 上传 / 导出", "编辑确认类别与处理状态")

with st.expander("查看可识别的方面和业务类别", expanded=True):
    st.markdown(
        "**餐饮方面模型**："
        "到店交通、商圈位置、店铺是否好找、排队等位、服务态度、停车便利、"
        "服务效率、价格水平、性价比、优惠活动、装修、噪音、空间、卫生、"
        "分量、口味、菜品外观、菜品推荐度。"
    )
    st.markdown("**业务规则提示**：质量问题、物流问题、退款退货、改进建议、购买意向。")
    st.caption(
        "整体情绪与餐饮方面情绪由模型分析；五类业务问题目前由关键词规则提示。"
        "餐饮方面模型适用于餐饮评价，其他行业的方面结果请人工复核。"
    )

sample_path = ROOT / "data" / "sample_voc_test.csv"
if sample_path.exists():
    st.download_button(
        "下载 24 条测试数据",
        data=sample_path.read_bytes(),
        file_name="sample_voc_test.csv",
        mime="text/csv",
        help="可在 CSV 批量分析页上传；expected_* 列是人工参考标签。",
    )


# Streamlit 每次控件交互都会重跑页面脚本；将大模型放入资源缓存可避免重复加载。
@st.cache_resource(show_spinner="正在加载 MacBERT…")
def load_model():
    """加载整体情绪模型，并根据运行环境选择 CUDA 或 CPU。"""
    if not (MODEL_DIR / "config.json").exists():
        return None, None, torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = (
        AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    )
    return tokenizer, model, device


def predict_texts(texts, tokenizer, model, device, batch_size=32):
    """批量预测整体情绪，并保留两类概率供页面显示。"""
    results = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            logits = model(**enc).logits
            probs = logits.softmax(dim=-1).cpu()
            for text, row in zip(batch_texts, probs):
                idx = int(row.argmax().item())
                results.append(
                    {
                        "反馈文本": text,
                        "情绪判断": LABELS[idx],
                        "置信度": round(float(row[idx]), 4),
                        "负向概率": round(float(row[0]), 4),
                        "正向概率": round(float(row[1]), 4),
                    }
                )
    return results


tokenizer, model, device = load_model()
if model is None:
    st.error("尚未找到训练好的 MacBERT 模型。请先运行 README 中的训练命令。")
    st.stop()
aspect_tokenizer, aspect_model = load_asap_model(device)
voc_tokenizer, voc_model, voc_meta = load_voc_model(device)

# 商家分类模型目前可能不存在；页面据此区分“模型预测”和“规则提示”。
st.caption(
    f"分析设备：{'RTX 5060 / CUDA' if device.type == 'cuda' else 'CPU'} · 选择下面的工作流程开始"
)
tab_single, tab_batch = st.tabs(["🔎 单条诊断", "📊 批量分析与复核"])


def predict_aspects(texts):
    """输出餐饮方面与极性；未训练/未安装方面权重时返回空结果。"""
    if aspect_model is None:
        return [[] for _ in texts]
    results = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            batch_texts = texts[start : start + 16]
            enc = aspect_tokenizer(
                batch_texts,
                truncation=True,
                max_length=384,
                padding=True,
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            probs = aspect_model(**enc).softmax(dim=-1).cpu()
            for row in probs:
                aspects = []
                for i, name in enumerate(ASPECT_ZH):
                    # presence 是“提及任一方面极性”的概率；最终是否出现仍由四分类 argmax 决定。
                    presence = 1.0 - float(row[i, 0])
                    predicted_class = int(row[i].argmax().item())
                    polarity_id = int(row[i, 1:].argmax().item()) + 1
                    if predicted_class != 0:
                        polarity = (
                            "负向"
                            if polarity_id == 1
                            else "中性" if polarity_id == 2 else "正向"
                        )
                        aspects.append(
                            {
                                "方面": name,
                                "情绪": polarity,
                                "提及概率": round(presence, 3),
                            }
                        )
                results.append(aspects)
    return results


def predict_custom_voc(texts):
    """按每类验证集阈值筛选商家模型结果，不以固定 0.5 作为所有标签门槛。"""
    if voc_model is None:
        return [[] for _ in texts]
    thresholds = voc_meta["thresholds"]
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            batch_texts = texts[start : start + 32]
            enc = voc_tokenizer(
                batch_texts,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            probs = voc_model(**enc).sigmoid().cpu()
            for row in probs:
                output.append(
                    [
                        {"类别": label, "置信度": round(float(row[i]), 3)}
                        for i, label in enumerate(VOC_LABELS)
                        if float(row[i]) >= float(thresholds[label])
                    ]
                )
    return output


with tab_single:
    # 单条工作流适合快速查看原文；模型结果和可解释规则结果分开呈现。
    st.markdown("#### 快速诊断一条客户反馈")
    st.caption(
        "查看情绪、可能的问题类别和建议跟进动作。结果用于辅助判断，请结合原文复核。"
    )
    text = st.text_area(
        "客户反馈",
        placeholder="例如：用了两周就漏水了，要求退款。",
        height=130,
        key="single_text",
    )
    if st.button("分析反馈", type="primary", disabled=not text.strip()):
        prediction = predict_texts([text.strip()], tokenizer, model, device)[0]
        intents = classify_intents(text.strip())
        aspects = predict_aspects([text.strip()])[0]
        learned_voc = predict_custom_voc([text.strip()])[0]
        is_negative = prediction["情绪判断"] == "负向反馈"
        st.markdown("### 分析结果")
        metric, confidence, urgency = st.columns(3)
        metric.metric("情绪判断", prediction["情绪判断"])
        confidence.metric("模型置信度", f"{prediction['置信度']:.1%}")
        urgency.metric("建议优先级", intents["urgency"])
        st.progress(
            prediction["负向概率"], text=f"负向概率 {prediction['负向概率']:.1%}"
        )
        st.progress(
            prediction["正向概率"], text=f"正向概率 {prediction['正向概率']:.1%}"
        )
        st.markdown("#### VOC 业务提示")
        if voc_model is not None:
            st.write(
                "商家数据模型："
                + (
                    "、".join(f"{x['类别']} ({x['置信度']:.0%})" for x in learned_voc)
                    if learned_voc
                    else "当前阈值下未识别出目标类别"
                )
            )
        else:
            st.caption(
                "商家专属类别模型尚未训练；下方类别来自可复核的规则提示。"
                "准备好标注数据后，可用 src/train_voc_multilabel.py 微调商家模型。"
            )
        if intents["categories"]:
            st.write("命中类别：" + "、".join(intents["categories"]))
        else:
            st.write("未命中质量、物流、退款、改进建议或购买意向规则。")
        st.caption(intents["source"])
        st.write("建议动作：" + "；".join(intents["actions"]))
        if aspect_model is not None:
            st.markdown("#### 餐饮评价细分方面")
            if aspects:
                st.dataframe(
                    pd.DataFrame(aspects), use_container_width=True, hide_index=True
                )
                aspect_actions = actions_for_aspects(aspects)
                if aspect_actions:
                    st.write("方面分析建议：" + "；".join(aspect_actions))
            else:
                st.write("没有达到当前提及阈值的方面；可人工补充标注。")
        else:
            st.warning("餐饮方面模型尚未训练。")
        if is_negative and prediction["置信度"] < 0.75:
            st.warning("模型判断不够确定，建议人工复核。")
        elif is_negative:
            st.warning("识别为负向反馈。请结合原文由客服判断是否需要优先跟进。")
        else:
            st.success("识别为正向反馈。")

with tab_batch:
    # 批量流程保留输入原始列，并将分析输出附加到每条有效评论上。
    st.markdown("#### 导入评论，集中发现待处理问题")
    st.caption(
        "支持 UTF-8 / Excel 常用 CSV。文本列名可为 review、review_text、text 或 反馈文本；也可以上传后手动选择。"
    )
    uploaded = st.file_uploader("选择 CSV", type=["csv"], key="batch_file")
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded.seek(0)
            frame = pd.read_csv(uploaded, encoding="gb18030")
        candidates = [
            c
            for c in ("review", "review_text", "text", "反馈文本")
            if c in frame.columns
        ]
        if not candidates:
            text_col = st.selectbox("选择反馈文本所在列", frame.columns.tolist())
        else:
            text_col = st.selectbox("反馈文本列", candidates, index=0)

        previous_upload = st.session_state.get("batch_upload_name")
        if previous_upload and previous_upload != uploaded.name:
            st.session_state.pop("batch_result", None)
            st.session_state.pop("batch_editor", None)
        st.session_state["batch_upload_name"] = uploaded.name

        if st.button("批量分析", type="primary", key="batch_analyze"):
            texts = frame[text_col].fillna("").astype(str).str.strip().tolist()
            valid_positions = [i for i, value in enumerate(texts) if value]
            if not valid_positions:
                st.error("没有找到非空反馈文本。")
            else:
                with st.spinner(f"正在分析 {len(valid_positions):,} 条反馈…"):
                    predictions = predict_texts(
                        [texts[i] for i in valid_positions], tokenizer, model, device
                    )
                    aspect_predictions = predict_aspects(
                        [texts[i] for i in valid_positions]
                    )
                    learned_voc_predictions = predict_custom_voc(
                        [texts[i] for i in valid_positions]
                    )
                result = frame.iloc[valid_positions].reset_index(drop=True).copy()
                pred_frame = pd.DataFrame(predictions).drop(columns=["反馈文本"])
                result = pd.concat([result, pred_frame], axis=1)
                intent_predictions = [
                    classify_intents(value)
                    for value in result[text_col].fillna("").astype(str)
                ]
                result["业务提示类别"] = [
                    "、".join(x["categories"]) for x in intent_predictions
                ]
                result["商家模型类别"] = [
                    "、".join(f"{x['类别']}({x['置信度']:.2f})" for x in row)
                    for row in learned_voc_predictions
                ]
                result["建议优先级"] = [x["urgency"] for x in intent_predictions]
                result["建议动作"] = [
                    "；".join(x["actions"]) for x in intent_predictions
                ]
                result["餐饮方面分析"] = [
                    "；".join(f"{item['方面']}:{item['情绪']}" for item in row)
                    for row in aspect_predictions
                ]
                aspect_actions = [
                    "；".join(actions_for_aspects(row)) for row in aspect_predictions
                ]
                result["餐饮运营建议"] = aspect_actions
                result["人工确认类别"] = ""
                result["处理状态"] = "待复核"
                st.session_state["batch_result"] = result
                st.session_state["batch_text_col"] = text_col
                st.session_state["batch_skipped"] = len(texts) - len(valid_positions)

    # 控件交互会触发 Streamlit rerun，因此分析结果放在 session_state 中延续显示。
    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        st.success(
            f"已分析 {len(result):,} 条反馈，跳过空白 "
            f"{st.session_state.get('batch_skipped', 0):,} 条。"
        )
        negative_count = int((result["情绪判断"] == "负向反馈").sum())
        tagged_count = int(result["业务提示类别"].astype(bool).sum())
        urgent_count = int((result["建议优先级"] == "高").sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("分析条数", f"{len(result):,}")
        m2.metric("负向反馈", f"{negative_count:,}")
        m3.metric("命中业务提示", f"{tagged_count:,}")
        m4.metric("高优先级", f"{urgent_count:,}")

        st.markdown("#### 先看问题分布")
        chart1, chart2 = st.columns(2)
        rule_counts = (
            result["业务提示类别"]
            .replace("", pd.NA)
            .dropna()
            .str.split("、")
            .explode()
            .value_counts()
        )
        with chart1:
            st.caption("业务问题提示（规则初筛）")
            if not rule_counts.empty:
                st.bar_chart(rule_counts, color="#4C78A8")
            else:
                st.info("这批反馈暂未命中五类业务提示规则。")
        with chart2:
            st.caption("整体情绪判断")
            st.bar_chart(result["情绪判断"].value_counts(), color="#59A14F")

        negative_aspects = [
            item.split(":", 1)[0]
            for value in result["餐饮方面分析"].fillna("")
            for item in value.split("；")
            if ":负向" in item
        ]
        if negative_aspects:
            st.markdown("#### 餐饮评价中的负面方面")
            st.bar_chart(
                pd.Series(negative_aspects).value_counts().rename("负向提及数"),
                color="#E15759",
            )

        st.markdown("#### 人工复核与处理")
        st.caption(
            "可编辑“人工确认类别”和“处理状态”。表格最多显示前 1,000 条，导出文件保留全部分析结果。"
        )
        edited = st.data_editor(
            result.head(1000),
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=[
                col for col in result.columns if col not in ["人工确认类别", "处理状态"]
            ],
            column_config={
                "人工确认类别": st.column_config.TextColumn(
                    "人工确认类别", help="填写复核后的最终分类"
                ),
                "处理状态": st.column_config.SelectboxColumn(
                    "处理状态", options=["待复核", "已确认", "需升级"]
                ),
            },
            key="batch_editor",
        )
        # 先复制完整结果，只把当前页面可编辑部分的人工字段写回导出副本。
        downloadable = result.copy()
        if not edited.empty:
            downloadable.loc[edited.index, ["人工确认类别", "处理状态"]] = edited[
                ["人工确认类别", "处理状态"]
            ].values
        csv_bytes = downloadable.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载完整分析结果 CSV",
            data=csv_bytes,
            file_name="voc_analysis_results.csv",
            mime="text/csv",
            type="primary",
            key="batch_download",
        )

st.divider()
st.caption(
    "数据提醒：本地原型只处理当前上传内容；上传前请去除姓名、电话、订单号等个人信息。公开训练数据的商业授权需另行核实。"
)
