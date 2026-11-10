# VOC 助手原型

这是一个本地运行的中文客户反馈分析原型，面向 Windows + NVIDIA GPU。它用商品评论训练整体情绪模型，用美团 ASAP 餐饮评论训练 18 个方面及情绪模型；质量、物流、退款、建议、购买意向目前由可人工复核的规则提示辅助。ASAP 不含退款、配送、建议和购买意向标签，所以这些业务标签还不能宣称达到模型分类精度。

## 电脑与运行环境

本机检测到 Ryzen 9 8940HX、16 GB 内存、RTX 5060 Laptop（约 8 GB 显存）。电脑已有 `nlp_gpu` Conda 环境，内含 CUDA 版 PyTorch，已确认可以识别 RTX 5060；直接使用它即可，不需要改动原 Anaconda 环境。

```powershell
conda activate nlp_gpu
cd "D:\path\to\VOC助手原型"  # 替换成你自己的克隆目录
pip install -r requirements.txt
```

本机环境实测为 PyTorch 2.14.1+cu130、Transformers 4.41.2、CUDA 可用。如在别的电脑复用项目，再打开 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)按对应硬件选择安装命令。验证显卡：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA不可用')"
```

GPU 训练前确认输出 `True` 和 RTX 5060。网页预测也能在 CPU 上运行，但会慢一些。

## 运行顺序

1. 克隆仓库后，原始语料和训练权重不会随代码提交。请按 `data/DATA_SOURCE.md` 的来源说明自行下载并放入 `data/raw/`：商品评论文件命名为 `online_shopping_10_cats.csv`；ASAP 餐饮数据放入 `data/raw/asap_data/`，包含 `train.csv`、`dev.csv`、`test.csv`。下载前请核对各数据集许可与使用范围。
2. 切分训练/验证/测试集：

   ```powershell
   python src/prepare_data.py
   ```

   原始标签中 `0` 为负向、`1` 为正向。

3. 建立 TF-IDF + Logistic Regression 基线：

   ```powershell
   python src/train_baseline.py
   ```

4. 微调 MacBERT。8 GB 显存默认设置为长度 128、微批次 8、梯度累积 2、FP16：

   ```powershell
   python src/train_macbert.py
   ```

   如显存不足可减小 `--batch-size 4`；需要更完整训练可设置 `--epochs 3`。模型和报告写入 `models/`、`reports/`。

5. 训练 ASAP 餐饮方面级模型（官方训练/验证/测试集）：

   ```powershell
   python src/train_asap_absa.py --epochs 3 --batch-size 4 --grad-accum 2 --max-length 384
   ```

   结果输出到 `reports/asap_metrics.json`，方面级权重放入 `models/asap_macbert/`。本地语料来源、类别和授权说明见 `data/DATA_SOURCE.md`。

6. 使用真实商家数据训练质量、物流、退款、建议、购买意向分类器：

   从 `data/annotation_template.csv` 制作人工标注数据，并准备互不重叠的训练/验证/测试 CSV（至少有 `review_text` 与 `issue_types` 两列）。`issue_types` 用分号分隔，标签只能使用：`质量问题;物流问题;退款退货;改进建议;购买意向`。建议按时间或客户分割，避免重复评论泄漏。

   ```powershell
   python src/train_voc_multilabel.py --train-csv data/merchant_train.csv --dev-csv data/merchant_dev.csv --test-csv data/merchant_test.csv
   ```

   该步骤需使用商家授权的真实标注数据。模型会在验证集上为每类调阈值，测试集报告 Macro/Micro-F1 与各类精确率、召回率；需要每类有足够的正例。

7. 打开网页前端：

   ```powershell
   python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
   ```

   页面支持单条反馈和 CSV 批量分析，并可下载分类结果。

8. 命令行单条情绪预测：

   ```powershell
   python src/predict.py "这台机器用了两周就漏水了"
   ```

## 质量判断

商品情绪模型报告 Accuracy、Macro-F1 和各类指标。ASAP 模型报告方面提及 F1、方面-情绪联合 F1、方面情绪 F1。它们只能支持各自公开数据的任务范围；规则提示不是模型置信度。

下一步请从愿意试点的商家取得授权导出数据，去除姓名、电话、订单号等个人信息。先统一标签定义，人工复核至少 2,000 条，按时间或商品划分留出测试集；高风险投诉优先看召回率。置信度不足的反馈交给人工确认，再将确认结果用于迭代训练。目标分数应由真实验证集决定，不能预先保证。

## 文件结构

- `src/prepare_data.py`：清洗、去重、分层切分。
- `src/train_baseline.py`：TF-IDF 字符 n-gram 基线。
- `src/train_macbert.py`：MacBERT 序列分类微调与评估。
- `src/train_asap_absa.py`：MacBERT 餐饮方面级情绪训练与评估。
- `src/asap_model.py`：18 个方面类别和方面模型结构。
- `src/rule_triage.py`：质量、物流、退款、建议、购买意向和优先级规则提示。
- `src/train_voc_multilabel.py`：使用商家标注数据微调五类 VOC 模型。
- `src/voc_model.py`：商家 VOC 模型结构与加载逻辑。
- `src/predict.py`：单条反馈预测。
- `app.py`：Streamlit 本地网页，支持单条与 CSV 批量预测。
- `data/label_schema.csv`：后续商家 VOC 标签定义。
- `data/annotation_template.csv`：商家反馈标注表格模板。
- `reports/`：评估结果。


## 代码注释与格式规范

Python 源文件使用中文模块说明、类/函数 docstring 和必要的行内注释解释关键设计决策。格式由 `pyproject.toml` 中的 Black 配置统一管理（行宽 88，目标 Python 3.10+）；`.editorconfig` 约定 UTF-8、LF 换行和 4 空格缩进。

```powershell
# 格式化
black --config pyproject.toml app.py src

# 检查格式但不改文件
black --check --config pyproject.toml app.py src
```

运行网页前，需要先训练整体情绪模型；若未安装 ASAP 权重，页面会跳过餐饮方面分析。模型权重和原始评论语料因体积及授权考虑不纳入 Git，需要在本地下载/训练生成。
