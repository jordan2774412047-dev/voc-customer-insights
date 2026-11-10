# 数据来源与限制

当前 CSV 为 `online_shopping_10_cats` 中文商品评论子集的镜像文件，下载地址：

https://huggingface.co/datasets/XiangPan/online_shopping_10_cats_62k/resolve/main/online_shopping_10_cats.csv

上游中文语料库说明其包含 10 个商品类别、六万多条评论和正负向标签：

https://github.com/SophonPlus/ChineseNlpCorpus

当前文件字段为 `cat,label,review`。它用于本地学习与原型验证。镜像页未提供清晰的商业再利用授权；不要将原始评论、基于它训练的权重或其衍生数据直接作为商业产品资产发布。商用前需核实上游授权，优先改用客户授权提供的数据。

## ASAP 中文餐饮方面级情感数据

- 上游仓库：https://github.com/Meituan-Dianping/asap
- 论文：Bu et al., NAACL 2021, https://aclanthology.org/2021.naacl-main.167/
- README 描述的数据：46,730 条大众点评餐饮评论；18 个方面类别的人工极性标注；官方训练/验证/测试集分别为 36,850 / 4,940 / 4,940 条。
- 当前本地文件：`data/raw/asap_data/{train,dev,test}.csv`。原始方面标签值：1 正向、0 中性、-1 负向、-2 未提及。
- 仓库包含 Apache-2.0 许可文件；评论来源是第三方平台用户内容。使用时保留论文与数据集归属信息，并在对外销售/分发前单独核实评论内容及其衍生模型权重的适用授权范围。

该语料适合训练餐饮评论的细粒度方面/情绪识别（口味、分量、服务、价格、卫生、排队等）。它不标注退款、外卖配送投诉、功能建议或购买意向，因此这些标签必须由目标商家的授权数据训练，或以明确标注的规则辅助，并由人工确认。
