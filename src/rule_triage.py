"""提供可解释的 VOC 关键词初筛、风险优先级和运营动作建议。"""

import re

# 每类可配置多条正则；这里的命中只是线索，不等同于模型置信度或最终判定。
RULES = {
    "质量问题": [
        r"坏了|损坏|故障|漏水|漏电|无法使用|不能用|开不了机|失灵|破损|裂了|碎了|发霉|变质|异物|不新鲜|质量差|有问题",
    ],
    "物流问题": [
        r"物流|快递|配送|外卖|骑手|送餐|送达|没送到|未送到|送错|送晚|延迟|超时|配送慢|包装破|撒了|洒了|少送|漏送|丢件",
    ],
    "退款退货": [
        r"退款|退货|退钱|退费|换货|退单|取消订单|申请赔偿|要求赔偿|售后处理|不想要了",
    ],
    "改进建议": [
        r"建议|希望|要是.*就好|如果能|最好能|可以增加|能不能增加|建议增加|建议改|建议优化|改进一下|希望改进|希望增加",
    ],
    "购买意向": [
        r"想买|准备买|打算买|怎么买|哪里买|多少钱|价格多少|有现货吗|可以预订|想预订|想订|批发|采购|再买|回购|购买链接",
    ],
}

ACTION = {
    "质量问题": "转售后核查商品/批次，必要时联系客户",
    "物流问题": "核对订单、配送时间与缺漏，转配送/门店处理",
    "退款退货": "创建退款/退货待办，由客服按订单规则处理",
    "改进建议": "进入产品/运营建议池，合并相似建议后评估",
    "购买意向": "转销售或门店跟进，并记录商品与时间需求",
}

# 安全/人身风险词优先级最高，需要尽快由人工查看原文。
HIGH_RISK = re.compile(
    r"食物中毒|过敏|受伤|烫伤|安全事故|异物|发霉|变质|漏电|起火|威胁|投诉到市场监管|报警"
)


def classify_intents(text):
    """返回命中的业务类别、规则优先级、建议动作及结果来源说明。"""
    text = str(text or "")
    matched = [
        label
        for label, patterns in RULES.items()
        if any(re.search(p, text, re.I) for p in patterns)
    ]
    urgency = (
        "高"
        if HIGH_RISK.search(text)
        else ("中" if "退款退货" in matched or "质量问题" in matched else "低")
    )
    actions = [ACTION[label] for label in matched]
    if not actions:
        actions = ["暂未命中规则；按低置信度反馈进入人工复核"]
    return {
        "categories": matched,
        "urgency": urgency,
        "actions": actions,
        "source": "规则提示，需人工确认",
    }


def actions_for_aspects(aspects):
    """根据餐饮方面分析结果汇总门店可执行的运营建议。"""
    negative = {x["方面"] for x in aspects if x["情绪"] == "负向"}
    positive = {x["方面"] for x in aspects if x["情绪"] == "正向"}
    actions = []
    if negative.intersection({"口味", "分量", "菜品外观", "卫生"}):
        actions.append("复核菜品品质与卫生反馈，按门店/菜品聚合排查")
    if negative.intersection({"服务态度", "排队等位", "服务效率", "停车便利"}):
        actions.append("检查高峰排班、等位时长和现场服务流程")
    if negative.intersection({"价格水平", "性价比", "优惠活动"}):
        actions.append("复核价格与促销说明，比较顾客提到的价值感受")
    if negative.intersection({"装修", "噪音", "空间"}):
        actions.append("检查环境与空间体验，评估可执行的门店改进")
    if not actions and positive:
        actions.append("汇总正向方面作为门店优势，供运营复用")
    return actions
