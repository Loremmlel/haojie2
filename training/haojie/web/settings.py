"""页面字段描述与训练配置共用默认值和校验。"""

import math
from dataclasses import asdict
from ..runtime.config import Config, cpu_limit

# 名称、分组和范围只描述可操作参数，最终组合约束由 Config 检查。
FIELDS = [
    ("workers", "搜索线程", "并发与资源", 1, 128, 1, "Rust 工作线程数"),
    (
        "games_per_worker",
        "每线程并行局数",
        "并发与资源",
        1,
        128,
        1,
        "等待推理时继续推进其他对局",
    ),
    ("batch", "学习批量", "并发与资源", 1, 1024, 1, "每次更新消费的样本数"),
    (
        "inference_batch",
        "推理批量上限",
        "并发与资源",
        1,
        1024,
        1,
        "每批最多接收的神经请求",
    ),
    (
        "replay_mib",
        "回放内存 · MiB",
        "并发与资源",
        1,
        65536,
        256,
        "回放张量预算，超限淘汰旧局",
    ),
    (
        "replay_games",
        "回放局数上限",
        "并发与资源",
        1,
        4096,
        1,
        "与内存预算共同限制回放窗口",
    ),
    ("simulations", "深搜模拟次数", "搜索", 1, 4096, 1, "完整命令级 PUCT 预算"),
    ("fast_simulations", "快搜模拟次数", "搜索", 1, 4096, 1, "快搜保留实际轨迹"),
    ("full_fraction", "深搜比例", "搜索", 0, 1, 0.05, "深搜步形成策略监督"),
    ("depth", "搜索深度", "搜索", 1, 128, 1, "以完整命令计数"),
    ("temperature", "采样温度", "搜索", 0, 5, 0.1, "0 为选择访问次数最多的动作"),
    (
        "learning_rate",
        "学习率",
        "学习与对手",
        0.000001,
        0.1,
        0.00001,
        "同步应用到 AdamW 参数组",
    ),
    (
        "sample_reuse",
        "样本复用额度",
        "学习与对手",
        0,
        100,
        0.5,
        "每个新样本提供的抽样次数",
    ),
    ("publish_updates", "模型发布间隔", "学习与对手", 1, 10000, 1, "单位：学习更新"),
    ("history_games", "历史入池间隔", "学习与对手", 1, 10000, 1, "单位：自然终局"),
    (
        "history_fraction",
        "历史对手比例",
        "学习与对手",
        0,
        1,
        0.05,
        "其余由当前自对弈和手工对手组成",
    ),
    (
        "teacher_fraction",
        "手工对手比例",
        "学习与对手",
        0,
        1,
        0.05,
        "与历史比例之和不超过 1",
    ),
    (
        "max_ply",
        "单方回合上限",
        "对局与目标",
        0,
        10000,
        1,
        "0 为关闭软截断，新分配局生效",
    ),
    (
        "full_game_fraction",
        "完整局比例",
        "对局与目标",
        0,
        1,
        0.05,
        "新分配局随机豁免回合上限",
    ),
    ("max_commands", "每局命令上限", "对局与目标", 1, 100000, 1, "超限保持未知结果"),
    (
        "calibration_games",
        "校准终局门槛",
        "对局与目标",
        5,
        10000,
        1,
        "仍需通过留出集检验",
    ),
    (
        "bootstrap_after_games",
        "自举终局门槛",
        "对局与目标",
        5,
        100000,
        1,
        "不小于校准门槛",
    ),
    (
        "bootstrap_weight",
        "截断价值权重",
        "对局与目标",
        0.01,
        1,
        0.05,
        "真实终局权重保持 1",
    ),
    (
        "chunk_commands",
        "样本打包间隔",
        "保存与停止",
        1,
        1024,
        1,
        "单位：命令；在内存打包",
    ),
    (
        "save_seconds",
        "自动保存 · 秒",
        "保存与停止",
        0,
        86400,
        30,
        "0 为只在正常停止时保存",
    ),
    (
        "seconds",
        "本次时限 · 秒",
        "保存与停止",
        0,
        604800,
        60,
        "0 为不限；每次继续重新计时",
    ),
    (
        "evaluation_seconds",
        "对弈评测间隔 · 秒",
        "定时评测",
        0,
        604800,
        60,
        "0 为关闭；每轮三档各 4 局，上一轮未结束时不重叠启动",
    ),
    ("games", "本次自然终局上限", "保存与停止", 0, 10000000, 1, "0 为不限"),
    ("updates", "本次更新上限", "保存与停止", 0, 100000000, 1, "0 为不限"),
    ("warmup_games", "示范终局数", "新训练初始化", 0, 10000, 1, "仅新建时可调整"),
    ("warmup_updates", "模仿更新次数", "新训练初始化", 0, 100000, 1, "仅新建时可调整"),
    ("seed", "训练随机种子", "新训练初始化", 0, 2**32 - 1, 1, "仅新建时可调整"),
]
INITIAL = {"seed", "warmup_games", "warmup_updates", "device"}


def defaults():
    return asdict(Config(workers=min(8, cpu_limit())))


def schema():
    values = defaults()
    return [
        dict(
            key=k,
            label=label,
            group=group,
            min=low,
            max=high,
            step=step,
            help=help_text,
            integer=isinstance(values[k], int),
            initial=k in INITIAL,
        )
        for k, label, group, low, high, step, help_text in FIELDS
    ]


def validate(values, previous=None):
    unknown = set(values) - set(defaults())
    if unknown:
        raise ValueError("未知参数：" + ", ".join(sorted(unknown)))
    config = {**defaults(), **(previous or {}), **values}
    for field in schema():
        value = config[field["key"]]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not field["min"] <= value <= field["max"]
            or field["integer"]
            and int(value) != value
        ):
            raise ValueError(
                f"{field['label']}须为 {field['min']}–{field['max']} 范围内的"
                + ("整数" if field["integer"] else "数值")
            )
        if field["integer"]:
            config[field["key"]] = int(value)
    if config["device"] not in ("cuda", "cpu"):
        raise ValueError("设备仅支持 cuda 或 cpu")
    if previous and any(config[k] != previous[k] for k in INITIAL):
        raise ValueError("随机种子、预训练和设备仅能在新建训练时修改")
    if config["workers"] * config["games_per_worker"] > 1024:
        raise ValueError("并行对局总数不能超过 1024")
    if config["teacher_fraction"] + config["history_fraction"] > 1:
        raise ValueError("历史对手比例与手工对手比例之和不能超过 1")
    Config(**config).validate()
    return config
