"""已抽定的一步样本按长度分组；不重抽样、不改消费次数或优化器步数。"""


def update_groups(examples, config):
    """填充计算至少减少四分之一才拆组，最多四组；顺序和重复样本均保留身份。

    成本按实体主干线性层和注意力估计；MC 已选候选头无需按最大候选数计价。
    这里只选择执行形状，学习器仍以整步的标签数归一化，全部反向后统一裁剪更新。
    """
    if len(examples) < 2:
        return [examples]
    width = 2 * config.width + config.ffn

    def cost(rows):
        tokens = max(len(row["entities"]) for row in rows) + 1
        return len(rows) * tokens * (width + tokens)

    ordered = sorted(examples, key=lambda row: len(row["entities"]))
    groups = [[ordered[0]]]
    for row in ordered[1:]:
        if len(row["entities"]) > 2 * len(groups[-1][0]["entities"]):
            groups.append([])
        groups[-1].append(row)
    while len(groups) > 4:
        at = min(
            range(len(groups) - 1),
            key=lambda i: cost(groups[i] + groups[i + 1]) - cost(groups[i]) - cost(groups[i + 1]),
        )
        groups[at : at + 2] = [groups[at] + groups[at + 1]]
    return groups if sum(cost(g) for g in groups) <= 0.75 * cost(examples) else [examples]
