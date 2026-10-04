"""单批CPU预取，不改样本顺序、权重或按累计步骤确定的随机源。"""

from concurrent.futures import ThreadPoolExecutor

import torch

from ..data import sample_indices, select_batch, synthetic_batch


def training_batches(
    dataset,
    config,
    *,
    seed,
    start,
    steps,
    size,
    entities=64,
    actions=64,
    bucket_size=0,
    prefetch=False,
    pin_memory=False,
):
    def build(step):
        if dataset is None:
            batch = synthetic_batch(config, size, entities, actions, seed + step)
        else:
            rng = torch.Generator().manual_seed(seed + step)
            batch = select_batch(dataset, sample_indices(dataset, size, rng, bucket_size))
        return {k: v.pin_memory() for k, v in batch.items()} if pin_memory else batch

    if not prefetch:
        for step in range(start, start + steps):
            yield build(step)
        return
    # 一份在途结果、一份当前结果；GPU使用独立拷贝，主机张量不会被下一批覆盖。
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="training-batch") as executor:
        future = executor.submit(build, start) if steps else None
        for step in range(start, start + steps):
            batch = future.result()
            if step + 1 < start + steps:
                future = executor.submit(build, step + 1)
            yield batch
