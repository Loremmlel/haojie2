"""逐局与学习批次独占可复用共享槽；Pipe 只交接所有权和数组描述。"""

from multiprocessing.shared_memory import SharedMemory

import numpy as np


class Slot:
    def __init__(self, name=None, size=16 * 1024 * 1024):
        self.owner = name is None
        self.memory = SharedMemory(name=name, create=self.owner, size=size)

    @property
    def name(self):
        return self.memory.name

    def array(self, count, offset=0, dtype=np.float32):
        size = np.dtype(dtype).itemsize * count
        if offset + size > self.memory.size:
            raise BufferError(
                f"共享槽需要 {offset + size} 字节，当前 {self.memory.size}"
            )
        return np.ndarray((count,), dtype=dtype, buffer=self.memory.buf, offset=offset)

    def close(self):
        self.memory.close()
        if self.owner:
            self.memory.unlink()


def pack(slot, arrays):
    offset, shapes = 0, []
    for value in arrays:
        value = np.asarray(value)
        slot.array(value.size, offset, value.dtype)[:] = value.reshape(-1)
        shapes.append((value.shape, value.dtype.str))
        offset += value.nbytes
    return shapes


def unpack(slot, shapes, copy=False):
    arrays, offset = [], 0
    for shape, dtype in shapes:
        count = int(np.prod(shape))
        array = slot.array(count, offset, dtype).reshape(shape)
        arrays.append(array.copy() if copy else array)
        offset += array.nbytes
    return arrays


def search_values(bits):
    # NumPy 没有原生 BF16；仅在 CPU 搜索边界无损解码 GPU 输出，不参与神经网络计算。
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def batch_arrays(batch):
    arrays, rows = [], []
    for bundle, decisions, *labels in batch:
        arrays.extend(bundle)
        weights = []
        for options, target, weight in decisions:
            arrays.extend((options, target))
            weights.append(weight)
        arrays.append(np.asarray(labels[2]))
        rows.append((weights, [*labels[:2], *labels[3:]]))
    return arrays, rows


def restore_batch(arrays, rows):
    arrays = iter(arrays)
    result = []
    for weights, labels in rows:
        bundle = tuple(next(arrays) for _ in range(4))
        decisions = [(next(arrays), next(arrays), w) for w in weights]
        aux = next(arrays)
        result.append((bundle, decisions, *labels[:2], aux, *labels[2:]))
    return result
