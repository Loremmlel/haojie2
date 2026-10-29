"""每个 worker 独占一个可复用槽；Pipe 交接所有权，大数组不经过 pickle。"""

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
