"""可复用 pinned 输入槽；完成传输事件之前不能覆盖槽位。"""

import torch


class Staging:
    def __init__(self, device):
        self.device = torch.device(device)
        self.slots = {}

    def transfer(self, name, value):
        if self.device.type != "cuda":
            return value.to(self.device)
        slot = self.slots.get(name)
        if slot is not None:
            buffer, event = slot
            event.synchronize()
            if buffer.dtype != value.dtype or buffer.numel() < value.numel():
                slot = None
        if slot is None:
            buffer = torch.empty(max(value.numel(), 1), dtype=value.dtype, pin_memory=True)
            event = torch.cuda.Event()
            self.slots[name] = buffer, event
        view = buffer[: value.numel()].view(value.shape)
        view.copy_(value)
        result = view.to(self.device, non_blocking=True)
        event.record(torch.cuda.current_stream(self.device))
        return result

    @property
    def bytes(self):
        return sum(buffer.numel() * buffer.element_size() for buffer, _ in self.slots.values())
