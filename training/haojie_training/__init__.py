"""浩劫训练工具；与浏览器发行包及权威规则引擎独立。"""

import os

# 在 Torch 导入前设置共享分配器选项，不加载或探测 CUDA；任一用户配置（包括空值）均保留。
# 已先导入 Torch 的嵌入宿主应在进程启动前设置环境；这里不重置已初始化的分配器。
if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
