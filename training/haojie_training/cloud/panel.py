"""Colab 面板回调：单元格立即返回，浏览器只提交现有回环 API 支持的控制意图。"""

import html
import json
import uuid

from ..console.opponents import ASSETS
from .notebook import Dashboard

LABELS = {
    "start": "开始",
    "pause": "暂停",
    "save": "保存本地",
    "backup": "备份到 Drive",
    "evaluate": "立即评测",
    "cancel-evaluation": "取消评测",
    "settings": "应用设置",
    "concurrency": "应用并发量",
}


class Panel:
    """每次展示有独立身份，旧输出失效；回调串行执行，不拥有训练进程生命周期。"""

    def __init__(self, api):
        self.api = api
        self.identity = "haojie-" + uuid.uuid4().hex
        self.dashboard = Dashboard()
        self.closed = False

    def close(self):
        self.closed = True

    def call(self, identity, action="status", values=None):
        if self.closed or identity != self.identity:
            return {"closed": True}
        try:
            if action != "status" and action not in LABELS:
                raise ValueError("面板不支持此操作")
            if action != "status":
                self.api(action=action, values=values)
            status = self.api()
            message = ""
            if action == "concurrency":
                requested = values["environments"]
                confirmed = status["config"]["environments"]
                if confirmed != requested:
                    raise RuntimeError(
                        f"服务目标仍为 {confirmed}，请求值为 {requested}；请重新读取后重试"
                    )
                message = f"服务已确认并发目标 {confirmed}；原局保留，缩容等待终局"
            elif action != "status":
                message = LABELS[action] + "请求已接收"
            return {
                "html": self.dashboard.render(status),
                "config": status["config"],
                "message": message,
            }
        except Exception as error:
            return {"error": str(error)}

    def render(self):
        initial = self.call(self.identity)
        if "error" in initial:
            raise RuntimeError(initial["error"])
        config = initial["config"]

        def field(key, label, low, high):
            value = html.escape(str(config[key]), quote=True)
            return (
                f'<label for="{self.identity}-{key}">{label}'
                f'<input id="{self.identity}-{key}" name="{key}" type="number" '
                f'min="{low}" max="{high}" step="1" required value="{value}"></label>'
            )

        buttons = "".join(
            f'<button type="button" data-action="{key}">{label}</button>'
            for key, label in LABELS.items()
            if key not in ("settings", "concurrency")
        )
        evaluation = "".join(
            [
                field("eval_games", "自动间隔（有效终局）", 1, 100000),
                field("eval_pairs", "每规则换座对数", 1, 20),
                field("eval_seconds", "每局时限（秒）", 2, 7200),
            ]
        )
        script = (ASSETS / "notebook.js").read_text(encoding="utf-8")
        script = script.replace("__PANEL_ID__", json.dumps(self.identity))
        return f"""<style>
.hj-panel.hj-game {{min-height:0;padding:18px;border:1px solid var(--line)}}
.hj-panel .hj-colab.hj-game {{padding:0;border:0}}
.hj-panel .controls,.hj-panel form {{display:flex;gap:10px;flex-wrap:wrap;align-items:end}}
.hj-panel form {{margin:12px 0}} .hj-panel label {{display:grid;gap:4px;font-size:12px}}
.hj-panel button,.hj-panel input,.hj-panel textarea {{font:inherit;color:var(--ink);
  background:var(--panel);border:1px solid var(--line);padding:8px}}
.hj-panel button {{cursor:pointer;min-height:40px}}
.hj-panel button:hover {{background:var(--teal-soft)}}
.hj-panel button:active {{border-color:var(--teal)}}
.hj-panel button:disabled {{cursor:wait;opacity:.6}}
.hj-panel :is(button,input,textarea):focus-visible {{outline:2px solid var(--teal);
  outline-offset:2px}}
.hj-panel input {{width:150px;max-width:100%}} .hj-panel textarea {{width:100%;resize:none}}
.hj-panel [data-notice] {{min-height:50px;font-size:13px;overflow-wrap:anywhere}}
.hj-panel [role=alert] {{color:var(--ember)}} .hj-panel [data-settings] {{margin:12px 0}}
.hj-panel .json-settings {{width:100%}} .hj-panel .json-settings label {{width:100%}}
</style><div id="{self.identity}" class="hj-game hj-panel">
<div class="controls">{buttons}<button type="button" data-refresh>暂停刷新</button></div>
<form data-kind="concurrency" novalidate>
{field("environments", "并发环境（1–128）", 1, 128)}
<button type="submit">应用并发量</button></form>
<p>调整并发保留在途对局、模型、优化器与已完成样本。扩容逐步加入环境，缩容等待多余对局结束。目标值由服务确认后显示。</p>
<details data-settings><summary>其他设置（暂停后应用）</summary>
<form data-kind="evaluation" novalidate>{evaluation}
<button type="submit">应用评测设置</button></form>
<p>每对包含模型先后手各一局，经典/神龛、三档手工 AI 和初始模型共 16 局 × 对数。</p>
<form data-kind="json" class="json-settings" novalidate>
<label for="{self.identity}-json">设置覆盖（JSON）</label>
<textarea id="{self.identity}-json" rows="4">{{}}</textarea>
<button type="submit">应用设置</button></form></details>
<div data-notice role="status" aria-live="polite">面板已连接，自动刷新中。</div>
<div data-view>{initial["html"]}</div></div><script>{script}</script>"""
