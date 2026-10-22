"""Notebook 的纯标准库 SVG 监控；只读取公开状态，不访问训练器或发出控制命令。"""

import html
import json
import math
import time
from collections import deque
from pathlib import Path

from ..console.opponents import summarize


def number(value):
    return f"{value:,.2f}".rstrip("0").rstrip(".") if finite(value) else "暂无"


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def chart(title, series, x_label, *, percent=False):
    """空值留白，横轴使用真实更新数或经过分钟；标题和数据均按 HTML 边界转义。"""
    points = [(x, y) for _, _, values in series for x, y in values if finite(x) and finite(y)]
    if not points:
        body = '<p class="empty">等待实际数据</p>'
    else:
        low_x, high_x = min(x for x, _ in points), max(x for x, _ in points)
        low_y = min(0, min(y for _, y in points))
        high_y = max(100 if percent else 0.01, max(y for _, y in points))
        paths = []
        for label, color, values in series:
            segment = []
            for x, y in [*values, (None, None)]:
                if not finite(x) or not finite(y):
                    if segment:
                        paths.append(
                            f'<polyline fill="none" stroke="var(--{color})" stroke-width="2" '
                            f'points="{" ".join(segment)}"/>'
                        )
                        segment = []
                    continue
                px = 46 + (x - low_x) / max(1e-9, high_x - low_x) * 430
                py = 150 - (y - low_y) / (high_y - low_y) * 124
                segment.append(f"{px:.2f},{py:.2f}")
                paths.append(
                    f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2" fill="var(--{color})">'
                    f"<title>{html.escape(label)}：{number(y)} · {number(x)}</title></circle>"
                )
        body = (
            f'<svg viewBox="0 0 500 190" role="img" aria-label="{html.escape(title)}">'
            '<path d="M46 26V150H476" fill="none" stroke="var(--line)"/>'
            f'<text x="2" y="30">{number(high_y)}</text>'
            f'<text x="2" y="153">{number(low_y)}</text>'
            f'<text x="46" y="170">{number(low_x)}</text>'
            f'<text x="476" y="170" text-anchor="end">{number(high_x)}</text>'
            f'<text x="260" y="187" text-anchor="middle">{x_label}</text>'
            + "".join(paths)
            + "</svg>"
        )
    legend = " · ".join(
        f'<span style="color:var(--{color})">{html.escape(label)}</span>'
        for label, color, _ in series
    )
    return f"<section><h3>{html.escape(title)}</h3><p>{legend}</p>{body}</section>"


def evaluation_tables(status):
    """实时批次和最近完成批次分别呈现；旧摘要缺少逐局记录时只展示已有总计。"""

    def rate(row):
        if not row or not row.get("n") or not finite(row.get("win_rate")):
            return "未评测"
        return f"{number(row['win_rate'] * 100)}%（{row['n']}局）"

    def table(report, title):
        results = (
            summarize(report["games"], breakdown=True)
            if "games" in report
            else report.get("results", {})
        )
        rows, details = [], []
        for key, label in (
            ("easy", "简单"),
            ("medium", "中等"),
            ("hard", "困难"),
            ("baseline", "初始模型"),
        ):
            row = results.get(key, {})
            seats = row.get("by_side", {})
            rows.append(
                f'<tr><th scope="row">{label}</th>'
                f"<td>{row.get('wins', 0)} / {row.get('losses', 0)} / {row.get('draws', 0)}</td>"
                f"<td>{row.get('n', 0)} / {row.get('total', 0)}</td><td>{rate(row)}</td>"
                f"<td>{rate(seats.get('1'))}</td><td>{rate(seats.get('2'))}</td>"
                f"<td>{row.get('unfinished', 0)} / {row.get('errors', 0)}</td></tr>"
            )
            for matchup, part in row.get("by_matchup", {}).items():
                rules, side = matchup.split("/")
                name = f"{label} · {'经典' if rules == 'classic' else '神龛'} · "
                name += "模型先手" if side == "1" else "模型后手"
                details.append(
                    f'<tr><th scope="row">{name}</th><td>{rate(part)}</td>'
                    f"<td>{part['wins']} / {part['losses']} / {part['draws']}</td>"
                    f"<td>{part['n']} / {part['total']}</td>"
                    f"<td>{part['unfinished']} / {part['errors']}</td></tr>"
                )
        heading = html.escape(title)
        identity = html.escape(str(report.get("model", "未知"))[:12])
        content = (
            f"<h3>{heading}</h3><p>冻结模型 {identity} · 更新 {number(report.get('updates'))}</p>"
            f'<div class="table-scroll" tabindex="0" role="region" aria-label="{heading}">'
            f'<table><caption>{heading}</caption><thead><tr><th scope="col">对手</th>'
            '<th scope="col">胜 / 负 / 平</th><th scope="col">有效 / 计划</th>'
            '<th scope="col">总胜率</th><th scope="col">模型先手</th><th scope="col">模型后手</th>'
            '<th scope="col">未完成或待测 / 错误</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
        )
        if details:
            content += (
                "<details><summary>按经典 / 神龛与先后手细分</summary>"
                '<div class="table-scroll" tabindex="0" role="region" aria-label="规则与座位细分">'
                '<table><caption>规则与座位细分</caption><thead><tr><th scope="col">对战组合</th>'
                '<th scope="col">胜率</th><th scope="col">胜 / 负 / 平</th>'
                '<th scope="col">有效 / 计划</th><th scope="col">未完成或待测 / 错误</th>'
                "</tr></thead><tbody>" + "".join(details) + "</tbody></table></div></details>"
            )
        return content

    config = status.get("config", {})
    interval = number(config.get("eval_games"))
    remaining = number(status.get("next_eval_games"))
    info = (
        '<section class="evaluation"><h2>手工 AI 对战评测</h2>'
        f"<p>自动间隔：{interval} 个有效训练终局，并需有新更新；距下次还需 {remaining} 局。"
        "训练中每次最多评测四局，再继续训练一轮；整批使用同一份冻结权重。</p>"
    )
    if status.get("evaluation_ready") is False:
        info += '<p role="alert">手工 AI 评测未就绪，请检查随包教师与 mini-racer 安装。</p>'
    job = status.get("evaluation_job")
    if job:
        completed, planned = job["completed_tasks"], job["planned"]
        info += (
            f'<p role="status">本轮评测进度：{completed} / {planned} 局'
            "（包含未完成或错误任务）</p>"
            f'<progress value="{completed}" max="{planned}" aria-label="本轮评测进度"></progress>'
            + table(job, "本轮进行中")
        )
    elif status.get("eval_pending"):
        info += '<p role="status">评测已排队，等待当前训练阶段完成。</p>'
    reports = status.get("evaluations", [])
    if reports:
        recent = table(reports[-1], "最近完成批次")
        info += f"<details><summary>最近完成批次</summary>{recent}</details>" if job else recent
    elif not job:
        info += "<p>尚无评测结果；达到自动间隔后执行，也可点击“立即评测”。</p>"
    return info + (
        "<p>胜率 = 胜局 / 真实终局；平局计入分母，超时、截断与待测归入未完成，错误另计。"
        "每格注明有效局数，不累计不同权重版本。</p></section>"
    )


class Dashboard:
    """保留最近 180 次刷新；换实验或恢复回退时重置速率，避免跨会话连接曲线。"""

    def __init__(self):
        self.history = deque(maxlen=180)
        self.previous = None
        self.started = time.monotonic()
        self.identity = None

    def render(self, status, *, now=None):
        now = time.monotonic() if now is None else now
        counts = status.get("counts", {})
        retained = counts.get("retained_samples", 0)
        updates = status.get("updates", 0)
        identity = (status.get("experiment"), status.get("restoration"))
        if (
            identity != self.identity
            or self.previous is None
            or retained < self.previous[1]
            or updates < self.previous[2]
        ):
            self.history.clear()
            self.previous = None
            self.started = now
            self.identity = identity
        rate = None
        if self.previous and now > self.previous[0]:
            rate = (retained - self.previous[1]) / (now - self.previous[0])
        resources = status.get("resources", {})
        self.history.append(
            (
                (now - self.started) / 60,
                rate,
                resources.get("cpu_percent"),
                resources.get("gpu_utilization"),
            )
        )
        self.previous = (now, retained, updates)
        losses = status.get("losses", [])
        curves = chart(
            "训练损失",
            [
                ("策略", "teal", [(r["step"], r.get("policy")) for r in losses]),
                ("价值", "ember", [(r["step"], r.get("value")) for r in losses]),
            ],
            "累计更新次数",
        )
        curves += chart(
            "有效终局样本 / 秒",
            [("每次刷新间隔", "teal", [(r[0], r[1]) for r in self.history])],
            "监控经过分钟",
        )
        curves += chart(
            "计算利用率（%）",
            [
                ("CPU / 有效配额", "teal", [(r[0], r[2]) for r in self.history]),
                ("GPU", "ember", [(r[0], r[3]) for r in self.history]),
            ],
            "监控经过分钟",
            percent=True,
        )
        performance = status.get("performance", {})
        phase_names = {
            "idle": "空闲",
            "sampling": "采样",
            "updating": "更新",
            "evaluating": "评测",
            "saving": "保存",
        }
        state_names = {
            "idle": "空闲",
            "paused": "已暂停",
            "starting": "正在启动",
            "sampling": "采样中",
            "updating": "更新中",
            "evaluating": "评测中",
            "saving": "保存中",
            "pausing": "正在暂停",
            "error": "训练出错",
        }
        concurrency = status.get("concurrency", {})
        evaluating = status.get("state") == "evaluating"
        backup = status.get("backup") or {}
        tiles = [
            ("累计更新 / 已保存", f"{number(updates)} / {number(status.get('saved_step'))}"),
            (
                "真实终局 / 截断",
                f"{number(counts.get('games', 0))} / {number(counts.get('truncated', 0))}",
            ),
            ("有效保留样本", number(retained)),
            (
                "评测进程 / 训练并发目标" if evaluating else "并发进程 / 目标",
                f"{number(concurrency.get('actual'))} / "
                f"{number(concurrency.get('requested'))}"
                + (" · 待生效" if concurrency.get("pending") and not evaluating else ""),
            ),
            (
                "进程树 RAM（MiB）",
                number(resources["rss"] / 2**20) if finite(resources.get("rss")) else "暂无",
            ),
            (
                "GPU 已分配 / 预留（MiB）",
                " / ".join(
                    number(resources[k] / 2**20) if finite(resources.get(k)) else "暂无"
                    for k in ("gpu_allocated", "gpu_reserved")
                ),
            ),
        ]
        metrics = "".join(
            f"<section><p>{label}</p><strong>{html.escape(value)}</strong></section>"
            for label, value in tiles
        )
        error = status.get("error") or backup.get("error")
        error_html = f'<p role="alert">{html.escape(str(error))}</p>' if error else ""
        tokens = (Path(__file__).parents[1] / "console/assets/tokens.css").read_text(
            encoding="utf-8"
        )
        details = html.escape(json.dumps(status, ensure_ascii=False, indent=2))
        return f"""<style>{tokens}
.hj-colab.hj-game {{min-height:0;padding:18px;border:1px solid var(--line);
  font-variant-numeric:tabular-nums;scrollbar-color:var(--teal) var(--paper)}}
.hj-colab h2 {{font-family:SimSun,serif;font-size:22px}}
.hj-colab h3 {{font-size:15px}} .hj-colab p {{font-size:12px}}
.hj-colab .metrics,.hj-colab .curves {{display:grid;gap:12px;margin:14px 0}}
.hj-colab .metrics {{grid-template-columns:repeat(3,minmax(0,1fr))}}
.hj-colab .curves {{grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr))}}
.hj-colab section {{min-width:0;background:var(--panel);border:1px solid var(--line);padding:12px}}
.hj-colab strong {{display:block;font-size:20px}} .hj-colab svg {{width:100%;display:block}}
.hj-colab text {{fill:var(--ink);font-size:15px}}
.hj-colab .empty {{height:190px;display:grid;place-items:center}}
.hj-colab [role=alert] {{color:var(--ember);min-height:24px}}
.hj-colab pre {{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}
.hj-colab details {{margin-top:10px}}
.hj-colab .evaluation {{margin:14px 0}} .hj-colab progress {{width:100%;accent-color:var(--teal)}}
.hj-colab .table-scroll {{overflow:auto;scrollbar-color:var(--teal) var(--paper)}}
.hj-colab .table-scroll:focus-visible {{outline:3px solid var(--teal);outline-offset:2px}}
.hj-colab table {{width:100%;border-collapse:collapse;font-size:12px}}
.hj-colab th,.hj-colab td {{padding:8px;text-align:left;border-bottom:1px solid var(--line)}}
.hj-colab caption {{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}}
.hj-colab .evaluation h3 {{margin-top:12px}}
@media(max-width:650px) {{.hj-colab .metrics {{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:420px) {{.hj-colab .metrics {{grid-template-columns:1fr}}}}
@media(forced-colors:active) {{.hj-colab {{scrollbar-color:auto}}}}
</style><div class="hj-game hj-colab">
<h2>浩劫 · 训练观测</h2><p>{state_names.get(status.get("state"), "状态未知")} ·
{phase_names.get(performance.get("phase"), "等待阶段")} ·
{html.escape(str(status.get("device", "")))} ·
备份：{html.escape(backup.get("state") or "未配置")}</p>
{error_html}<div class="metrics">{metrics}</div><div class="curves">{curves}</div>
{evaluation_tables(status)}
<p>损失来自服务记录；吞吐与利用率保留最近 180 次刷新。无可用读数时留白。</p>
<details><summary>详细状态与排错信息</summary><pre>{details}</pre></details></div>"""
