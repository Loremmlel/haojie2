"""纯标准库验证 Notebook 刷新、中断隔离与图表口径；Linux 无需 Torch。"""

import ast
import asyncio
import html
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from haojie_training.cloud.notebook import Dashboard
from haojie_training.console.opponents import summarize

REPO = Path(__file__).resolve().parents[3]


def cells():
    notebook = json.loads((REPO / "notebooks/haojie_colab.ipynb").read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


class NotebookTests(unittest.TestCase):
    def test_notebook_and_bootstrap_syntax(self):
        code = cells()
        self.assertEqual(len(code), 5)
        for cell in code:
            ast.parse("".join(cell["source"]))
            self.assertEqual(cell["outputs"], [])
        ast.parse((REPO / "scripts/colab/bootstrap.py").read_text(encoding="utf-8"))

    def test_monitor_refresh_failure_and_cancel_never_send_commands(self):
        for fail in (False, True):
            view = Mock()
            api = (
                Mock(side_effect=OSError("连接失败"))
                if fail
                else Mock(return_value={"resources": {"cpu_percent": 25}})
            )

            async def stop_display(_):
                raise asyncio.CancelledError

            scope = {
                "display": lambda *args, **kwargs: view,
                "print": Mock(),
                "HTML": str,
                "html": html,
                "Dashboard": Dashboard,
                "asyncio": SimpleNamespace(
                    sleep=stop_display, CancelledError=asyncio.CancelledError
                ),
                "api": api,
            }
            panel = ast.parse("".join(cells()[-2]["source"]))
            start = next(
                i
                for i, node in enumerate(panel.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "dashboard" for t in node.targets)
            )
            monitor = compile(
                ast.Module(body=panel.body[start:], type_ignores=[]),
                "monitor-cell",
                "exec",
                ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )
            asyncio.run(eval(monitor, scope))
            api.assert_called_once_with()
            view.update.assert_called_once()
            if fail:
                self.assertIn("状态未知", view.update.call_args.args[0])
            else:
                self.assertEqual(scope["dashboard"].history[-1][2], 25)
            scope["print"].assert_called_once()

    def test_evaluation_breakdown_keeps_unknown_and_seat_denominators(self):
        games = [
            {"difficulty": "easy", "side": side, "start": {"rules": rules}, "result": result}
            for side, rules, result in (
                (1, "classic", "win"),
                (2, "classic", "loss"),
                (1, "shrine", "draw"),
                (2, "shrine", "unfinished"),
                (2, "shrine", "error"),
            )
        ]
        compact = summarize(games)
        detailed = summarize(games, breakdown=True)
        easy = detailed["easy"]
        self.assertEqual(easy["n"], 3)
        self.assertEqual(easy["win_rate"], 1 / 3)
        self.assertEqual(easy["by_side"]["1"]["win_rate"], 0.5)
        self.assertEqual(easy["by_side"]["2"]["win_rate"], 0)
        self.assertIsNone(easy["by_matchup"]["shrine/2"]["win_rate"])
        self.assertEqual(easy["by_matchup"]["shrine/2"]["errors"], 1)
        self.assertNotIn("by_side", compact["easy"])
        status = {
            "config": {"eval_games": 64},
            "next_eval_games": 12,
            "evaluations": [
                {
                    "model": "a" * 64,
                    "updates": 32,
                    "games": games,
                    "results": compact,
                }
            ],
        }
        shown = Dashboard().render(status)
        self.assertIn("50%（2局）", shown)
        self.assertIn("33.33%（3局）", shown)
        self.assertIn("经典 · 模型先手", shown)
        self.assertIn("距下次还需 12 局", shown)
        status["evaluation_job"] = {
            "model": "b" * 64,
            "updates": 64,
            "completed_tasks": 4,
            "planned": 5,
            "results": detailed,
        }
        shown = Dashboard().render(status)
        self.assertIn("本轮评测进度：4 / 5 局", shown)
        self.assertIn("冻结模型 aaaaaaaaaaaa", shown)
        self.assertIn("冻结模型 bbbbbbbbbbbb", shown)
        # 没有逐局记录的旧批次保留总计，不从别的版本补先后手。
        del status["evaluation_job"]
        del status["evaluations"][0]["games"]
        shown = Dashboard().render(status)
        self.assertIn("33.33%（3局）", shown)
        self.assertNotIn("50%（2局）", shown)

    def test_rates_unknown_values_reset_and_escaping(self):
        dashboard = Dashboard()
        status = {
            "experiment": "a",
            "updates": 1,
            "counts": {"retained_samples": 20},
            "losses": [{"step": 1, "policy": 0.4, "value": 0.2}],
        }
        initial = dashboard.render(status, now=10)
        self.assertEqual(initial.count("<svg "), 1)
        self.assertIsNone(dashboard.history[-1][1])
        status["counts"]["retained_samples"] = 40
        status["resources"] = {"cpu_percent": 0, "gpu_utilization": None}
        shown = dashboard.render(status, now=15)
        self.assertEqual(shown.count("<svg "), 3)
        self.assertEqual(dashboard.history[-1][1], 4)
        self.assertIsNone(dashboard.history[-1][3])
        for index in range(200):
            dashboard.render(status, now=20 + index)
        self.assertEqual(len(dashboard.history), 180)
        status.update(experiment="b", error="<script>bad()</script>")
        shown = dashboard.render(status, now=300)
        self.assertEqual(len(dashboard.history), 1)
        self.assertIsNone(dashboard.history[0][1])
        self.assertNotIn("<script>", shown)
        self.assertIn("&lt;script&gt;", shown)
        status["counts"]["retained_samples"] = 0
        dashboard.render(status, now=305)
        self.assertEqual(len(dashboard.history), 1)

    @unittest.skipIf(os.name == "nt", "进程组中断使用 Linux 信号语义")
    def test_notebook_launch_survives_process_group_interrupt(self):
        # 外层驱动自成进程组，核对 Notebook 原始 Popen 表达式，不触及测试宿主进程组。
        driver = r"""
import ast, json, os, signal, subprocess, sys, tempfile, time
from pathlib import Path
repo = Path(sys.argv[1])
notebook = json.loads((repo / "notebooks/haojie_colab.ipynb").read_text())
source = "".join(notebook["cells"][4]["source"])
launch = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen")
signal.signal(signal.SIGINT, lambda *_: None)
with tempfile.TemporaryFile() as log:
    command = [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(60)"]
    REPO = repo
    process = eval(compile(ast.Expression(launch), "launch", "eval"))
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            log.seek(0)
            if b"ready" in log.read():
                break
            assert process.poll() is None
            time.sleep(0.05)
        else:
            raise AssertionError("子进程未就绪")
        os.killpg(os.getpgrp(), signal.SIGINT)
        time.sleep(0.2)
        assert process.poll() is None, "Notebook 中断终止了后台服务"
        assert os.getsid(process.pid) == process.pid
    finally:
        process.terminate()
        process.wait(timeout=10)
"""
        subprocess.run(
            [sys.executable, "-c", driver, str(REPO)],
            check=True,
            timeout=20,
            start_new_session=True,
        )
