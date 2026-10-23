"""注入公开帧验证控制层连接，不把双方轮流作为轨迹协议的前提。"""

import unittest
from unittest.mock import patch

from test_contracts import example

from haojie_training.online.model import Config
from haojie_training.online.session import Session, Settings


class SeatConnection(unittest.TestCase):
    def test_reactions_repeated_seats_pause_and_terminal(self):
        session = Session(
            Settings(concurrency=0, workers=1, block=8),
            "cpu",
            Config(width=32, layers=1, heads=4, ffn=64, history=16),
        )
        session.games[1] = dict(
            worker=0,
            learner=1,
            opponent=0,
            commands=0,
            microsteps=0,
            failures=0,
            backtracks=0,
            waiting=False,
            seed=71,
        )
        count = 0

        def receive(root_id, actor, *, selected=0, kind=0, winner=-1):
            nonlocal count
            count += int(selected >= 0)
            root, node = example(key=(1, root_id))
            node["history_start"] = 0
            frame = dict(
                game=1,
                root=root_id,
                opponent=0,
                commands=root_id - 1,
                microsteps=count,
                failures=0,
                backtracks=0,
                actor=actor,
                learner=1,
                kind=kind,
                winner=winner,
                selected=selected,
                logp=-0.7,
                root_input=root,
                node=node,
            )
            with patch("haojie_training.online.session.frames", return_value=[frame]):
                session._accept(0, b"")
            if kind == 0:
                # 代替推理后的等待位置；选择与概率在下一个公开帧确认。
                session.awaiting[1] = session.requests[0].pop()

        try:
            receive(1, 1, selected=-1)
            receive(2, 1)
            self.assertEqual(len(session.traces.partial[1]), 1)
            receive(3, 2)
            self.assertEqual(session.traces.pending[1]["node"]["root"], (1, 2))
            receive(4, 2)
            self.assertEqual(len(session.traces.partial[1]), 1)
            receive(4, 0, kind=2)
            self.assertEqual(session.metrics["completed"], 0)
            self.assertEqual(session.traces.pending[1]["discount"], 1)
            receive(5, 1, selected=-1)
            self.assertEqual(len(session.traces.partial[1]), 2)
            receive(6, 2)
            receive(6, 0, kind=1, winner=2)
            terminal = session.traces.ready[0]
            self.assertEqual(
                [s["node"]["root"] for s in terminal["steps"]], [(1, 1), (1, 2), (1, 5)]
            )
            self.assertEqual([s["discount"] for s in terminal["steps"]], [1, 1, 0])
            self.assertEqual(terminal["steps"][-1]["reward"], -1)
            self.assertIsNone(terminal["bootstrap"])
            self.assertEqual(session.metrics["losses"], 1)
            self.assertFalse(session.games)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
