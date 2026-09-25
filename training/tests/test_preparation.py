"""验证整局划分、真实收益所属方、缺失终局遮罩及可变长度填充。"""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from haojie_training.data import collate_examples, load_dataset, select_batch, synthetic_batch
from haojie_training.evaluate import validate_split
from haojie_training.model import ModelConfig
from haojie_training.prepare import prepare


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_padding_and_batch_crop_preserve_local_pointers(self):
        config = ModelConfig.tiny()
        examples = [
            {k: v[0] for k, v in synthetic_batch(config, 1, n, a).items()}
            for n, a in [(7, 11), (3, 4)]
        ]
        data = collate_examples(examples, config)
        selected = select_batch(data, torch.tensor([1]))
        n = int(examples[1]["entity_mask"].sum())
        a = int(examples[1]["candidate_mask"].sum())
        self.assertEqual(selected["entities"].shape[1], n)
        self.assertEqual(selected["candidates"].shape[1], a)
        torch.testing.assert_close(selected["sources"][0], examples[1]["sources"][:a])
        torch.testing.assert_close(selected["entities"][0], examples[1]["entities"][:n])

    def test_whole_game_split_and_outcome_masks(self):
        config = ModelConfig()
        tensors = synthetic_batch(config, 1, 3, 3)
        data = {
            key: value[0].tolist()
            for key, value in tensors.items()
            if key not in {"policy", "value", "value_mask"}
        }
        selected = int(tensors["policy"][0].argmax())
        header = {
            "type": "encoding",
            "ruleset": "test-rules",
            "source_sha256": "test-source",
            "schema": {
                "encoding": "test-v1",
                **{
                    key: getattr(config, key)
                    for key in [
                        "entity_features",
                        "global_features",
                        "action_features",
                        "kind_count",
                    ]
                },
            },
        }
        rows = [header]
        for game in range(4):
            rows.append(
                {
                    "type": "game",
                    "game": game,
                    "group": f"game-{game % 3}",
                    "game_id": f"trajectory-{game}",
                    "teachers": {"1": {"difficulty": "hard"}, "2": {"difficulty": "medium"}},
                }
            )
            for index, (actor, step) in enumerate([(1, 0), (1, 1), (2, 0)]):
                rows.append(
                    {
                        "type": "example",
                        "input": data,
                        "selected": selected,
                        "index": index,
                        "step": step,
                        "actor": actor,
                        "command": "move",
                        "stage": "action" if step == 0 else "point",
                    }
                )
            rows.append(
                {
                    "type": "outcome",
                    "game": game,
                    "commands": 2,
                    "terminated": game % 3 == 0,
                    "truncated": game == 1,
                    "interrupted": game == 2,
                    "returns": {"1": 1, "2": -1} if game % 3 == 0 else None,
                }
            )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "teacher.jsonl"
            source.write_text("test input", encoding="utf-8")
            with patch("haojie_training.prepare.encoded_rows", return_value=iter(rows)):
                prepare([source], root / "prepared")
            train, training = load_dataset(root / "prepared/train.pt", config)
            validation, meta = load_dataset(root / "prepared/validation.pt", config)
            validate_split(training, meta)
            self.assertEqual(len(training["groups"]) + len(meta["groups"]), 3)
            self.assertEqual(len(training["game_ids"]) + len(meta["game_ids"]), 4)
            for split in [training, meta]:
                self.assertEqual(
                    "trajectory-0" in split["game_ids"], "trajectory-3" in split["game_ids"]
                )
            for batch, provenance in [(train, training), (validation, meta)]:
                for i, record in enumerate(provenance["records"]):
                    known = record["group"] == "game-0" and record["step"] == 0
                    self.assertEqual(bool(batch["value_mask"][i]), known)
                    if known:
                        self.assertEqual(
                            float(batch["value"][i]), 1 if record["actor"] == 1 else -1
                        )
            report = json.loads((root / "prepared/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(sum(s["value_labels"] for s in report["splits"].values()), 4)
            with self.assertRaises(ValueError):
                validate_split(training, training)
            duplicate = [
                dict(row, game_id="trajectory-0")
                if row["type"] == "game" and row["game"] == 3
                else row
                for row in rows
            ]
            with patch("haojie_training.prepare.encoded_rows", return_value=iter(duplicate)):
                with self.assertRaisesRegex(ValueError, "重复"):
                    prepare([source], root / "duplicate")
            with patch("haojie_training.prepare.encoded_rows", return_value=iter(rows)):
                filtered = prepare([source], root / "hard-only", teacher_difficulty="hard")
            self.assertEqual(sum(g["selected_decisions"] for g in filtered["games"]), 4)
            self.assertEqual(sum(g["excluded_decisions"] for g in filtered["games"]), 4)
            self.assertEqual(sum(s["value_labels"] for s in filtered["splits"].values()), 2)
            for split in ["train", "validation"]:
                _, selected = load_dataset(root / f"hard-only/{split}.pt", config)
                self.assertTrue(all(r["actor"] == 1 for r in selected["records"]))
                self.assertEqual(selected["groups"], filtered["splits"][split]["groups"])

            second = root / "teacher-2.jsonl"
            second.write_text("second input", encoding="utf-8")
            per_file_rows = {source: rows[:11], second: [header, *rows[11:]]}
            barrier = threading.Barrier(2, timeout=5)

            def concurrent_rows(path, _node):
                barrier.wait()
                yield from per_file_rows[path]

            with patch("haojie_training.prepare.encoded_rows", side_effect=concurrent_rows):
                parallel = prepare([source, second], root / "parallel", workers=2)
            self.assertEqual(parallel["splits"], report["splits"])
            self.assertEqual(parallel["games"], report["games"])
            for split in ["train", "validation"]:
                expected, expected_meta = load_dataset(root / f"prepared/{split}.pt", config)
                actual, actual_meta = load_dataset(root / f"parallel/{split}.pt", config)
                self.assertEqual(actual_meta["records"], expected_meta["records"])
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key])

            duplicate_files = {
                source: per_file_rows[source],
                second: [header, dict(rows[11], game_id="trajectory-0"), *rows[12:]],
            }
            with patch(
                "haojie_training.prepare.encoded_rows",
                side_effect=lambda path, _node: iter(duplicate_files[path]),
            ):
                with self.assertRaisesRegex(ValueError, "重复"):
                    prepare([source, second], root / "cross-file-duplicate", workers=2)

            # 固定原分组，不因新切分种子而重选验证局；分片必须保留重复抽样和局内指针。
            from haojie_training.data import sample_indices, select_batch, token_counts

            with patch("haojie_training.prepare.encoded_rows", return_value=iter(rows)):
                sharded = prepare(
                    [source],
                    root / "sharded",
                    split_seed=999,
                    shard_size=3,
                    split_manifest=root / "prepared/manifest.json",
                )
            for split in ("train", "validation"):
                expected, expected_meta = load_dataset(root / f"prepared/{split}.pt", config)
                actual, actual_meta = load_dataset(root / f"sharded/{split}.pt", config)
                self.assertEqual(actual_meta["records"], expected_meta["records"])
                self.assertEqual(
                    sharded["splits"][split]["groups"], report["splits"][split]["groups"]
                )
                for indices in (torch.arange(len(expected["value"])), torch.tensor([2, 0, 2])):
                    before, after = select_batch(expected, indices), select_batch(actual, indices)
                    for key in before:
                        torch.testing.assert_close(before[key], after[key])
                torch.testing.assert_close(
                    token_counts(actual, "entity_mask"), expected["entity_mask"].sum(1)
                )
                first = sample_indices(actual, 32, torch.Generator().manual_seed(4), 3)
                second_draw = sample_indices(actual, 32, torch.Generator().manual_seed(4), 3)
                torch.testing.assert_close(first, second_draw)
                self.assertTrue(bool(((first >= 0) & (first < len(expected["value"]))).all()))
                bad = torch.load(root / f"sharded/{split}.pt", weights_only=True)
                bad["shards"][0]["sha256"] = "corrupt"
                torch.save(bad, root / "sharded/broken.pt")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    load_dataset(root / "sharded/broken.pt", config)


if __name__ == "__main__":
    unittest.main()
