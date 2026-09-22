"""推理协议直接复用训练输入约束，防止JSON隐式类型转换和隐藏状态进入模型。"""

import copy
import unittest

import torch

from haojie_training.data import INPUT_KEYS, synthetic_batch
from haojie_training.inference import decode_input
from haojie_training.model import ModelConfig, PolicyValueNet


class InferenceTests(unittest.TestCase):
    def test_same_inputs_same_outputs_and_strict_json_boundary(self):
        torch.set_num_threads(1)
        config = ModelConfig.tiny()
        source = synthetic_batch(config, size=1, entities=4, actions=3)
        raw = {key: source[key][0].tolist() for key in INPUT_KEYS}
        converted = decode_input(raw, config)
        model = PolicyValueNet(config).eval()
        with torch.inference_mode():
            expected = model(source)
            actual = model(converted)
        for a, b in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        invalid = [dict(raw, seed=123), dict(raw, policy=[1, 0, 0])]
        for key, value in [
            ("sources", [0.5, -1, -1]),
            ("entity_mask", [1, 1, 1, 1]),
            ("kinds", [999, 1, 1, 1]),
            ("candidate_mask", [False, False, False]),
        ]:
            invalid.append(dict(raw, **{key: value}))
        nonfinite = copy.deepcopy(raw)
        nonfinite["entities"][0][0] = float("nan")
        invalid.append(nonfinite)
        for example in invalid:
            with self.assertRaises(ValueError):
                decode_input(example, config)


if __name__ == "__main__":
    unittest.main()
