"""冻结策略交付；正式推理与训练使用同一根编码、解码和原生动作状态机。"""

from dataclasses import asdict
from pathlib import Path

import torch

from ..model import FORMAT, Config, Network, validate_weights
from ..native import packet
from ..session import Session, Settings

POLICY_FORMAT = "haojie-root-policy-v1"


def export(checkpoint, destination):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["format"] != FORMAT:
        raise ValueError("只接受当前连续轨迹检查点")
    payload = dict(
        format=POLICY_FORMAT,
        config=state["config"],
        model=state["model"],
        native_identity=state["native_identity"],
        steps=state["steps"],
    )
    _validate(payload)
    with Path(destination).open("xb") as stream:
        torch.save(payload, stream)


def _validate(payload):
    if payload["format"] != POLICY_FORMAT:
        raise ValueError("冻结策略格式不支持")
    validate_weights(payload["model"])


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate(payload)
    model = Network(Config(**payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def frozen_session(policy, opponent, *, device="cuda", workers=2, seed=1):
    """只执行正式原生对局的冻结会话；不采集学习轨迹、不更新任何权重。"""
    own, own_data = load(policy)
    other, other_data = load(opponent)
    session = Session(
        Settings(concurrency=0, workers=workers, seed=seed), device, own.config, training=False
    )
    for payload in (own_data, other_data):
        for key in ("rulesHash", "ruleset"):
            if payload["native_identity"][key] != session.workers[0].identity[key]:
                session.close()
                raise ValueError("冻结策略与正式规则不一致")
    session.models = {0: own.to(device).eval()}
    session.opponents = {0: other.to(device).eval()}
    for model in (session.models[0], session.opponents[0]):
        model.flatten_history()
    return session


def start_pairs(session, seeds, shrine=False):
    """每个正式种子各走双座，权重整局固定；完整结果由正式终局提供。"""
    if session.training or session.games or any(w.busy for w in session.workers):
        raise RuntimeError("双座评测需要空的冻结会话")
    jobs = [[] for _ in session.workers]
    for seed in seeds:
        if not 0 < seed < 2**32:
            raise ValueError("评测种子必须是非零 uint32")
        for seat in (1, 2):
            game = session.next_game
            session.next_game += 1
            worker = (game - 1) % len(jobs)
            jobs[worker].append((game, seed, seat, 0, shrine))
            session.games[game] = dict(
                worker=worker,
                learner=seat,
                opponent=0,
                commands=0,
                microsteps=0,
                failures=0,
                backtracks=0,
                waiting=False,
                seed=seed,
            )
    for worker, admissions in zip(session.workers, jobs):
        if admissions:
            worker.submit(0, packet(admissions))


def initial_policy(destination, native_identity, seed=1, config=Config()):
    """预先冻结从零初始化的固定对手，供不同训练种子共享。"""
    torch.manual_seed(seed)
    model = Network(config)
    with Path(destination).open("xb") as stream:
        torch.save(
            dict(
                format=POLICY_FORMAT,
                config=asdict(config),
                model=model.state_dict(),
                native_identity=native_identity,
                steps=0,
            ),
            stream,
        )
