import { ensure } from '../../../../src/engine/core/state';
import type { Player } from '../../../../src/engine/types';
import type { Observation } from '../../../../src/ai/types';
import { decisionOwner } from '../../../../src/ai/observation';
import { TrainingActionTree } from '../../../../src/ai/training/action-tree';
import { encodeDecision } from '../../../../src/ai/training/encoding/decision';
import type { PolicyEvaluator } from '../../../../src/ai/training/decoder';
import { PythonPolicy } from '../../python-policy';

/** 只发八项公开根输入；模型预测属于当前实际操作者，必须转换为搜索根视角。 */
export function valueEvaluator(evaluate: PolicyEvaluator) {
  return async (observation: Observation, root: Player) => {
    const actor = decisionOwner(observation);
    const node = new TrainingActionTree(observation, actor).node();
    const output = await evaluate(encodeDecision(observation, actor, node));
    ensure(Number.isFinite(output.value) && Math.abs(output.value) <= 1, '网络价值输出无效。');
    // 首轮固定缩放，不让未验证的非终局预测与真实±1终局处于相同幅度。
    return 0.25 * output.value * (actor === root ? 1 : -1);
  };
}

export async function openValueModel(checkpoint: string, signal?: AbortSignal) {
  const model = await PythonPolicy.start({
    python: 'training/.venv/Scripts/python.exe',
    checkpoint,
    device: 'cuda',
    precision: 'fp32',
    threads: 1,
    timeoutMs: 120000,
    signal,
  });
  return { model, value: valueEvaluator((input) => model.evaluate(input, signal)) };
}
