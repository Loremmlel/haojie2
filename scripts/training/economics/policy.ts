import type { EncodedDecision } from '../../../src/ai/training/encoding/decision';

/** 实验私有随机流；不接收规则种子、正式随机状态或历史，不能预测后续抽牌。 */
export function randomStream(seed: number) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return ((state >>> 0) + 0.5) / 0x100000000;
  };
}

/**
 * 独立冷启动策略：实体 MLP + 平均池化 + 状态/来源/目标/动作 MLP。
 * 仅测真实小网络前向成本，不加载旧权重、教师评分或搜索标签，不宣称已学会策略。
 * 全部公开实体都进入池化，候选不裁剪；输入只读，权重在构造后固定。
 */
export class TinyPolicy {
  readonly width = 32;
  readonly parameters: number;
  readonly weights: Float32Array[] = [];
  private readonly embedding: Float32Array;
  private readonly entity: Float32Array;
  private readonly global: Float32Array;
  private readonly action: Float32Array;
  private readonly output: Float32Array;

  constructor(readonly seed: number) {
    const random = randomStream(seed);
    const matrix = (inputs: number, outputs: number) => {
      const weights = Float32Array.from(
        { length: (inputs + 1) * outputs },
        () => (random() * 2 - 1) / Math.sqrt(inputs),
      );
      this.weights.push(weights);
      return weights;
    };
    this.embedding = matrix(255, 16);
    this.entity = matrix(80, this.width);
    this.global = matrix(32, this.width);
    this.action = matrix(64 + 3 * this.width, this.width);
    this.output = matrix(this.width, 1);
    this.parameters = this.weights.reduce((n, w) => n + w.length, 0);
  }

  private layer(input: ArrayLike<number>, weights: Float32Array, outputs: number) {
    const result = new Float32Array(outputs);
    for (let j = 0; j < outputs; j++) {
      const offset = j * (input.length + 1);
      let value = weights[offset + input.length];
      for (let i = 0; i < input.length; i++) value += input[i] * weights[offset + i];
      result[j] = Math.tanh(value);
    }
    return result;
  }

  logits(input: EncodedDecision): number[] {
    const entities = input.entities.map((row, i) =>
      this.layer(
        [...row, ...this.embedding.subarray(input.kinds[i] * 16, (input.kinds[i] + 1) * 16)],
        this.entity,
        this.width,
      ),
    );
    const context = this.layer(input.globals, this.global, this.width);
    const count = input.entity_mask.filter(Boolean).length;
    for (let i = 0; i < entities.length; i++)
      if (input.entity_mask[i])
        for (let j = 0; j < this.width; j++) context[j] += entities[i][j] / Math.max(1, count);
    const empty = new Float32Array(this.width);
    return input.candidates.map((row, i) => {
      const hidden = this.layer(
        [
          ...row,
          ...context,
          ...(entities[input.sources[i]] ?? empty),
          ...(entities[input.targets[i]] ?? empty),
        ],
        this.action,
        this.width,
      );
      let value = this.output[this.width];
      for (let j = 0; j < this.width; j++) value += hidden[j] * this.output[j];
      return value;
    });
  }
}
