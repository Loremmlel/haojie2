import type { EncodedDecision } from '../../../src/ai/training/encoding/decision';

const scalar = new DataView(new ArrayBuffer(4));
function bf16(value: number) {
  scalar.setFloat32(0, value, true);
  const bits = scalar.getUint32(0, true);
  return (bits & 0x7fffffff) > 0x7f800000
    ? (bits >>> 16) | 0x40
    : (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
}
function decoded(bits: number) {
  scalar.setUint32(0, bits << 16, true);
  return scalar.getFloat32(0, true);
}

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
  readonly weights: Uint16Array[] = [];
  private readonly embedding: Uint16Array;
  private readonly entity: Uint16Array;
  private readonly global: Uint16Array;
  private readonly action: Uint16Array;
  private readonly output: Uint16Array;

  constructor(readonly seed: number) {
    const random = randomStream(seed);
    const matrix = (inputs: number, outputs: number) => {
      const weights = Uint16Array.from({ length: (inputs + 1) * outputs }, () =>
        bf16((random() * 2 - 1) / Math.sqrt(inputs)),
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

  private layer(input: ArrayLike<number>, weights: Uint16Array, outputs: number) {
    const result = new Uint16Array(outputs);
    for (let j = 0; j < outputs; j++) {
      const offset = j * (input.length + 1);
      let value = decoded(weights[offset + input.length]);
      for (let i = 0; i < input.length; i++)
        value += decoded(bf16(input[i])) * decoded(weights[offset + i]);
      result[j] = bf16(Math.tanh(value));
    }
    return result;
  }

  logits(input: EncodedDecision): number[] {
    return this.evaluate(input);
  }

  /** 决策内固定行与权重只读；仅复用逐实体 MLP，不缓存含前缀/掩码的池化结果。 */
  decision() {
    const cache = new WeakMap<number[], { kind: number; hidden: Uint16Array }>();
    return (input: EncodedDecision) => this.evaluate(input, cache);
  }

  private evaluate(
    input: EncodedDecision,
    cache?: WeakMap<number[], { kind: number; hidden: Uint16Array }>,
  ): number[] {
    const entityInput = Array<number>(80).fill(0);
    const entities = input.entities.map((row, i) => {
      const kind = input.kinds[i],
        known = cache?.get(row);
      if (known && known.kind === kind) return known.hidden;
      for (let j = 0; j < 64; j++) entityInput[j] = row[j];
      for (let j = 0; j < 16; j++) entityInput[64 + j] = decoded(this.embedding[kind * 16 + j]);
      const hidden = this.layer(entityInput, this.entity, this.width);
      cache?.set(row, { kind, hidden });
      return hidden;
    });
    const context = this.layer(input.globals, this.global, this.width);
    const count = input.entity_mask.filter(Boolean).length;
    for (let i = 0; i < entities.length; i++)
      if (input.entity_mask[i])
        for (let j = 0; j < this.width; j++)
          context[j] = bf16(decoded(context[j]) + decoded(entities[i][j]) / Math.max(1, count));
    const empty = new Uint16Array(this.width);
    const actionInput = Array<number>(64 + 3 * this.width).fill(0);
    for (let j = 0; j < this.width; j++) actionInput[64 + j] = decoded(context[j]);
    return input.candidates.map((row, i) => {
      for (let j = 0; j < 64; j++) actionInput[j] = row[j];
      const source = entities[input.sources[i]] ?? empty,
        target = entities[input.targets[i]] ?? empty;
      for (let j = 0; j < this.width; j++) {
        actionInput[64 + this.width + j] = decoded(source[j]);
        actionInput[64 + 2 * this.width + j] = decoded(target[j]);
      }
      const hidden = this.layer(actionInput, this.action, this.width);
      let value = decoded(this.output[this.width]);
      for (let j = 0; j < this.width; j++) value += decoded(hidden[j]) * decoded(this.output[j]);
      return decoded(bf16(value));
    });
  }
}
