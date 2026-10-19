// 冻结长局验证实际使用的 TS 参照，避免运行时混入另一个版本。
export { sampleWorker } from '../../sampling/worker';
export { TrainingEnvironment } from '../../../../src/match/training';
export { createGame } from '../../../../src/engine/commands/game';
export { applyPlayerCommand } from '../../../../src/engine/online/authority';
export { observe, decisionOwner } from '../../../../src/ai/observation';
