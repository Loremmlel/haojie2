// 基准通过 esbuild 冻结整个依赖图，避免旧实现间接导入正在修改的引擎。
export { sampleCommand, emptyMetrics } from '../economics/sample';
export { TinyPolicy, randomStream } from '../economics/policy';
export { readTrainingRecords } from '../records/replay';
export { TrainingActionTree } from '../../../src/ai/training/action-tree';
export { encodeDecision } from '../../../src/ai/training/encoding/decision';
export { createGame, applyCommand, inspectCommand } from '../../../src/engine/commands/game';
export { applyPlayerCommand } from '../../../src/engine/online/authority';
export { readRecordLines } from '../records/io';
