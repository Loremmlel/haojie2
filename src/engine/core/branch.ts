import type { GamePosition } from '../types';
import { isRuntimePosition, importRuntimePosition, forkRuntimePosition } from '../runtime/position';

/** 外部对象只在进入查询上下文时规范化一次；运行器之后只建立类型化分支。 */
export function createPositionFork<S extends GamePosition>(position: S): () => S {
  const runtime = isRuntimePosition(position) ? position : importRuntimePosition(position);
  return () => forkRuntimePosition(runtime);
}

/** 紧凑分支已经是规则数据，不需要代理图收口。 */
export function commitPosition<S extends GamePosition>(position: S): S {
  return position;
}

/** 正式入口保留外部输入独立性；拥有型运行器复用布局，冷历史按需隔离。 */
export function forkPosition<S extends GamePosition>(position: S, _willCommit = false): S {
  return isRuntimePosition(position)
    ? forkRuntimePosition(position)
    : importRuntimePosition(position);
}
