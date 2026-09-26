import type { GamePosition } from '../types';

const unsupported = Symbol('non-data position');

/**
 * 复制引擎已构造的纯数据局面；不删字段，不共享可变子对象，不修改输入。
 * 普通对象与数组走轻量路径，保留 undefined、稀疏数组及重复引用/环的关系。
 * 若扩展字段出现非普通对象，整份回退 structuredClone，避免跨类型引用被拆散。
 * 这不是存档校验器；不接受带访问器或自定义数组属性的非数据模型。
 */
export function clonePosition<S extends GamePosition>(position: S): S {
  const seen = new Map<object, object>();
  const copy = (value: any): any => {
    if (value === null || typeof value !== 'object') {
      if (typeof value === 'function' || typeof value === 'symbol') throw unsupported;
      return value;
    }
    if (seen.has(value)) return seen.get(value);
    const array = Array.isArray(value);
    const prototype = Object.getPrototypeOf(value);
    if (!array && prototype !== Object.prototype && prototype !== null) throw unsupported;
    const result = array ? value.slice() : { ...value };
    seen.set(value, result);
    for (const key of Object.keys(result)) result[key] = copy(result[key]);
    return result;
  };
  try {
    return copy(position);
  } catch (error) {
    if (error !== unsupported) throw error;
    return structuredClone(position);
  }
}
