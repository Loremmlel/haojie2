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
    const previous = seen.get(value);
    if (previous !== undefined) return previous;
    const array = Array.isArray(value);
    if (!array) {
      const prototype = Object.getPrototypeOf(value);
      if (prototype !== Object.prototype && prototype !== null) throw unsupported;
    }
    const result = array ? value.slice() : { ...value };
    seen.set(value, result);
    // 浅复制已保留所有标量；只递归子对象，避免再次写入棋子的每个标量字段。
    const child = (key: string | number) => {
      const item = result[key];
      if (item !== null && typeof item === 'object') result[key] = copy(item);
      else if (typeof item === 'function' || typeof item === 'symbol') throw unsupported;
    };
    if (array) {
      for (let i = 0; i < result.length; i++) child(i);
    } else for (const key of Object.keys(result)) child(key);
    return result;
  };
  try {
    return copy(position);
  } catch (error) {
    if (error !== unsupported) throw error;
    return structuredClone(position);
  }
}
