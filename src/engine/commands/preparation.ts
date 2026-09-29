import { RuleError } from '../core/state';

/** 仅限同一不可变局面的只读准备，成功值和规则拒绝都只计算一次；其他异常不缓存。 */
export function memoizePreparation<K, V>(prepare: (key: K) => V): (key: K) => V {
  const cache = new Map<K, { value: V } | { error: RuleError }>();
  return (key) => {
    let result = cache.get(key);
    if (!result) {
      try {
        result = { value: prepare(key) };
      } catch (error) {
        if (!(error instanceof RuleError)) throw error;
        result = { error };
      }
      cache.set(key, result);
    }
    if ('error' in result) throw result.error;
    return result.value;
  };
}
