import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

/** 新记录规范向量；不使用 JSON 字符串顺序作为跨语言指纹。 */
export function nativeHash(value: unknown) {
  const h = createHash('sha256');
  const length = (n: number) => {
    const b = Buffer.alloc(8);
    b.writeBigUInt64LE(BigInt(n));
    h.update(b);
  };
  function add(v: any) {
    if (v === null) {
      h.update(Buffer.from([0]));
      return;
    }
    if (typeof v === 'boolean') {
      h.update(Buffer.from([1, Number(v)]));
      return;
    }
    if (typeof v === 'number') {
      assert.ok(Number.isFinite(v));
      h.update(Buffer.from([2]));
      const b = Buffer.alloc(8);
      b.writeDoubleLE(v === 0 ? 0 : v);
      h.update(b);
      return;
    }
    if (typeof v === 'string') {
      h.update(Buffer.from([3]));
      const b = Buffer.from(v);
      length(b.length);
      h.update(b);
      return;
    }
    if (Array.isArray(v)) {
      h.update(Buffer.from([4]));
      length(v.length);
      v.forEach(add);
      return;
    }
    assert.equal(typeof v, 'object');
    h.update(Buffer.from([5]));
    const keys = Object.keys(v)
      .filter((k) => v[k] !== undefined)
      .sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)));
    length(keys.length);
    for (const k of keys) {
      add(k);
      add(v[k]);
    }
  }
  add(value);
  return h.digest('hex');
}
