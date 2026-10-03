import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { nativeHash } from '../../scripts/training/native/pipeline/hash';

test('原生记录规范固定字节向量覆盖负零、缺失值和UTF-8键顺序', () => {
  const vectors = JSON.parse(
    readFileSync(
      new URL('../../native/engine-prototype/data/hash-vectors.json', import.meta.url),
      'utf8',
    ),
  );
  for (const vector of vectors) {
    assert.equal(
      nativeHash(vector.value),
      createHash('sha256').update(Buffer.from(vector.bytes, 'hex')).digest('hex'),
    );
  }
  assert.equal(
    nativeHash(null),
    '6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d',
  );
  assert.notEqual(nativeHash({}), nativeHash({ a: null }));
  assert.equal(nativeHash({}), nativeHash({ a: undefined }));
});
