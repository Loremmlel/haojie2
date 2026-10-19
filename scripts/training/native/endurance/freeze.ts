import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { freeze } from '../sampling/artifacts';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    executable: { type: 'string' },
  },
});
assert.ok(values.output && values.executable);
await freeze(
  'scripts/training/native/endurance/api.ts',
  resolve(values.output),
  resolve(values.executable),
);
