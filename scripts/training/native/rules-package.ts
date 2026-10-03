import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import {
  CATALOG,
  COMBAT_RULES,
  RULESET_ID,
  SUMMON_POOL,
  ULTIMATE_POOL,
  SHRINE_POOL,
} from '../../../src/engine/catalog';
import { SYNTHESIS_RECIPES } from '../../../src/engine/setup/synthesis';
import { ENCODING_SCHEMA } from '../../../src/ai/training/encoding/schema';
import { DECISION_STAGES } from '../../../src/ai/training/encoding/decision';

// 仅开发/CI 使用；训练安装与运行只读取随源码提交的确定性字节。
const path = 'native/engine-prototype/data/rules.json';
const content =
  JSON.stringify({
    op: 'init',
    protocol: 'haojie-native-engine-v4',
    ruleset: RULESET_ID,
    catalog: CATALOG,
    combat: COMBAT_RULES,
    summonPools: { normal: SUMMON_POOL, ultimate: ULTIMATE_POOL, shrine: SHRINE_POOL },
    recipes: SYNTHESIS_RECIPES,
    encoding: { ...ENCODING_SCHEMA, decision_stages: DECISION_STAGES },
  }) + '\n';
const manifest =
  JSON.stringify(
    {
      format: 'haojie-rules-package-v1',
      sha256: createHash('sha256').update(content).digest('hex'),
      ruleset: RULESET_ID,
      encoding: ENCODING_SCHEMA.encoding,
    },
    null,
    2,
  ) + '\n';
if (process.argv.includes('--check')) {
  if (
    readFileSync(path, 'utf8') !== content ||
    readFileSync(path.replace('rules.json', 'manifest.json'), 'utf8') !== manifest
  )
    throw new Error('规则包过期，请显式重新生成并提交');
} else {
  mkdirSync('native/engine-prototype/data', { recursive: true });
  writeFileSync(path, content);
  writeFileSync(path.replace('rules.json', 'manifest.json'), manifest);
}
