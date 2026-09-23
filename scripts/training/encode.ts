import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { createReadStream, readFileSync, readdirSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import { encodeDecision } from '../../src/ai/training/encoding/decision';
import { ENCODING_SCHEMA } from '../../src/ai/training/encoding/schema';
import { HAOJIE_RULESET } from '../../src/engine/online/player-view';
import { ensure } from '../../src/engine/core/state';

/** 记录实际工作树内容，未提交的编码/规则修改同样改变指纹。 */
export function encodingSourceHash() {
  const root = fileURLToPath(new URL('../../', import.meta.url));
  const hash = createHash('sha256');
  for (const folder of ['src/engine', 'src/ai', 'src/match']) {
    const path = resolve(root, folder);
    for (const name of readdirSync(path, { recursive: true })
      .map(String)
      .filter((p) => p.endsWith('.ts'))
      .sort()) {
      const file = resolve(path, name);
      hash.update(relative(root, file).replaceAll('\\', '/'));
      hash.update(readFileSync(file, 'utf8').replaceAll('\r\n', '\n'));
    }
  }
  hash.update(readFileSync(fileURLToPath(import.meta.url), 'utf8').replaceAll('\r\n', '\n'));
  return hash.digest('hex');
}

/**
 * 将可信教师记录转换为版本化网络输入。游戏种子仅留在整局元数据，绝不传入编码器。
 * 逐行处理并等待消费者背压；格式/动作覆盖失败即报错，不跳过样本。
 * 缺少outcome的尾局显式标记interrupted，后续只可使用策略标签。
 */
export async function* encodeTeacherFile(path: string) {
  yield {
    type: 'encoding',
    format: 'haojie-encoded-jsonl-v1',
    schema: ENCODING_SCHEMA,
    source_sha256: encodingSourceHash(),
    ruleset: HAOJIE_RULESET,
  };
  const lines = createInterface({ input: createReadStream(path, 'utf8'), crlfDelay: Infinity });
  const games = new Set<number>();
  let current: number | undefined,
    count = 0;
  const interrupted = () => ({
    type: 'outcome',
    game: current,
    commands: count,
    terminated: false,
    truncated: false,
    interrupted: true,
    returns: null,
    winner: null,
  });
  for await (const line of lines) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    if (row.type === 'game') {
      if (current !== undefined) yield interrupted();
      ensure(
        Number.isSafeInteger(row.game) && row.game >= 0 && !games.has(row.game),
        '对局编号重复或无效。',
      );
      ensure(
        row.ruleset === HAOJIE_RULESET,
        '教师记录规则版本不匹配，不能用当前编码器解释旧规则。',
      );
      ensure(row.rules === 'classic' || row.rules === 'shrine', '未知规则模式。');
      ensure(
        Number.isSafeInteger(row.seed) && row.seed >= 1 && row.seed <= 0xffffffff,
        '对局复现种子无效。',
      );
      games.add(row.game);
      current = row.game;
      count = 0;
      const group = `${row.ruleset}:${row.rules}:${row.seed}`;
      ensure(
        row.gameId === undefined ||
          (typeof row.gameId === 'string' && row.gameId.length > 0 && row.gameId.length <= 512),
        '教师对局标识无效。',
      );
      yield {
        type: 'game',
        game: current,
        group,
        game_id: row.gameId ?? group,
        teachers: row.teachers,
        primary_player: row.primaryPlayer,
        rules: row.rules,
        seed: row.seed,
        difficulty: row.difficulty,
        budget: row.budget,
        source: row.source,
        origin: row.origin,
      };
    } else if (row.type === 'sample') {
      ensure(
        current !== undefined && row.game === current && row.index === count,
        '样本不属于当前对局或序号不连续。',
      );
      ensure(row.actor === 1 || row.actor === 2, '样本所属方无效。');
      const tree = new TrainingActionTree(row.observation, row.actor);
      let trace;
      try {
        trace = tree.trace(row.command);
      } catch (error) {
        throw new Error(`game=${current}, command=${count}: ${JSON.stringify(row.command)}`, {
          cause: error,
        });
      }
      for (const [step, { node, selected }] of trace.entries())
        yield {
          type: 'example',
          game: current,
          index: count,
          step,
          actor: row.actor,
          command: row.command.type,
          stage: node.stage,
          selected,
          input: encodeDecision(row.observation, row.actor, node),
        };
      count++;
    } else if (row.type === 'outcome') {
      ensure(
        current !== undefined && row.game === current && row.commands === count,
        '终局记录与样本数不一致。',
      );
      const wasInterrupted =
        row.interrupted === true ||
        (typeof row.interrupted === 'string' && row.interrupted.length > 0);
      ensure(
        row.interrupted == null || row.interrupted === false || wasInterrupted,
        '中断标记必须为空、布尔值或非空原因。',
      );
      ensure(
        typeof row.terminated === 'boolean' &&
          typeof row.truncated === 'boolean' &&
          Number(row.terminated) + Number(row.truncated) + Number(wasInterrupted) === 1,
        '结束记录必须且只能是真实终局、显式截断或中断之一。',
      );
      if (row.terminated) {
        ensure(row.winner === 1 || row.winner === 2 || row.winner === 'draw', '终局胜负无效。');
        for (const p of [1, 2])
          ensure(
            row.returns?.[p] === (row.winner === 'draw' ? 0 : row.winner === p ? 1 : -1),
            '收益与实际胜负不一致。',
          );
      } else ensure(row.returns === null && row.winner === null, '截断局不能有胜负/价值标签。');
      yield {
        type: 'outcome',
        game: current,
        commands: count,
        terminated: row.terminated,
        truncated: row.truncated,
        interrupted: wasInterrupted,
        interruptionReason: typeof row.interrupted === 'string' ? row.interrupted : undefined,
        returns: row.returns,
        winner: row.winner,
      };
      current = undefined;
    } else throw new Error(`未知教师记录类型：${String(row.type)}`);
  }
  if (current !== undefined) yield interrupted();
  ensure(games.size > 0, '教师文件没有对局。');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  ensure(process.argv.length === 3, '用法：tsx scripts/training/encode.ts <教师JSONL>');
  for await (const row of encodeTeacherFile(process.argv[2]))
    if (!process.stdout.write(JSON.stringify(row) + '\n')) await once(process.stdout, 'drain');
}
