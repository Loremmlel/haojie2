import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs, isDeepStrictEqual } from 'node:util';
import { readTrainingRecords } from '../records/replay';
import { encodingSourceHash } from '../encode';
import { PythonPolicy } from '../python-policy';
import { beamDecode } from '../improvement/beam';
import { canonicalTrainingCommand, TrainingActionTree } from '../../../src/ai/training/action-tree';
import { encodeDecision } from '../../../src/ai/training/encoding/decision';
import { decisionOwner, fingerprint, hash, imagined } from '../../../src/ai/observation';
import { decide } from '../../../src/ai/planning/search';
import { allocateBudget, emptyBudget } from '../../../src/ai/budget';
import { sampleTrainingTransition } from '../../../src/ai/training/simulation';
import type { Observation } from '../../../src/ai/types';
import type { Command } from '../../../src/engine/types';
import { rollout, pairedRanking, type RolloutJob } from './rollout';

const digest = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');
const read = (path: string) => JSON.parse(readFileSync(path, 'utf8'));
const write = (path: string, value: unknown) =>
  writeFileSync(path, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
const origin = 'artifacts/training/improvement-20260926';

/** 每个工作进程仅接收公开局面；墙钟保护只产生未知，不能把半成品送回排名。 */
async function worker(
  job: RolloutJob,
): Promise<ReturnType<typeof rollout> | { stop: string; complete: false; elapsedMs: number }> {
  return new Promise((resolveJob, reject) => {
    const child = spawn(
      process.execPath,
      ['--import', 'tsx', fileURLToPath(import.meta.url), '--worker'],
      { stdio: ['ignore', 'ignore', 'inherit', 'ipc'], windowsHide: true },
    );
    let settled = false;
    const timer = setTimeout(() => {
      settled = true;
      child.kill();
      resolveJob({ stop: 'wall-limit', complete: false, elapsedMs: 120000 });
    }, 120000);
    child.once('error', reject);
    child.once('exit', (code) => {
      clearTimeout(timer);
      if (!settled) reject(new Error(`续弈子进程异常退出 ${code}`));
    });
    child.once('message', (message: any) => {
      settled = true;
      clearTimeout(timer);
      child.kill();
      if (message.error) reject(new Error(message.error));
      else resolveJob(message.result);
    });
    child.send(job);
  });
}

if (process.argv.includes('--worker')) {
  process.on('disconnect', () => process.exit(0));
  process.once('message', (job: RolloutJob) => {
    try {
      process.send!({ result: rollout(job) });
    } catch (error) {
      process.send!({ error: String(error) });
    }
  });
} else {
  const { values } = parseArgs({
    options: {
      output: { type: 'string' },
      stage: { type: 'string', default: 'freeze' },
    },
  });
  assert.ok(values.output);
  assert.ok(['freeze', 'pilot', 'development', 'confirm'].includes(values.stage!));
  const output = resolve(values.output);
  mkdirSync(output, { recursive: true });
  const input = `${origin}/final-selected-greedy/games.jsonl.gz`;
  const checkpoint = `${origin}/value-selected/model.pt`;
  const certificate = `${origin}/value-selected/certificate.json`;
  const sourceFiles = [
    'scripts/training/counterfactual/run.ts',
    'scripts/training/counterfactual/rollout.ts',
    'scripts/training/improvement/beam.ts',
    'scripts/training/python-policy.ts',
    'scripts/training/search/puct.ts',
    'scripts/training/improvement/model.py',
    'scripts/training/improvement/serve.py',
    'scripts/training/search/recovery/spatial.py',
    'training/haojie_training/model.py',
    'training/haojie_training/inference.py',
    'training/haojie_training/data.py',
    'training/haojie_training/runtime.py',
  ];
  const hashes = Object.fromEntries(sourceFiles.map((path) => [path, digest(path)]));
  const sourceHash = encodingSourceHash();
  const inputHash = digest(input);
  const buckets = new Map<string, any[]>();
  const observations = new Map<string, Observation>();
  const coverage: Record<string, number> = {};
  let header: any,
    replayed = 0,
    games = 0;
  // 先完整校验输入；分层只依赖原命令类型，哈希抽样不看未来胜负、模型分歧或完成难度。
  for await (const row of readTrainingRecords(input)) {
    if (row.type === 'game') {
      header = row;
      games++;
    }
    if (row.type !== 'decision') continue;
    replayed++;
    if (row.actor !== header.networkPlayer) continue;
    const phase = row.observation.pending.length ? 'reaction' : row.observation.phase;
    coverage[phase] = (coverage[phase] ?? 0) + 1;
    if (phase !== 'play') continue;
    const stratum = row.command.type === 'deploy' ? 'deploy' : 'action';
    const rank = createHash('sha256')
      .update(`breakthrough-v1:${row.before}:${row.actor}`)
      .digest('hex');
    const p = {
      seed: header.seed,
      game: row.game,
      index: row.index,
      before: row.before,
      actor: row.actor,
      stratum,
      rank,
    };
    const key = `${header.seed}:${stratum}`;
    const bucket = buckets.get(key) ?? [];
    if (!bucket.some((p) => p.before === row.before)) bucket.push(p);
    bucket.sort((a, b) => a.rank.localeCompare(b.rank));
    bucket.length = Math.min(bucket.length, 2);
    buckets.set(key, bucket);
    if (bucket.includes(p)) observations.set(p.before, row.observation);
  }
  const families = [...new Set([...buckets.values()].flat().map((p) => p.seed))].sort();
  assert.equal(families.length, 8);
  const positions: any[] = [];
  for (let slot = 0; slot < 4; slot++)
    for (const [i, seed] of families.entries()) {
      const stratum = (i + slot) % 2 === 0 ? 'deploy' : 'action';
      const bucket = buckets.get(`${seed}:${stratum}`)!;
      assert.equal(bucket.length, 2);
      positions.push(bucket[Math.floor(slot / 2)]);
    }
  const identity = {
    input,
    inputHash,
    checkpoint,
    checkpointHash: digest(checkpoint),
    certificateHash: digest(certificate),
    sourceHash,
    hashes,
    positions,
  };
  const protocolPath = resolve(output, 'protocol.json');
  if (values.stage === 'freeze') {
    write(protocolPath, {
      format: 'haojie-counterfactual-v2',
      ...identity,
      games,
      replayed,
      coverage,
      pilot: positions.slice(0, 8).map((p) => p.before),
      continuation: {
        difficulty: 'easy',
        nodes: 40,
        cache: 'shared TrainingTeacher, cold per branch',
      },
      maxCommands: 200,
      maxWork: 8000,
      workAccounting:
        'teacher request 40; count actual atomic enumeration attempts; crossing branch limit returns unknown and records unexecuted work',
      wallMs: 120000,
      workers: 4,
      pilotScenarios: 2,
      developmentScenarios: 4,
      candidateRule:
        'beam first two; append cold hard800 and easy40, deduplicate; fill to four from beam then hard trace; retain beam baseline; original domain is retained neural subset, full eight logged',
      boundary:
        'root ply + 2, original active side, summon/synthesis, no pending; engine terminal also complete',
      costGate:
        'at least 7/8 roots with >=1 scenario completed by all four-or-fewer candidates; no tactical regression',
      confirmRule:
        'only after cost gate; frozen first 8 roots, all candidates, 2 new independent scenarios, 120 plies / 1800 commands / 72000 work; terminal labels only',
      stopRule:
        'cost gate failure stops expansion; heuristic-only differences cannot authorize training or whole-game promotion',
      randomRule:
        'hash of stage/root ordinal/scenario, never official seed/rng; each transition gets its own simulation source',
      scope:
        'development student states from previously evaluated 8 families; not blind evaluation',
    });
    console.log(JSON.stringify({ frozen: positions.length, families, games, replayed, coverage }));
  } else {
    const protocol = read(protocolPath);
    for (const [key, value] of Object.entries(identity))
      assert.deepEqual(protocol[key], value, `冻结身份改变: ${key}`);
    if (values.stage !== 'pilot')
      assert.equal(read(resolve(output, 'pilot/summary.json')).costPassed, true);
    const dir = resolve(output, values.stage!);
    mkdirSync(dir, { recursive: true });
    const count = values.stage === 'development' ? 32 : 8;
    const scenarios = values.stage === 'development' ? 4 : 2;
    const cert = read(certificate);
    assert.equal(cert.passed, true);
    assert.equal(cert.checkpoint_sha256, digest(checkpoint));
    const policy = await PythonPolicy.start({
      python: 'training/.venv/Scripts/python.exe',
      checkpoint,
      module: 'scripts.training.improvement.serve',
      device: 'cpu',
      precision: 'fp32',
      threads: 1,
      timeoutMs: 60000,
    });
    const candidateRows: any[] = [];
    const results: any[] = [];
    const boundaries = new Map<string, Observation>();
    const started = performance.now();
    try {
      for (const [index, p] of positions.slice(0, count).entries()) {
        const o = observations.get(p.before)!;
        const beam = await beamDecode(o, p.actor, policy.evaluate);
        assert.ok(beam.command && beam.candidates.length, `束暂停 ${p.before}`);
        const easy = decide(o, p.actor, 'easy', { simulations: 40, mode: 'work' });
        const hard = decide(o, p.actor, 'hard', { simulations: 800, mode: 'work', trace: true });
        const production = decide(o, p.actor, 'hard', {
          ...allocateBudget(imagined(o), 'hard', emptyBudget()),
          trace: false,
          mode: 'work',
        });
        assert.ok(easy.command && hard.command && production.command);
        const candidates: Command[] = [];
        const offer = (command: Command) => {
          const c = canonicalTrainingCommand(o, p.actor, command);
          if (candidates.length < 4 && !candidates.some((old) => isDeepStrictEqual(old, c)))
            candidates.push(c);
        };
        beam.candidates.slice(0, 2).forEach((c) => offer(c.command));
        offer(hard.command);
        offer(easy.command);
        beam.candidates.forEach((c) => offer(c.command));
        hard.trace?.alternatives.forEach((c) => offer(c.command));
        const inDomain = (command: Command) =>
          candidates.findIndex((c) =>
            isDeepStrictEqual(c, canonicalTrainingCommand(o, p.actor, command)),
          );
        candidateRows.push({
          index,
          position: p,
          beam,
          easy,
          hard,
          production,
          candidates,
          originalDomain: candidates
            .map((_, i) => i)
            .filter((i) =>
              beam.candidates.some((c) =>
                isDeepStrictEqual(candidates[i], canonicalTrainingCommand(o, p.actor, c.command)),
              ),
            ),
          coldIndex: inDomain(hard.command),
          easyIndex: inDomain(easy.command),
          productionIndex: inDomain(production.command),
          prunedBeam: beam.candidates.filter((c) => inDomain(c.command) < 0).length,
        });
      }
      write(resolve(dir, 'candidates.json'), candidateRows);
      const jobs = candidateRows.flatMap((r) =>
        r.candidates.flatMap((command: Command, candidate: number) =>
          Array.from({ length: scenarios }, (_, scene) => ({
            index: r.index,
            candidate,
            scene,
            job: {
              observation: observations.get(r.position.before)!,
              command,
              sample: hash(`counterfactual-${values.stage}:${r.index}:${scene}`),
              maxCommands: values.stage === 'confirm' ? 1800 : protocol.maxCommands,
              maxWork: values.stage === 'confirm' ? 72000 : protocol.maxWork,
              terminal: values.stage === 'confirm',
              maxPlies: 120,
            },
          })),
        ),
      );
      let cursor = 0;
      await Promise.all(
        Array.from({ length: 4 }, async () => {
          while (cursor < jobs.length) {
            const task = jobs[cursor++];
            const result = await worker(task.job);
            const { job, ...id } = task;
            const { observation: end, ...record } = result as any;
            // 独立重放每条返回命令，检查实际边界与标签对应；不读取trace的outcomes冒充后继。
            if (end) {
              let o = job.observation;
              for (const [i, step] of record.steps.entries()) {
                assert.equal(step.before, fingerprint(o));
                assert.equal(step.actor, decisionOwner(o));
                assert.equal(step.sample, hash(`counterfactual-transition:${job.sample}:${i}`));
                o = sampleTrainingTransition(o, step.actor, step.command, step.sample);
                assert.equal(step.after, fingerprint(o));
              }
              assert.deepEqual(o, end);
              assert.equal(record.final, fingerprint(end));
              boundaries.set(`${id.index}:${id.candidate}:${id.scene}`, end);
              if (record.stop === 'boundary') {
                assert.equal(end.ply, job.observation.ply + 2);
                assert.equal(end.active, job.observation.active);
                assert.equal(end.pending.length, 0);
                assert.ok(['summon', 'synthesis'].includes(end.phase));
              }
            }
            results.push({ ...id, sample: job.sample, ...record });
            write(
              resolve(dir, `part-${task.index}-${task.candidate}-${task.scene}.json`),
              results.at(-1),
            );
            console.log(
              JSON.stringify({
                ...id,
                stop: record.stop,
                commands: record.steps?.length,
                work: record.work,
              }),
            );
          }
        }),
      );
      results.sort((a, b) => a.index - b.index || a.candidate - b.candidate || a.scene - b.scene);
      // 价值排序使用真正完成的同一边界；证书未覆盖的阶段仍是未知，不能补零。
      for (const row of results) {
        row.networkValue = null;
        if (!row.complete) continue;
        if (row.value !== null) {
          row.networkValue = row.value;
          continue;
        }
        const o = boundaries.get(`${row.index}:${row.candidate}:${row.scene}`)!;
        if (!cert.covered_phases.includes(o.phase)) continue;
        const actor = decisionOwner(o);
        const node = new TrainingActionTree(o, actor).node();
        const prediction = await policy.evaluate(encodeDecision(o, actor, node));
        row.networkValue = prediction.value * (actor === positions[row.index].actor ? 1 : -1);
      }
      const ranks = candidateRows.map((r) => {
        const matrix = r.candidates.map((_: unknown, candidate: number) =>
          results.filter((v) => v.index === r.index && v.candidate === candidate),
        );
        const ranking = pairedRanking(matrix);
        return {
          index: r.index,
          position: r.position,
          ...ranking,
          original: pairedRanking(r.originalDomain.map((i: number) => matrix[i])),
          network: pairedRanking(
            matrix.map((rows: any[]) =>
              rows.map((v) => ({
                complete: v.complete && v.networkValue !== null,
                heuristic: v.networkValue,
              })),
            ),
          ),
          terminalPaths: matrix.flat().filter((v: any) => v.stop === 'terminal').length,
        };
      });
      write(resolve(dir, 'results.json'), results);
      write(resolve(dir, 'rankings.json'), ranks);
      const completeRoots = ranks.filter((r) => r.paired.length > 0).length;
      const summary = {
        stage: values.stage,
        protocolHash: digest(protocolPath),
        candidateHash: digest(resolve(dir, 'candidates.json')),
        resultsHash: digest(resolve(dir, 'results.json')),
        roots: count,
        scenarios,
        paths: results.length,
        completeRoots,
        costPassed: values.stage === 'pilot' ? completeRoots >= 7 : null,
        stops: Object.fromEntries(
          [...new Set(results.map((r) => r.stop))].map((s) => [
            s,
            results.filter((r) => r.stop === s).length,
          ]),
        ),
        commands: results.reduce((n, r) => n + (r.steps?.length ?? 0), 0),
        work: results.reduce((n, r) => n + (r.work ?? 0), 0),
        heuristicChanges: ranks.filter((r) => !r.fallback && r.best !== 0).length,
        elapsedMs: performance.now() - started,
        inference: policy.totals,
        noTraining: true,
        strengthClaim: false,
      };
      write(resolve(dir, 'summary.json'), summary);
      console.log(JSON.stringify(summary));
    } finally {
      policy.close();
    }
    assert.equal(encodingSourceHash(), sourceHash);
    assert.equal(digest(input), inputHash);
    for (const [path, sha] of Object.entries(hashes)) assert.equal(digest(path), sha);
  }
}
