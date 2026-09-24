import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { inspectTeacherFiles } from '../../scripts/training/inspect-teacher';

const run = promisify(execFile);

test('教师种子对按空闲槽动态分发，换边轨迹完整且不重复', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'haojie-dispatch-'));
  const output = join(folder, 'run');
  try {
    await run(
      process.execPath,
      [
        'scripts/training/dispatch/self-play.mjs',
        '--games',
        '6',
        '--seed',
        '12345',
        '--workers',
        '2',
        '--difficulty',
        'easy',
        '--nodes',
        '40',
        '--opponent',
        'medium',
        '--opponent-nodes',
        '40',
        '--commands',
        '1',
        '--plies',
        '1',
        '--output-dir',
        output,
      ],
      { timeout: 60_000, windowsHide: true },
    );
    const summary = JSON.parse(await readFile(join(output, 'summary.json'), 'utf8'));
    assert.deepEqual(
      summary.jobs.map((job: any) => job.seed),
      [12345, 12346, 12347],
    );
    assert.equal(summary.maxActive, 2);
    assert.ok(summary.jobs.every((job: any) => job.completedAt && !job.error));
    assert.ok(
      summary.jobs[2].startedAt >=
        [summary.jobs[0].completedAt, summary.jobs[1].completedAt].sort()[0],
    );
    const inspected = await inspectTeacherFiles(summary.jobs.map((job: any) => job.record));
    assert.equal(inspected.counts.games, 6);
    assert.equal(inspected.counts.commands, 6);
    assert.equal(inspected.counts.truncated, 6);
    const audit = join(folder, 'parallel-inspection.json');
    await run(
      process.execPath,
      [
        'scripts/training/dispatch/inspect.mjs',
        ...summary.jobs.map((job: any) => job.record),
        '--encode',
        '--workers',
        '2',
        '--output',
        audit,
      ],
      { timeout: 60_000, windowsHide: true },
    );
    const parallel = JSON.parse(await readFile(audit, 'utf8'));
    assert.equal(parallel.maxActive, 2);
    assert.deepEqual(parallel.counts, inspected.counts);
    assert.deepEqual(parallel.pairScores, inspected.pairScores);
    await assert.rejects(
      run(
        process.execPath,
        [
          'scripts/training/dispatch/inspect.mjs',
          summary.jobs[0].record,
          summary.jobs[0].record,
          '--workers',
          '2',
          '--output',
          join(folder, 'duplicate-inspection.json'),
        ],
        { timeout: 60_000, windowsHide: true },
      ),
      /跨文件重复对局/,
    );
  } finally {
    await rm(folder, { recursive: true });
  }
});

test('子进程失败后停止领取新种子，保留已尝试的报告', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'haojie-dispatch-failure-'));
  const output = join(folder, 'run');
  try {
    await assert.rejects(
      run(
        process.execPath,
        [
          'scripts/training/dispatch/self-play.mjs',
          '--games',
          '6',
          '--seed',
          '12400',
          '--workers',
          '2',
          '--difficulty',
          'easy',
          '--nodes',
          '1',
          '--opponent',
          'medium',
          '--opponent-nodes',
          '1',
          '--output-dir',
          output,
        ],
        { timeout: 60_000, windowsHide: true },
      ),
    );
    const summary = JSON.parse(await readFile(join(output, 'summary.json'), 'utf8'));
    assert.equal(summary.jobs.length, 2);
    assert.ok(summary.jobs.every((job: any) => job.error));
    assert.equal(
      summary.jobs.some((job: any) => job.seed === 12402),
      false,
    );
  } finally {
    await rm(folder, { recursive: true });
  }
});
