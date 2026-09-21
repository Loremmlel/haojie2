import {
  applyPlayerCommand,
  canRebasePlayerCommand,
  getPlayerView,
  parseCommand,
  type GameState,
  type Player,
} from '../../src/engine';
import type { CommandReceipt, OnlineUpdate } from '../../src/ui/online/types';

export interface DemoRequest {
  requestId: string;
  baseRevision: number;
  command: unknown;
}
/** 可运行的教学宿主，不提供持久化、认证或生产房间服务。
 * 正式环境从已认证连接取得操作者身份，且该对象必须仅存在于服务器。 */
export class DemoRoom {
  private state: GameState;
  private receipts = new Map<string, { input: string; receipt: CommandReceipt }>();
  revision = 0;
  constructor(
    readonly matchId: string,
    initialState: GameState,
  ) {
    this.state = structuredClone(initialState);
  }
  /** 仅测试和调试可读取权威局面；真实宿主不能将此快照发送至浏览器。 */
  inspect(): GameState {
    return structuredClone(this.state);
  }
  update(viewer: Player, kind: OnlineUpdate['kind'] = 'update'): OnlineUpdate {
    return JSON.parse(
      JSON.stringify({
        matchId: this.matchId,
        revision: this.revision,
        kind,
        view: getPlayerView(this.state, viewer),
      }),
    ) as OnlineUpdate;
  }
  submit(actor: Player, request: DemoRequest): CommandReceipt {
    if (
      !request.requestId ||
      request.requestId.length > 128 ||
      !Number.isSafeInteger(request.baseRevision) ||
      request.baseRevision < 0
    )
      return { ok: false, message: '请求编号或修订号不合法。' };
    const key = `${actor}:${request.requestId}`;
    const input = JSON.stringify({ baseRevision: request.baseRevision, command: request.command });
    const cached = this.receipts.get(key);
    if (cached)
      return cached.input === input
        ? structuredClone(cached.receipt)
        : { ok: false, message: '同一请求编号不能代表不同命令。' };
    let receipt: CommandReceipt;
    try {
      const command = parseCommand(request.command);
      if (
        request.baseRevision > this.revision ||
        (request.baseRevision !== this.revision &&
          !canRebasePlayerCommand(this.state, actor, command))
      )
        receipt = { ok: false, message: '局面已更新，请同步后重新选择操作。' };
      else {
        const next = applyPlayerCommand(this.state, actor, command);
        // 真实宿主必须在回执前将新局面、修订号和请求记录原子持久化。
        this.state = next;
        this.revision++;
        receipt = { ok: true, revision: this.revision };
      }
    } catch (e) {
      receipt = { ok: false, message: e instanceof Error ? e.message : '命令未被接受。' };
    }
    this.receipts.set(key, { input, receipt });
    return structuredClone(receipt);
  }
}
