import { useState } from 'react';
import type { GameState } from '../../engine';
import type { GameModal } from '../game/types';
import { Codex } from '../library/Codex';
import { Rules } from '../library/Rules';
import { Modal } from '../shared/Modal';
import { Icon } from '../shared/visuals';

export function GameDialogs({
  modal,
  state: s,
  onClose,
  startGame,
}: {
  modal: GameModal;
  state: GameState;
  onClose: () => void;
  startGame: (seedText: string, demo: boolean) => void;
}) {
  const [seedText, setSeedText] = useState('');
  return (
    <>
      {modal === 'codex' && <Codex onClose={() => onClose()} />}
      {modal === 'rules' && <Rules onClose={() => onClose()} />}
      {modal === 'log' && (
        <Modal
          title="战场纪事"
          subtitle="最近180条事件；悔棋时一并恢复。"
          onClose={() => onClose()}
        >
          <ol className="full-log">
            {s.log.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ol>
        </Modal>
      )}
      {modal === 'new' && (
        <Modal
          title="浩劫，再起"
          subtitle="当前局面会被替换。重要对局请先导出存档。"
          onClose={() => onClose()}
        >
          <div className="new-match-art">
            <Icon name="sword" size={44} />
            <span>26普通召唤 · 28终极召唤 · 117格战场</span>
          </div>
          <label className="seed-field">
            对局种子<span>留空随机</span>
            <input
              type="number"
              step="1"
              value={seedText}
              onChange={(e) => setSeedText(e.target.value)}
              placeholder="例如 20260907"
            />
          </label>
          <button className="primary full-width" onClick={() => startGame(seedText, false)}>
            开始正式对局
            <Icon name="arrow" />
          </button>
          <button className="secondary full-width" onClick={() => startGame(seedText, true)}>
            载入演示棋局
          </button>
          <p className="fine-print">
            正式对局空棋盘、300血、0人头。演示局包含终极随从、装备、叠放和可兑换人头。
          </p>
        </Modal>
      )}
    </>
  );
}
