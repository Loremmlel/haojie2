import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Board } from '../../src/ui/board/Board';
import { useEffectPlayback } from '../../src/ui/board/vfx/useEffectPlayback';
import { vfxScenarios } from './scenarios';
import '../../src/ui/styles/index.css';
import './preview.css';
const scenarios = vfxScenarios();
function Preview() {
  const [index, setIndex] = useState(0);
  const [after, setAfter] = useState(false);
  const playback = useEffectPlayback();
  const scenario = scenarios[index];
  function select(i: number) {
    playback.clear();
    setAfter(false);
    setIndex(i);
  }
  function play() {
    setAfter(true);
    playback.play(scenario.after.events);
  }
  return (
    <div className="hj-game vfx-preview">
      <header>
        <b>浩劫 · 特效工坊</b>
        <span>开发预览 / 不打包进正式游戏</span>
      </header>
      <main>
        <aside>
          <h2>动作目录</h2>
          <div className="vfx-preview-menu">
            {scenarios.map((s, i) => (
              <button key={s.name} aria-pressed={i === index} onClick={() => select(i)}>
                {s.name}
              </button>
            ))}
          </div>
          <p>{scenario.detail}</p>
          <div className="vfx-preview-actions">
            <button className="primary" onClick={play}>
              播放动作
            </button>
            <button
              onClick={() => {
                playback.clear();
                setAfter(false);
              }}
            >
              复位 / 取消
            </button>
          </div>
          <p>
            快速连续点击测试并存上限。系统“减少动态效果”保留静态范围与结果。该面板用真实引擎命令生成事件，不改变棋局规则。
          </p>
        </aside>
        <div className="vfx-preview-board">
          <Board
            state={after ? scenario.after : scenario.before}
            intent={{ kind: 'none' }}
            selectedId={null}
            onCell={() => {}}
            effects={playback.batches}
          />
        </div>
      </main>
    </div>
  );
}
createRoot(document.getElementById('root')!).render(<Preview />);
