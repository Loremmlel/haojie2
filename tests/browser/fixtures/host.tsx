import { createRoot } from 'react-dom/client';
import { createDemoGame, HaojieGame } from '../../../src';
import type { GameState } from '../../../src';
declare global {
  interface Window {
    testStates: GameState[];
  }
}
window.testStates = [];
// 专用测试宿主，生产入口不导入。
createRoot(document.getElementById('root')!).render(
  <>
    <button id="host-button" style={{ borderRadius: 0 }}>
      宿主按钮
    </button>
    <HaojieGame
      initialState={createDemoGame()}
      storageKey={null}
      onStateChange={(s) => {
        window.testStates[0] = s;
      }}
    />
    <HaojieGame
      initialState={createDemoGame()}
      storageKey={null}
      onStateChange={(s) => {
        window.testStates[1] = s;
      }}
    />
  </>,
);
