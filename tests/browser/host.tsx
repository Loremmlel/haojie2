import { createRoot } from 'react-dom/client';
import { createDemoGame, HaojieGame } from '../../src';
import type { GameState } from '../../src';
declare global {
  interface Window {
    testStates: GameState[];
  }
}
window.testStates = [];
// Dedicated test host; never imported by the production entry point.
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
