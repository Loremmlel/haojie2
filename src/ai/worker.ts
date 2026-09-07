import { decide } from './search';
import type { SearchRequest, SearchResponse } from './types';
const port = self as unknown as {
  onmessage: ((e: MessageEvent<SearchRequest>) => void) | null;
  postMessage: (r: SearchResponse) => void;
};
port.onmessage = ({ data }) => {
  try {
    port.postMessage({
      id: data.id,
      decision: decide(data.observation, data.side, data.difficulty, data.limits),
    });
  } catch (error) {
    port.postMessage({ id: data.id, error: error instanceof Error ? error.message : 'AI计算失败' });
  }
};
