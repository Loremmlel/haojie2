import { createServer } from 'node:http';
import { watch } from 'node:fs';
import { buildGame } from './build.mjs';

let html = await buildGame();
let building = false, queued = false, timer;
async function rebuild() {
  if (building) { queued = true; return; }
  building = true;
  try { html = await buildGame(); console.log('Updated. Refresh your browser to view changes.'); }
  catch (error) { console.error(error); }
  finally { building = false; if (queued) { queued = false; void rebuild(); } }
}
const port = Number(process.env.PORT ?? 4173);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PORT must be between 1 and 65535.');
const server = createServer((req, res) => {
  if (req.url === '/favicon.ico') { res.writeHead(204); res.end(); return; }
  if (req.url !== '/' && req.url !== '/index.html') { res.writeHead(404); res.end('Not found'); return; }
  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' }); res.end(html);
});
const watcher = watch('src', { recursive: true }, () => { clearTimeout(timer); timer = setTimeout(() => void rebuild(), 100); });
server.listen(port, '127.0.0.1', () => console.log(`Haojie is ready at http://127.0.0.1:${port}`));
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { watcher.close(); server.close(); });
