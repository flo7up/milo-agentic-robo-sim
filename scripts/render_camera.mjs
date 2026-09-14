import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, sep, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createInterface } from 'node:readline';
import { chromium } from '../frontend/node_modules/playwright-core/index.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const service = process.argv.includes('--serve');
const inputIndex = process.argv.indexOf('--input');
if (!service && (inputIndex < 0 || !process.argv[inputIndex + 1])) throw new Error('--input is required');
const input = service ? null : resolve(process.argv[inputIndex + 1]);
const manifest = service ? null : JSON.parse(await readFile(resolve(input, 'manifest.json'), 'utf8'));
if (manifest && manifest.schema !== 'milo-render-benchmark-v1') throw new Error('Unknown render manifest');
const dist = resolve(root, 'frontend/dist');
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
    const texture = /^\/api\/textures\/([a-z]+)\.png$/.exec(pathname);
    const path = texture ? resolve(root, 'assets/textures', `${texture[1]}.png`) : resolve(dist, `.${pathname === '/' ? '/index.html' : pathname}`);
    if (request.method !== 'GET' || (!texture && !path.startsWith(`${dist}${sep}`))) throw new Error('Not found');
    const body = await readFile(path);
    response.writeHead(200, { 'Content-Type': ({ '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png' })[extname(path)] ?? 'application/octet-stream', 'Cache-Control': 'no-store' });
    response.end(body);
  } catch { response.writeHead(404); response.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('websocket', () => errors.push('Camera worker opened a live WebSocket'));
  await page.goto(`http://127.0.0.1:${server.address().port}/?cameraWorker=1`);
  await page.waitForFunction(() => !!window.miloGpuCamera, null, { timeout: 30000 });
  if (service) {
    let loadedEpisode = null;
    let rendererInfo = null;
    process.stdout.write(`${JSON.stringify({ ready: true })}\n`);
    for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
      const request = JSON.parse(line);
      if (request.close) break;
      try {
        const episode = `${request.packet.run_id}:${request.packet.episode_epoch}`;
        if (loadedEpisode !== episode) {
          rendererInfo = await page.evaluate(packet => window.miloGpuCamera.load(packet, 'enhanced'), request.packet);
          if (!rendererInfo.hardware) throw new Error('Enhanced camera requires a hardware GPU renderer');
          loadedEpisode = episode;
        }
        const frame = await page.evaluate(({ packet, depth }) => window.miloGpuCamera.capture(packet, depth), { packet: request.packet, depth: request.depth });
        if (errors.length) throw new Error(errors.join('\n'));
        process.stdout.write(`${JSON.stringify({ id: request.id, frame, renderer: rendererInfo.renderer })}\n`);
      } catch (error) {
        loadedEpisode = null;
        process.stdout.write(`${JSON.stringify({ id: request.id, error: error.message })}\n`);
      }
    }
  } else {
  const cases = [];
  let renderer, hardware, gpuMemory;
  for (const entry of manifest.cases) {
    if (!/^[a-z_]+-\d+-\d+$/.test(entry.name) || entry.snapshot !== `${entry.name}/snapshot.json`) throw new Error('Unsafe case path');
    const packet = JSON.parse(await readFile(resolve(input, entry.snapshot), 'utf8'));
    for (const quality of ['standard', 'enhanced']) {
      const loaded = await page.evaluate(async ({ packet, quality, seed }) => window.miloGpuCamera.load(packet, quality, seed), { packet, quality, seed: manifest.lighting_seed ?? 0 });
      renderer = loaded.renderer;
      hardware = loaded.hardware;
      await page.evaluate(packet => window.miloGpuCamera.capture(packet), packet);
      if (!gpuMemory) {
        try { gpuMemory = execFileSync('nvidia-smi', ['--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'], { encoding: 'utf8' }).trim(); }
        catch { gpuMemory = 'unavailable'; }
      }
      const timings = [], roundtrip = [];
      let frame;
      for (let sample = 0; sample < manifest.samples; sample++) {
        const started = performance.now();
        frame = await page.evaluate(packet => window.miloGpuCamera.capture(packet), packet);
        roundtrip.push(performance.now() - started);
        timings.push(frame.elapsed_ms);
        if (frame.run_id !== packet.run_id || frame.episode_epoch !== packet.episode_epoch || frame.observation_seq !== packet.observation_seq || frame.simulated_time_s !== packet.snapshot.simulated_time_s) throw new Error('Mismatched frame identity');
      }
      await writeFile(resolve(input, entry.name, `${quality}.png`), Buffer.from(frame.rgb, 'base64'));
      await writeFile(resolve(input, entry.name, `${quality}-depth.bin`), Buffer.from(frame.depth_f32, 'base64'));
      cases.push({ name: entry.name, quality, capture_ms: timings, roundtrip_ms: roundtrip, ...loaded });
      process.stdout.write(`${entry.name} ${quality}: ${Math.round(timings.reduce((sum, value) => sum + value, 0) / timings.length)} ms\n`);
    }
  }
  if (errors.length) throw new Error(errors.join('\n'));
  await writeFile(resolve(input, 'gpu-results.json'), JSON.stringify({ renderer, hardware, gpu_memory_sample: gpuMemory, cases }, null, 2));
  }
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}