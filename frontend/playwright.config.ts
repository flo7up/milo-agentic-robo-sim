import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const python = process.env.ROBOSIM_PYTHON ?? fileURLToPath(new URL('../.runtime/env/python.exe', import.meta.url));

export default defineConfig({
  testDir: './tests',
  timeout: 45000,
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:8001', channel: 'msedge', screenshot: 'only-on-failure' },
  webServer: {
    command: `"${python}" -m tests.test_agent`, cwd: root,
    url: 'http://127.0.0.1:8001/api/state', reuseExistingServer: false, timeout: 30000,
  },
});