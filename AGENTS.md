# Robot Lab Development

This project uses the microsoft-foundry skill for local model control. Before working on or answering questions about Foundry agents, read the microsoft-foundry skill. If you are in VS Code, read the vscode-microsoft-foundry skill first.

- This is a local FastAPI/PyBullet application using the OpenAI Responses SDK, not a hosted Foundry agent. Preserve its architecture; do not provision cloud resources or scaffold azd without a user request.
- Run physics tests with `./.runtime/env/python.exe -m pytest -q` on this Windows setup.
- Frontend: `npm --prefix frontend run build`, then `npm --prefix frontend test`. Playwright owns an isolated port-8001 fixture using real physics and scripted inference. It does not contact Foundry.
- Use the CFS-protected npm feed `https://packagefeedproxy.microsoft.io/npm/`. Do not bypass blocked registries.
- Never read or print credential-bearing environment values. The app supports backend/.env and root .env; existing process variables take precedence.
- Keep model inputs restricted to AgentObservation and the corresponding head-camera image. Never include spectator/evaluator state or execute tools outside the validated worker.
- Stop, takeover, reset, and disconnect must invalidate pending model motion. Keep inference serialized and feedback interval changes effective during waits.
- Label scripted tests separately from real-model validation. A configured deployment name does not establish the underlying model identity or benchmark success.