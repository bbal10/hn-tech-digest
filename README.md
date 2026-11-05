# hn-tech-digest

A one-shot [Mistral Workflow](https://docs.mistral.ai/studio/workflows/overview) that:

1. Fetches the HackerNews front page (`hacker-news.firebaseio.com/v1/frontpage.json`).
2. Uses a Mistral LLM to select **tech-industry news only** (AI, chips, big-tech launches, security, dev tools, funding) — personal blogs and off-topic posts are excluded — and writes a short summary per story plus a 180–260 word podcast script (~1–2 minutes).
3. Generates the audio briefing with Voxtral TTS (`voxtral-mini-tts-3-0`, voice `gb_jane_neutral`, MP3).
4. Sends the digest to your Gmail inbox with the MP3 attached as `hn-tech-digest.mp3`, acting as the triggering user (on-behalf-of) through the `gmail` connector.

The workflow is **not scheduled**. Trigger it on demand:

- Mistral Console → Workflows → `hn-tech-digest` → Start workflow, or
- `mistral_mcp_execute_workflow` / `POST /v1/workflows/hn-tech-digest/execute` with input `{"recipient": "you@gmail.com", "max_items": 8}` (both optional).

## Layout

- `apps/worker/src/worker/workflows/hn_digest.py` — workflow, activities, connector slot
- `apps/worker/src/worker/entrypoints/worker.py` — worker entrypoint (discovers workflows, runs the worker)
- `Dockerfile` — image built by the managed deployment; `CMD` starts the worker
- `tests/` — unit tests with mocked HTTP/LLM/connector effects

## Local development

```bash
uv sync
uv run pytest
```

Local worker runs need `MISTRAL_API_KEY` in the environment; the managed deployment on Mistral Cloud authenticates with an injected service-account token instead.

## Deployment

Managed, hardened deployment (required for `on_behalf_of=True`):

```
deployment_name = hn-tech-digest
github_url      = https://github.com/bbal10/hn-tech-digest
revision        = <pushed commit SHA>
dockerfile_path = Dockerfile (build directory = repo root)
hardened        = true
```

Prerequisites: the [Mistral GitHub App](https://github.com/apps/mistralai) installed on this repository, and pay-as-you-go billing enabled on the Mistral organization.
