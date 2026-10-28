# CLAUDE.md

## Model API access (Cloudflare AI Gateway)

Model calls go through the group's Cloudflare AI Gateway, which exposes an OpenAI-compatible endpoint. Examples:
`~/.local/bin/cf-python-example.py` (openai SDK) and `~/.local/bin/cf-curl-example.sh` (curl).

- Base URL: the gateway's `/compat` endpoint, as in the example scripts (chat completions at `/compat/chat/completions`).
  It is kept out of the repository: export it as `OPENAI_BASE_URL`, which the openai SDK reads when `base_url` is unset.
- Auth: `Authorization: Bearer $OPENAI_API_KEY` (the openai SDK reads `OPENAI_API_KEY` itself).
- Every request also carries the header `cf-aig-metadata: {"project_id": "$CF_PROJECT_ID", "user_id": "$CF_USER_ID"}`
  (with the openai SDK: `OpenAI(base_url=..., default_headers={"cf-aig-metadata": json.dumps({...})})`).
- Model ids are prefixed with the provider, e.g. `google-ai-studio/gemini-3.8-flash`.
- `OPENAI_API_KEY`, `CF_PROJECT_ID`, `CF_USER_ID` and `MODEL_ID` are already exported in the user's shell. Never print,
  log or commit their values, nor the gateway URL.
- Pricing of `gemini-3.8-flash`: $0.75 per million input tokens, $3.75 per million output tokens
  (https://ai.google.dev/gemini-api/docs/pricing#gemini-3.8-flash).
- In DexAgent: `agent.provider: openai`, `agent.base_url: null` (the SDK takes `OPENAI_BASE_URL`), `agent.model` the prefixed id, and
  `agent.price_usd_per_mtok` / `agent.budget_usd` set (the run refuses to start without them). The openai client must
  send the `cf-aig-metadata` header.
