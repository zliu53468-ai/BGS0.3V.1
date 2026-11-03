# BBB V2 Physics Forecast First → BGS LINE prediction-only port

## Exact source / release identity

- Source repo: `zliu53468-ai/BBB`, main commit `7f49802f6f514eb7bf44aa3a61bdf4b1ffcc9920`
- Source PR #118: sampling upgrade PF256 / likelihood384 / physical EV160
- Source PR #119: compact UI only; **no prediction-core changes**
- Target repo: `zliu53468-ai/BGS0.3V.1`, based on `abf93d64734c309de8560d5306973e2042808c4c`
- Source source blobs copied byte-for-byte into `bbb_engine/` and SHA1 Git blob-checked by `bbb_v2_runtime.py`.

### Core forecast configuration (unchanged)

| Component | Value |
| --- | --- |
| Blind physics particles N (also hard cap) | 256 |
| Conditional likelihood proposals L | 384 |
| Per-particle physical EV MC samples M | 160 |
| Likelihood nominal samples / round | 98,304 |
| Physical EV nominal samples / forecast | 40,960 |
| ESS resampling threshold | 0.5N = 128 |
| Base / overdrive / one-round cushion | 0.005 / 0 / 0.035 |
| Physical / road feature weights | 5 / 0.1 |
| Early-Shoe Guard / dynamic distribution guard | Identical to BBB source |

### LINE integration

- Reuse existing `nodejs-wheel==24.19.0` from **unchanged** requirements file.
- Stateful, scope-key isolated Node worker avoids redoing full history replay on every B/P/T manual update.
- The production Python `predictor.predict` entrypoint replaces its old Final57 overlay with an isolated V2 overlay; old Final57 code remains for rollback/regression tests.
- The existing `runtime_app._attach_formal_bankroll_advice` boundary enforces `bet_allowed=false`, zero stake, and preserves the research direction display only when the original BBB research gate passes.
- BBB V2's B/P/T-only **official BET action remains SKIP**, even if a research direction (B/P) is shown. We never claim that simulated EV is validated edge, nor invent observed card values.
- User, venue, room and shoe scoping relies on the existing `scope_key` with hash isolation. Different scopes and undo are tested.

### Untouched components

**No changes** to `app.py`, `store.py`, `static/liff.html`, `screen_pipeline.py`, `screenshot_predictor.py`, `room_ocr.py`, image/OCR settings, activation and trial logic, access codes, `render.yaml`, `Procfile`, `requirements.txt`, existing model JSONs, Kelly module and persistent sessions. Existing SHA256-protected-files verification remains part of the CI.

### Production warning & tests

- LINE's first screenshot may import 50+ historical rounds and V2 computes ~139,264 nominal legal-hand proposal + MC samples per completed round at default settings. The initial replay may exceed LINE's deadline even with a persistent Node worker. A cold 50-round import is explicitly tested with a bounded 9-second worker budget; do **not** merge if it fails.
- Tests cover original JS blob identity, exact N/L/M, research-only zero-stake safety, scope isolation, repeated request idempotency, undo, old Final57 compatibility, old fixed gap tests and protected source file checks.
- Production Render autoDeploy still depends on the pre-existing main/Render settings and cannot be assumed completed until deployment health is observed.

### Rollback

Revert the port PR merge commit on `BGS0.3V.1/main`. Do not clear persistent sessions, stored licenses, activation/trial expiry times or disk volumes.
