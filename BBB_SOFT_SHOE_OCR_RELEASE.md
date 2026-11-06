# Restore latest LINE OCR screenshot panel + exact BBB Soft Shoe prediction

- **Target repository:** \`zliu53468-ai/BGS0.3V.1\` main, baseline \`e045e9d2f6e241511cd6dd9e757df540c386f4f4\`.
- **Last OCR-enabled LINE panel:** merged **PR #79**, commit \`1a02da63811439ab81f21959e4f15ecd2265750e\`, before PR #80 disabled screenshot input.
- **Exact BBB web prediction source:** \`zliu53468-ai/BBB\` main **\`ededd462ac68932c8a44d91f2e3fc28280c9aa88\`** (PR #158 middle-skip + PR #159 display-only Big Road). The official model used by the web page is \`soft_shoe_decision_runtime.js\`; Big Road is not part of prediction.

## Restored LINE workflow

1. Select venue via the existing image carousel. Enter bankroll if necessary.
2. Tap Start Analysis, submit an actual screenshot; existing OCR/road detector extracts full historical B/P/T outcomes (including ties). Existing quality rejection and UID run-id/version CAS remain.
3. The exact BBB engine receives the FULL B/P/T current-shoe sequence; unlike old independent player/banker score mode, it updates latent shoe particles from observed wins/losses/ties. Results are returned in the existing "分析數據" yellow/white Flex card, with B/P/T rate display and red/blue/green manual result buttons after first image.
4. Subsequent button presses append one actual B/P/T outcome and re-evaluate the full sequence. Existing authorization, trial/expiry, persistent sessions, single-UID locking, stale-result rejection and end-analysis cleanup remain in place.
5. 72-hand model depth cap matches BBB exactly; exceeding it fails visibly rather than silently discarding older shoes. Invalid/low-quality OCR and timeouts never result in invented predictions.

## Prediction semantics

- Vendored BBB JS & untrained model bundle are byte-identical and **Git blob SHA1 pinned**:
  - \`bbb_engine/soft_shoe_decision_runtime.js\`: \`7d299113cdd9cade5739c9a3e71be3ee2d80b1cd\`
  - \`bbb_engine/soft_shoe_model_bundle.js\`: \`e05280fccff3d7122a6cab0593ba4a736a71f4df\`
- LINE Node bridge reconstructs the exact same deterministic \`run(history,{},model)\` as BBB web. No per-UID predictive randomness, cached cross-UID state, mock observed ranks or point-score pairs; UID still scopes storage and analysis lifecycle.
- \`PARTICLE_COUNT=128\`, \`LIKELIHOOD_DRAWS=14\`, \`NEXT_HAND_DRAWS=14\` and the BBB \`research_view\` moderate gate (rounds >= 3, EV gap 2pp, gap-to-noise 0.70, ESS >= .35, diversity >= .20, max gap SE .08) are unchanged.
- The BBB model bundle is **untrained**; the direction shown is a research display, not a validated positive-EV wagering recommendation. The existing BGS bankroll bridge now refuses to turn this into 5–30% suggested stakes; official action remains Skip/zero bankroll fraction. Avoid calling heuristic "confidence" a hit rate.
- Screenshot scanning sees only B/P/T and road layout. It does NOT reveal actual dealt ranks or exact card composition. No claims of improved betting odds.

## Scope / rollback

Allowed source changes: \`app.py\` OCR input flow/panel copy from PR #79, \`runtime_app.py\` prediction-only seam, two new vendored source blobs, Node/Python transport and tests. \`store.py\`, \`room_ocr.py\`, \`screen_pipeline.py\`, \`screenshot_predictor.py\`, \`road_detector.py\`, \`static/liff.html\`, \`render.yaml\`, tokens, access codes, trial and disk settings are untouched. Historical score-only files remain for rollback; they are no longer the **default** LINE flow.

The protected manifest changes only its \`app.py\` hash (the UI restoration was explicitly requested). Existing member sessions, activation state and persistent disk are not wiped or rewritten by the release. To roll back, revert this PR merge commit and re-deploy; **do not** clear persistent member data.

## Validation / deployment

\`python -m unittest -v test_bbb_soft_shoe_ocr.py\` checks web parity, source hashes, middle-skip, cold 50-round budget, original OCR panel, zero-stake boundary and protected source hashes. Keep the existing Final57 and V4 workflows for compatibility testing, even though they no longer drive the OCR path.

Render \`autoDeploy: true\` tracks main. GitHub merge/CI does not prove the external Render instance has restarted or an actual LINE image was scanned; verify deployed \`/health\` and \`/prediction-version\`, then send a real screenshot through LINE to confirm end-to-end receipt.
