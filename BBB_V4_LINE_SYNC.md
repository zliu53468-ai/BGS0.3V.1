# LINE prediction synchronization with BBB V4

Pinned BBB source: 5487b8aab2d9011b64d0a1587129abe6c89787e3.

The eight files in bbb_engine/v4 are copied without modification from BBB.
worker_v26.cjs invokes StablePhysicalPointV4 and only adapts its report to the existing LINE fields. The historical Python module/import names remain compatible.

Entry policy is exactly BBB's: estimator SE >0.55%, smoothed and raw B/P EV values must each be unequal (no minimum gap), the two EV choices agree, and batch direction agreement >=55%. EV sign is unrestricted. Deplete is not a mandatory entry condition. The same 3/5-batch policy, fresh cryptographic seeds, smoothing, EOR, Deplete diagnostics and exact reference are preserved.

No changes to app.py card builders, LINE transport, access codes, authorization, store.py, session schemas, static assets, bankroll display or render.yaml.

Validation: exact git-blob verification of vendored sources; six fixed-seed adapter/core parity cases; replay isolation; Python source/config contract; original LINE card builder checks; parser and UID/stale-run tests. Fresh runs across the website and LINE can differ because they use separate fresh seeds; equality is asserted with a shared explicit test seed.

Production deployment follows main through the existing Render autoDeploy configuration. No profitability or hit-rate improvement is claimed.
