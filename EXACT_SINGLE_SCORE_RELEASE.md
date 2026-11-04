# Independent conditional particles v1

Each point-only input creates a fresh PF96. Each particle independently draws a legal first-four proposal, enumerates legal third-card branches matching both final totals, and accepts with the matching probability mass. This yields conditional draws without the previous finite-proposal likelihood weights. Each accepted hand depletes only its own hypothetical shoe; M96 next-hand simulation and V4 display thresholds remain unchanged.

A per-particle 4096-proposal limit fails the request atomically rather than substituting an unconditioned shoe. Reusing this mode across multiple hands is rejected. Legacy historical/BPT modes retain their existing sampler.

The initial composition remains an eight-deck assumption, not observed remaining cards. Determinism, complete conditional draws and cross-platform parity are correctness checks, not evidence of increased real-world hit rate. Roll back by removing exactSingleScore:true from the web CONFIG and LINE worker CONFIG together, and repin the LINE source.
