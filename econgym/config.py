"""Canonical baseline presets and classifier thresholds for the EconGym
Bertrand environment (v0).

`BASE` mirrors the original Paper-1 `config.py` baseline (grid K=7, the coarse
contrast grid). The paper's *headline* baseline uses K=21; the reproduction
config in `targets.json` overrides K -> 21 explicitly, so `BASE` is left at the
original value and callers pass `K=21` when reproducing Table 1.
"""

BASE = dict(
    n=2,            # firms
    K=7,            # price grid points (original BASE; paper headline uses K=21)
    T=30_000,       # learning horizon
    alpha=0.10,     # learning rate
    gamma=0.95,     # discount factor (patience)
    epsilon=0.10,   # initial exploration
    eps_decay=3e-4, # annealed schedule: eps_t = eps / (1 + eps_decay * t)
    c=1.0,          # marginal cost
    p_min=0.0,
    p_max=10.0,
    T0=2_000,       # classification / averaging window (final T0 periods)
)

# Classification thresholds.
# H_STAR is the paper's data-derived antimode of the K-normalised entropy
# marginal (Section 4). The original config.py carried a 0.50 placeholder that
# was "re-estimated in the classifier-validation step"; 0.288 is the value the
# published results actually use and the robust choice for the K=21 baseline.
H_STAR = 0.288
H_STAR_DEFAULT = 0.50   # historical placeholder kept for reference only
P_STAR_FACTOR = 2.0     # collusive if mean price >= P_STAR_FACTOR * c
