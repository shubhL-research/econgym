# EconGym

[![CI](https://github.com/shubhL-research/econgym/actions/workflows/ci.yml/badge.svg)](https://github.com/shubhL-research/econgym/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

**EconGym** is a small, Gymnasium-style **suite of economics reinforcement-learning
environments** - modular market/game "physics" you can drop learning agents into
and measure what emerges. Every environment ships with (1) faithful economic
physics, (2) its **closed-form theoretical benchmark** exposed via an
`equilibrium()` / `benchmark()` hook, and (3) a test that **validates the env
against that benchmark** - either the static equilibrium matches the closed form
exactly, best-response / greedy learners converge to the known equilibrium within
a stated tolerance, or (for dominant-strategy / SPE games) profitable-deviation
checks confirm the equilibrium.

The whole suite reuses one shared interface: a PettingZoo-parallel-style
`EconEnv` base, a single `Agent` API, one seeded `run_episode`, and a pure,
agent-free `solvers` package that is the single source of truth for every
benchmark (each env's `equilibrium()` delegates to it, so env and solver can never
drift apart).

## Environments

| Environment | class | native action | closed-form benchmark | theory |
|---|---|---|---|---|
| Homogeneous-good **Bertrand** (v0 collusion reproduction) | `BertrandEnv` | price index on a `K`-point grid | discrete Bertrand-Nash `(p_comp−c)/n` vs joint-monopoly `(p_max−c)/n`; RL-collusion `delta_index` | Bertrand (1883); Calvano et al. (2020, *AER*) |
| **Cournot** quantity competition | `CournotEnv` | quantity `q_i ≥ 0` (`Box`) | `q*=(a−c)/(b(n+1))`, `P*=(a+nc)/(n+1)`, `π*=(a−c)²/(b(n+1)²)` | Cournot (1838) |
| Differentiated-good **Bertrand** | `BertrandDiffEnv` | price `p_i` (`Box`) | `p*=(α+βc)/(2β−γ(n−1))` with derived `q*`, `π*` | linear-demand differentiated Bertrand (Singh & Vives 1984) |
| **First-price** sealed-bid auction | `FirstPriceAuctionEnv` | bid `b_i∈[0,1]` (`Box`) | symmetric BNE `b(v)=((n−1)/n)v`; `E[rev]=(n−1)/(n+1)` | Vickrey (1961); revenue equivalence |
| **Second-price** (Vickrey) auction | `SecondPriceEnv` | bid `b_i∈[0,1]` (`Box`) | weakly-dominant truthful `b(v)=v`; `E[rev]=(n−1)/(n+1)` | Vickrey (1961) |
| Linear **public goods** (VCM) | `PublicGoodsEnv` | contribution `c_i∈[0,w]` (`Box`) | dominant free-ride `c*=0`, `π=w`; social optimum `c=w`, `π=rnw` | Samuelson (1954); VCM (Isaac & Walker 1988) |
| Infinitely-repeated **Prisoner's Dilemma** | `RepeatedPDEnv` | C/D index (`Discrete(2)`) | one-shot Nash `(D,D)`; grim-trigger SPE iff `δ ≥ (T−R)/(T−P)` | Folk Theorem (Friedman 1971) |
| **Rubinstein** alternating-offers bargaining | `RubinsteinEnv` | offered split (`Box`) | unique SPE `1/(1+δ)` : `δ/(1+δ)`, immediate agreement | Rubinstein (1982) |

Each row has a `tests/test_env_*.py` that validates the env against the benchmark
in the last column, plus `tests/test_solvers.py` for the solver functions and
`tests/test_integration.py` for suite-wide `equilibrium() == solver` agreement.

## Shared agents & solvers

**Agents** (`econgym/agents/`, one shared `Agent` API):

| agent | class | kind |
|---|---|---|
| Memory-1 ε-greedy Q-learner | `QLearner` | tabular RL (v0 collusion) |
| Stateless mean-based bandit | `MeanBased` | running-mean bandit (v0 collusion) |
| Thompson sampling | `Thompson` | Bayesian bandit |
| UCB1 | `UCB1` | optimism-under-uncertainty bandit |
| Regret matching | `RegretMatching` | no-regret / correlated-eq dynamics |
| Fictitious play | `FictitiousPlay` | belief-based best response |
| Best response | `BestResponse` | myopic exact best response |

**Solvers** (`econgym/solvers/`, pure & deterministic - the benchmark oracle):
`closed_form` (`cournot_nash`, `bertrand_diff_nash`, `first_price_bne`,
`second_price_bne`, `public_goods_nash`, `repeated_pd_threshold`,
`rubinstein_split`), `normal_form.support_enumeration` (all Nash of a 2-player
bimatrix via support enumeration), and `best_response` (`best_response_iteration`
continuous fixed-point + `fictitious_play` discrete belief dynamics).

## The two faces (the v0 headline - do not "fix" it)

The v0 homogeneous-good Bertrand environment is a **faithful reproduction** of
**Paper 1 - "The Two Faces of Algorithmic Collusion: Memory, Spuriousness, and a
Model-Free Entropy Diagnostic in Homogeneous-Good Bertrand Competition"** (Shubh
Lamba). Both faces hold **simultaneously** at the K=21 baseline:

- **Frequency (regime share):** the *stateless mean-based* learner collides into
  supra-competitive prices **more often** than the stateful Q-learner
  (`mean_collusive_share ≫ q_collusive_share`; two-proportion `z ≫ 0`).
- **Intensity (`delta_index`):** when the Q-learner *does* collude it colludes
  **harder** (`q_delta_mean > mean_delta_mean`; two-sample `z ≪ 0`).

"The stateless learner colludes more often but weakly; the stateful learner less
often but harder." This is the paper's central result, not a bug.

Single-seed **Q-learning traces are byte-for-byte identical** to the original
research code, and after one intended, documented one-draw RNG offset the
**mean-based trace is also byte-exact** (see *Faithfulness*). The faithfulness
lock runs everywhere via a vendored golden trace recorded from the original code
(`tests/data/bytematch_golden.npz`, `tests/test_bytematch_golden.py`), and
additionally cross-checks the live original source when present
(`tests/test_metrics.py`).

## Install

```bash
pip install -e .          # numpy only; test extra: pip install -e ".[test]"
```

## Quickstart

```python
from econgym import BertrandEnv, QLearner, run_episode, metrics

env = BertrandEnv(n=2, K=7, c=1.0, p_max=10.0)
agents = [QLearner(env, alpha=.1, gamma=.95, epsilon=.1, eps_decay=3e-4)
          for _ in range(2)]
res = run_episode(env, agents, T=30_000, seed=0, track_conv=True)

d = metrics.delta_index(res.prices, res.profits, env.grid, env.c, env.n, T0=2000)
h = metrics.mean_entropy(res.prices, env.K, T0=2000)[1]
print(f"delta={d:.3f}  entropy={h:.3f}  converged={res.converged}")
```

Run it with `python examples/quickstart.py`. Swap `QLearner` for `MeanBased` to
run the stateless benchmark; the runner is **n-general** (pass `n>2` firms and one
agent per firm). Query any environment's closed-form benchmark directly:

```python
import econgym
print(econgym.list_envs())             # ['Bertrand-v0', 'BertrandDiff-v0', 'Cournot-v0', ...]
env = econgym.make("Cournot-v0", n=3)  # Gym-style registry; kwargs pass through
print(env.equilibrium())               # {'q_i': ..., 'Q': ..., 'P': ..., 'profit_i': ...}
```

## v0 model (identical to Paper 1)

- **Market.** `K`-point linear price grid on `[p_min, p_max]`. Lowest price serves
  one unit of inelastic demand; ties split equally (`step_profits`). Static
  benchmarks: `nash_profit` = `(p_comp − c)/n` with `p_comp = min{p ∈ grid :
  p ≥ c}` (0 iff a grid point equals `c`); `monopoly_profit` = `(p_max − c)/n`.
- **Q-learner.** Memory-1, state = opponent's previous-period price index,
  `Q(s,a) += α·(π + γ·maxₐ' Q(s',a') − Q(s,a))`. ε-greedy with annealed
  `εₜ = ε/(1 + eps_decay·t)`; exploitation is deterministic first-max.
- **Mean-based.** Stateless bandit; running mean payoff per price; matched ε
  exploration; **randomized** tie-break on exploit (`_randargmax`).
- **Metrics.** K-normalised Shannon `entropy_bits` (Miller-Madow corrected),
  `mean_entropy` over the final `T0` window, collusion `delta_index` ∈ [0,1]
  (discrete Bertrand-Nash = 0, joint monopoly = 1), non-circular `is_converged`
  (greedy-policy stability over the final 10%), and the `regime` classifier
  (`Chaotic` if `H_norm ≥ H*`, else `Competitive`/`Collusive` by mean price).

## Reproduction & faithfulness

`tests/test_reproduce_paper1.py` re-runs the paper's headline baseline (**K=21**,
`T=30000`, seeds 0..49, both learners) through the *real* package and asserts the
aggregate Table-1 statistics in `targets.json` within tolerance - including both
signed z-statistics (`delta_z ≤ −6`, `share_z ≥ 5`). Every target, tolerance, and
acceptance bound is **loaded from `targets.json`** (the single source of truth,
resolved relative to the source tree) rather than hard-coded in the test.

A single shared `np.random.default_rng(seed)` is threaded through the episode.
Draw order: (1) each `QLearner.reset` draws `rng.normal(0, 1e-6, (K,K))` in agent
order (`MeanBased.reset` draws nothing); (2) `env.reset` draws one
`rng.integers(0, K, size=n)` for the initial prices; (3) per step, in agent order,
`rng.random()` **always** (the ε-test), then `rng.integers(K)` **iff** exploring.

- **Q-learning: byte-for-byte identical** to the original - locked everywhere by
  the vendored golden trace (`tests/test_bytematch_golden.py`) and additionally
  cross-checked against the live original source when present
  (`tests/test_metrics.py`; the original path is overridable via
  `ECONGYM_ORIGINAL_SRC`, and if that variable is set but the source is missing
  the check hard-fails instead of skipping).
- **Mean-based: byte-exact after one documented offset draw.** The unified
  `env.reset` adds exactly one `rng.integers` draw the original `run_meanbased`
  lacked; after that single offset the mean-based trace is byte-exact (the
  `_randargmax` randomized tie-break RNG contract is pinned by a source-independent
  unit test). Measured effect on the 50-seed K=21 aggregates: Δdelta ≈ 0.004,
  Δentropy ≈ 0.006, Δconv ≈ 0.02 - inside tolerance.

```bash
pip install -e ".[test]"
python -m pytest -q       # full suite; the reproduce test finishes in ~3 min
```

## Roadmap (wave 2 - deferred)

The v1 suite is the shared spine; the following are deferred to a later wave and
will reuse the same `EconEnv` / `Agent` / `run_episode` / `solvers` interface:

- **Deep-RL agents** - DQN / PPO function-approximation learners for the
  continuous (`Box`) environments.
- **LLM-agent arena** - drop language-model agents into the same envs and compare
  against the closed-form benchmarks.
- **Combinatorial auctions** - multi-item / package bidding with VCG benchmarks.
- **Macro agent-based model (ABM)** - a heterogeneous-agent macro environment.
- **Network games** - strategic interaction on graphs (public goods / games on
  networks).
- **Search-and-matching** - labor/marketplace search with Diamond–Mortensen–
  Pissarides-style benchmarks.

## Publishing to PyPI

Releases publish automatically via `.github/workflows/publish.yml` using PyPI
Trusted Publishing (no stored token). One-time setup: create a PyPI account and
an `econgym` project, add a Trusted Publisher for `shubhL-research/econgym` with
workflow `publish.yml`, then publish a GitHub Release. Until then, install from
source with `pip install -e .`.

## Citation

> Shubh Lamba. *The Two Faces of Algorithmic Collusion: Memory, Spuriousness, and
> a Model-Free Entropy Diagnostic in Homogeneous-Good Bertrand Competition.*
> Submitted to the *Journal of Economic Interaction and Coordination* (JEIC).
> Code & data archived on Zenodo: **DOI 10.5281/zenodo.20788510**.
> Paper repository: https://github.com/shubhL-research/algorithmic-bertrand-rl

```bibtex
@software{lamba_econgym_2026,
  author  = {Lamba, Shubh},
  title   = {EconGym: a Gymnasium-style suite of economics RL environments
             with closed-form equilibrium benchmarks},
  year    = {2026},
  version = {0.2.0},
  doi     = {10.5281/zenodo.20788510},
  url     = {https://github.com/shubhL-research/econgym}
}
```

## License

MIT - see [LICENSE](LICENSE).
