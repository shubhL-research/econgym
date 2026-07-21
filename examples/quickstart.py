"""EconGym quickstart: two Q-learners in the v0 Bertrand market. Prints the
collusion index Delta, the normalised entropy, and whether the greedy policies
converged."""
from econgym import BertrandEnv, QLearner, run_episode, metrics

env = BertrandEnv(n=2, K=7, c=1.0, p_max=10.0)
agents = [QLearner(env, alpha=.1, gamma=.95, epsilon=.1, eps_decay=3e-4)
          for _ in range(2)]
res = run_episode(env, agents, T=30_000, seed=0, track_conv=True)
d = metrics.delta_index(res.prices, res.profits, env.grid, env.c, env.n, T0=2000)
h = metrics.mean_entropy(res.prices, env.K, T0=2000)[1]
print(f"delta={d:.3f}  entropy={h:.3f}  converged={res.converged}")
