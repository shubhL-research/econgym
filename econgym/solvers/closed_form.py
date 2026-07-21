"""Closed-form (analytic) equilibria -- the single source of truth.

Each environment's ``equilibrium()`` delegates to the matching function here so
that the environment and its solver can NEVER disagree. Functions take model
parameters and return a small dict of the closed-form quantities.

This module is append-only across the v1 build: new environments add their own
function without touching existing ones.
"""
from __future__ import annotations

import numpy as np


def bertrand_diff_nash(alpha, beta, gamma, c, n):
    """Symmetric interior Nash of the differentiated-good (linear-demand) Bertrand game.

    Model (per firm ``i``)::

        q_i   = alpha - beta * p_i + gamma * sum_{j != i} p_j       (0 < gamma < beta)
        pi_i  = (p_i - c) * q_i

    FOC ``d pi_i / d p_i = 0`` gives the best-response map
    ``BR_i(S_{-i}) = (alpha + beta*c + gamma*S_{-i}) / (2*beta)`` with
    ``S_{-i} = sum_{j != i} p_j``. Imposing symmetry ``p_i = p*`` (so
    ``S_{-i} = (n-1) p*``) yields

    .. math::

        p^* = \\frac{alpha + beta \\, c}{2 beta - gamma (n-1)}.

    Derived quantities at the symmetric equilibrium::

        q_i^*  = alpha - beta * p^* + gamma * (n-1) * p^*
        pi_i^* = (p^* - c) * q_i^*

    Parameters
    ----------
    alpha, beta, gamma, c : float
        Demand intercept, own-price slope, cross-price slope, marginal cost.
        Requires ``0 < gamma < beta``.
    n : int
        Number of firms. Interior/stability requires ``2*beta - gamma*(n-1) > 0``.

    Returns
    -------
    dict
        ``{"p", "q_i", "profit_i", "contraction_factor"}`` where
        ``contraction_factor = gamma*(n-1)/(2*beta)`` is the best-response
        contraction modulus (``< 1`` iff the symmetric Nash is a stable fixed
        point of best-response dynamics).
    """
    n = int(n)
    denom = 2.0 * beta - gamma * (n - 1)
    if denom <= 0.0:
        raise ValueError(
            "non-interior differentiated Bertrand: require 2*beta - gamma*(n-1) > 0, "
            f"got 2*{beta} - {gamma}*{n - 1} = {denom}"
        )
    p = (alpha + beta * c) / denom
    q_i = alpha - beta * p + gamma * (n - 1) * p
    profit_i = (p - c) * q_i
    return {
        "p": float(p),
        "q_i": float(q_i),
        "profit_i": float(profit_i),
        "contraction_factor": float(gamma * (n - 1) / (2.0 * beta)),
    }


def second_price_bne(n: int = 2) -> dict:
    """Equilibrium of a second-price (Vickrey) sealed-bid auction with ``n``
    bidders whose private values are iid ``Uniform[0, 1]``.

    Truthful bidding ``b(v) = v`` is a **weakly dominant** strategy (dominance,
    not merely a Bayes-Nash equilibrium). Fix any opponent bids and let ``m`` be
    the highest of them. Bidding truthfully wins iff ``v > m`` and pays ``m``,
    capturing surplus ``v - m > 0`` exactly when winning is profitable and never
    buying an unprofitable win. Any deviation either forfeits a profitable win or
    buys an unprofitable one, so the payoff can only weakly fall. The bid
    function is therefore ``n``-independent (slope 1).

    Expected seller revenue = mean of the second-highest of ``n`` iid ``U[0,1]``
    draws (the price the winner pays) = ``E[X_(n-1)] = (n-1)/(n+1)`` -- identical
    to the first-price auction's expected revenue (revenue equivalence).

    Parameters
    ----------
    n : int
        Number of bidders (default 2). Only the *revenue* depends on ``n``; the
        dominant strategy ``b(v)=v`` does not.

    Returns
    -------
    dict
        ``{"bid_fn", "bid_slope", "dominant", "expected_revenue"}`` where
        ``bid_fn`` is the identity ``v -> v``, ``bid_slope == 1.0``,
        ``dominant is True``, and ``expected_revenue == (n-1)/(n+1)``.
    """
    n = int(n)
    return {
        "bid_fn": (lambda v: v),
        "bid_slope": 1.0,
        "dominant": True,
        "expected_revenue": (n - 1) / (n + 1),
    }


def repeated_pd_threshold(T: float, R: float, P: float, S: float) -> dict:
    """Closed-form benchmark for the infinitely-repeated Prisoner's Dilemma.

    Stage bimatrix (own payoff; row = own action C/D, col = opponent action)::

                       opp C     opp D
            own C    (R, R)    (S, T)
            own D    (T, S)    (P, P)

    with the PD orderings ``T > R > P > S`` (``D`` strictly dominates ``C`` in the
    stage game) and efficiency ``2R > T + S`` (mutual cooperation is efficient).

    One-shot Nash is ``(D, D)`` (``T > R`` and ``P > S`` => ``D`` strictly
    dominant). Grim-trigger cooperation is a subgame-perfect equilibrium of the
    infinitely-repeated game iff the common per-period discount factor satisfies

    .. math::

        \\delta \\ge \\frac{T - R}{T - P}          \\quad\\text{(folk-theorem threshold)}

    Derivation. On the cooperative path a player earns ``R / (1 - delta)``. A
    one-shot deviation earns ``T`` today, then the grim punishment ``P`` forever:
    ``T + delta * P / (1 - delta)``. Cooperation is sustained iff

        ``R / (1 - delta) >= T + delta * P / (1 - delta)``
        <=> ``R >= (1 - delta) * T + delta * P``
        <=> ``delta * (T - P) >= T - R``
        <=> ``delta >= (T - R) / (T - P)``.

    Parameters
    ----------
    T, R, P, S : float
        Temptation, Reward, Punishment, Sucker stage payoffs (``T > R > P > S``).

    Returns
    -------
    dict
        ``{"grim_threshold": (T-R)/(T-P), "one_shot_nash": ("D", "D")}``.

    Raises
    ------
    ValueError
        If ``T <= P`` (the threshold ``(T-R)/(T-P)`` is otherwise undefined /
        non-positive-denominator -- the PD ordering ``T > R > P > S`` guarantees
        ``T > P``).
    """
    T, R, P, S = float(T), float(R), float(P), float(S)
    if T <= P:
        raise ValueError(
            f"repeated_pd_threshold requires T > P (got T={T}, P={P}); the "
            "grim-trigger threshold (T-R)/(T-P) is otherwise undefined."
        )
    return {
        "grim_threshold": (T - R) / (T - P),
        "one_shot_nash": ("D", "D"),
    }


def rubinstein_split(delta: float) -> dict:
    """Unique subgame-perfect equilibrium of the Rubinstein alternating-offers
    bargaining game (two players splitting a unit pie, symmetric per-round
    discount ``delta``).

    Two players alternate proposing a split of a unit pie: player 1 proposes in
    odd rounds, player 2 in even rounds, with a common per-round discount factor
    ``delta in [0, 1)``. A proposer offers a split; the responder accepts (both
    payoffs realized, discounted by ``delta**(round-1)``) or rejects (roles swap
    and the game continues).

    The stationary SPE is derived by a one-shot indifference / stationarity
    argument. Let ``v`` be the proposer's SPE share. If the responder rejects it
    becomes the proposer next round and secures ``v`` then, worth ``delta*v``
    discounted to the present, so it must be offered exactly ``delta*v`` to be
    kept indifferent; the proposer therefore keeps ``1 - delta*v``. Stationarity
    (the proposer's share is the same in every subgame) gives

    .. math::

        v = 1 - \\delta v  \\;\\Longrightarrow\\; v (1 + \\delta) = 1
          \\;\\Longrightarrow\\; v = \\frac{1}{1 + \\delta},

    so the responder receives ``delta / (1 + delta)`` and agreement occurs
    immediately in round 1 (no costly delay). As ``delta -> 1`` both shares tend
    to ``1/2`` (the symmetric split); at ``delta = 0`` the proposer takes the
    whole pie.

    Parameters
    ----------
    delta : float
        Common per-round discount factor, ``0 <= delta < 1``.

    Returns
    -------
    dict
        ``{"proposer_share": 1/(1+delta), "responder_share": delta/(1+delta),
        "agreement_round": 1}``. The two shares sum to exactly 1.

    Raises
    ------
    ValueError
        If ``delta`` is not in ``[0, 1)`` (at ``delta = 1`` the game has a
        continuum of SPE and the unique-split formula no longer applies).
    """
    delta = float(delta)
    if not (0.0 <= delta < 1.0):
        raise ValueError(
            f"rubinstein_split requires 0 <= delta < 1, got delta={delta}. "
            "(At delta=1 the alternating-offers game has a continuum of SPE.)"
        )
    proposer_share = 1.0 / (1.0 + delta)
    responder_share = delta / (1.0 + delta)
    return {
        "proposer_share": proposer_share,
        "responder_share": responder_share,
        "agreement_round": 1,
    }


def cournot_nash(a, b, c, n):
    """Symmetric interior Nash of an ``n``-firm homogeneous-good Cournot game.

    Model. ``n`` firms choose quantities ``q_i >= 0``; inverse demand
    ``P = a - b*Q`` with ``Q = sum_i q_i``; constant marginal cost ``c``; payoff
    ``pi_i = (P - c) * q_i``. Firm ``i`` solves
    ``max_{q_i} (a - b(q_i + Q_{-i}) - c) q_i`` with FOC
    ``a - b*Q_{-i} - 2*b*q_i - c = 0``, giving the best-response map
    ``BR_i(Q_{-i}) = (a - c - b*Q_{-i}) / (2b)``. Imposing symmetry
    ``q_j = q_i* for all j`` (so ``Q_{-i} = (n-1) q_i*``) yields the textbook
    solution

    .. math::

        q_i^*    = \\frac{a - c}{b\\,(n + 1)}, \\quad
        Q^*      = \\frac{n\\,(a - c)}{b\\,(n + 1)}, \\quad
        P^*      = \\frac{a + n\\,c}{n + 1}, \\quad
        \\pi_i^* = \\frac{(a - c)^2}{b\\,(n + 1)^2}.

    Requires the interior condition ``a > c`` (otherwise output collapses to 0).
    Reduces correctly to monopoly at ``n = 1`` (``q* = (a-c)/(2b)``,
    ``P* = (a+c)/2``, ``pi* = (a-c)^2/(4b)``) and to marginal-cost pricing
    ``P* -> c`` as ``n -> infinity``.

    Note on best-response dynamics: the *simultaneous* best-response map has an
    eigenvalue ``-(n-1)/2`` along the aggregate direction, so undamped Picard
    iteration is only a contraction for ``n = 2`` (neutrally stable at ``n = 3``,
    divergent for ``n >= 4``). A relaxed / damped iteration with factor
    ``lambda in (0, 4/(n+1))`` converges to this same Nash fixed point.

    Parameters
    ----------
    a, b, c : float
        Demand intercept, demand slope (``b > 0``), and marginal cost (``a > c``
        for an interior equilibrium).
    n : int
        Number of firms (``n >= 1``).

    Returns
    -------
    dict
        ``{"q_i": q_i*, "Q": Q*, "P": P*, "profit_i": pi_i*}`` (python floats).
    """
    a = float(a)
    b = float(b)
    c = float(c)
    n = int(n)
    if b <= 0.0:
        raise ValueError(f"cournot_nash requires demand slope b > 0, got b={b}")
    if n < 1:
        raise ValueError(f"cournot_nash requires n >= 1 firms, got n={n}")

    q_i = (a - c) / (b * (n + 1))
    Q = n * q_i
    P = (a + n * c) / (n + 1)
    profit_i = (a - c) ** 2 / (b * (n + 1) ** 2)
    return {
        "q_i": float(q_i),
        "Q": float(Q),
        "P": float(P),
        "profit_i": float(profit_i),
    }


def first_price_bne(n) -> dict:
    """Symmetric Bayes-Nash equilibrium of a first-price sealed-bid auction with
    ``n`` bidders whose private values are iid ``Uniform[0, 1]``.

    Each bidder submits a bid; the highest bid wins and **pays its own bid**
    (pay-your-bid). Suppose every opponent plays ``b(v) = ((n-1)/n) v``. A bidder
    with value ``v`` bidding ``b`` wins iff *every* opponent value satisfies
    ``v_j < b * n/(n-1)`` -- probability ``(b * n/(n-1))^{n-1}`` for
    ``b in [0, (n-1)/n]`` -- so expected payoff is
    ``U(b) = (v - b) * (b * n/(n-1))^{n-1}``. Setting ``dU/db = 0`` gives
    ``b = ((n-1)/n) v``, i.e. the candidate is a mutual best response.

    Revenue is the winner's (highest) bid ``((n-1)/n) * max_i v_i``; the maximum
    of ``n`` iid ``U[0,1]`` draws has mean ``n/(n+1)``, so
    ``E[revenue] = ((n-1)/n)(n/(n+1)) = (n-1)/(n+1)`` -- equal to the second-price
    auction's expected revenue (the revenue-equivalence theorem).

    Parameters
    ----------
    n : int
        Number of bidders (``n >= 2`` for an interior equilibrium; ``n == 1``
        yields ``bid_slope == 0`` -- a monopsony bidder shades to zero).

    Returns
    -------
    dict
        ``{"bid_slope": (n-1)/n, "bid_fn": v -> ((n-1)/n) v,
        "expected_revenue": (n-1)/(n+1)}``. ``bid_fn`` is vectorized over array
        input.
    """
    n = int(n)
    slope = (n - 1) / n
    return {
        "bid_slope": slope,
        "bid_fn": (lambda v, s=slope: s * np.asarray(v, dtype=float)),
        "expected_revenue": (n - 1) / (n + 1),
    }


def public_goods_nash(w, r, n) -> dict:
    """Closed-form benchmark for the linear public-goods (VCM) game.

    Model. ``n`` players each hold endowment ``w`` and contribute ``c_i in [0, w]``
    to a common pool; payoff ``pi_i = w - c_i + r * sum_j c_j`` (the ``j``-sum
    includes ``i``), with the social-dilemma condition ``1/n < r < 1``.

    Own-contribution marginal payoff ``d pi_i / d c_i = -1 + r < 0`` (since
    ``r < 1``), so **free-riding** ``c_i = 0`` is a strictly dominant strategy and
    the unique Nash gives ``pi_i = w``. Social welfare
    ``sum_i pi_i = n*w + (r*n - 1) * sum_j c_j`` has slope ``r*n - 1 > 0`` in every
    ``c_j`` (since ``r > 1/n``), so it is maximised at **full contribution**
    ``c_i = w``, giving ``pi_i = r*n*w``. The per-player Nash-vs-optimum gap is
    ``(r*n - 1) * w``.

    Parameters
    ----------
    w : float
        Per-player endowment.
    r : float
        Marginal per-capita return (MPCR); the social dilemma holds for
        ``1/n < r < 1``.
    n : int
        Number of players.

    Returns
    -------
    dict
        ``{"nash_contribution": 0.0, "nash_payoff": w, "optimum_contribution": w,
        "optimum_payoff": r*n*w, "gap_per_player": (r*n - 1)*w}``.
    """
    w = float(w)
    r = float(r)
    n = int(n)
    return {
        "nash_contribution": 0.0,
        "nash_payoff": w,
        "optimum_contribution": w,
        "optimum_payoff": r * n * w,
        "gap_per_player": (r * n - 1.0) * w,
    }
