#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FerQ-Bc HOBO->QUBO encoder (Algorithm 2 of the manuscript), with a rigorous
spectrum-preservation verifier and ancilla sharing (Corollary 1(iv)).

Domain-wall chain for |T|<=5, binary-counter chain for |T|>5. High-order monomials
that share a sub-aggregate T reuse the same chain (one chain, extra couplers, zero
extra ancillas), which is the paper's structured-problem advantage. Verified against
Appendix worked Example A (degree-5 domain wall) and by exhaustive spectrum checks."""
from __future__ import annotations
import os, sys, math, itertools
from collections import defaultdict, Counter
import numpy as np

try:
    from reducers.base import HOBO
except ImportError:                       # allow running from outside the repo root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from reducers.base import HOBO


# ---- polynomial helpers over binary vars (x^2 = x) ------------------------- #
def add_term(P, keys, coeff):
    if abs(coeff) < 1e-12:
        return
    P[frozenset(keys)] += coeff

def square_of_linear(P, lin, const, scale):
    """Add scale * (sum_v lin[v] x_v + const)^2 to P (binary, x^2=x)."""
    vs = list(lin.keys())
    add_term(P, [], scale * const * const)
    for v in vs:
        a = lin[v]
        add_term(P, [v], scale * (a * a + 2 * a * const))
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            add_term(P, [vs[i], vs[j]], scale * 2 * lin[vs[i]] * lin[vs[j]])

def _lit_poly(lit):
    v, comp = lit
    return {frozenset([v]): 1.0} if comp == 0 else {frozenset([]): 1.0, frozenset([v]): -1.0}

def _mul(p, q):
    out = defaultdict(float)
    for a, ca in p.items():
        for b, cb in q.items():
            out[frozenset(a | b)] += ca * cb
    return out

def rosenberg_penalty(P, lit_u, lit_v, w, scale):
    """Add scale*(uv - 2uw - 2vw + 3w); vanishes iff w = u*v (u,v literals, w var)."""
    U, V, W = _lit_poly(lit_u), _lit_poly(lit_v), {frozenset([w]): 1.0}
    expr = defaultdict(float)
    for t, c in _mul(U, V).items(): expr[t] += c
    for t, c in _mul(U, W).items(): expr[t] += -2 * c
    for t, c in _mul(V, W).items(): expr[t] += -2 * c
    expr[frozenset([w])] += 3.0
    for t, c in expr.items():
        add_term(P, t, scale * c)


# ---- chain construction (penalties only, no objective coupler) ------------- #
def build_chain(P, T, next_aux, lam):
    """Build the hybrid chain whose output qubit equals prod_{i in T} x_i on valid
    states. Adds chain penalties to P. Returns (output_qubit_index, next_aux)."""
    m = len(T)
    if m <= 5:                                   # domain-wall chain
        q = list(range(next_aux, next_aux + m)); next_aux += m
        out = q[m - 1]
        for k in range(m - 1):                   # ordering  sum q_{k+1}(1-q_k)
            add_term(P, [q[k + 1]], lam)
            add_term(P, [q[k], q[k + 1]], -lam)
        lin = {qi: 1.0 for qi in q}              # count  (sum q - sum_{i in T} x_i)^2
        for i in T:
            lin[i] = lin.get(i, 0.0) - 1.0
        square_of_linear(P, lin, 0.0, lam)
        return out, next_aux

    # binary-counter chain (m > 5)
    ell = math.ceil(math.log2(m + 1))
    r = list(range(next_aux, next_aux + ell)); next_aux += ell
    mbits = [(m >> j) & 1 for j in range(ell)]
    clit = [(r[j], 0 if mbits[j] == 1 else 1) for j in range(ell)]   # comparison literals
    a1 = next_aux; next_aux += 1
    rosenberg_penalty(P, clit[0], clit[1], a1, lam)                  # a1 = c0 c1
    prev = a1
    for k in range(2, ell - 1):
        ak = next_aux; next_aux += 1
        rosenberg_penalty(P, (prev, 0), clit[k], ak, lam)
        prev = ak
    qT = next_aux; next_aux += 1                                     # output q^T
    rosenberg_penalty(P, (prev, 0), clit[ell - 1], qT, lam)
    lin = {r[j]: float(2 ** j) for j in range(ell)}                  # (sum 2^j r_j - sum x)^2
    for i in T:
        lin[i] = lin.get(i, 0.0) - 1.0
    square_of_linear(P, lin, 0.0, lam)
    return qT, next_aux


def ferqbc_encode(hobo: HOBO, lam=None, share=True):
    """Encode a HOBO into a QUBO via FerQ-Bc. Returns (qubo: HOBO, n_aux).
    share=True groups monomials by a shared sub-aggregate T (Corollary 1(iv)).

    Penalty: by default a PER-CHAIN local penalty is used,
        lambda_T = sum_{S : T_S = T} |c_S| + 1  >  the couplers routed through chain T,
    which preserves the ground state (each violation of chain T costs >= lambda_T, more
    than any objective gain from its couplers) while keeping coefficients as small as
    possible. Pass a scalar `lam` to force a single global penalty instead."""
    P = defaultdict(float)
    for S, c in hobo.terms.items():
        if len(S) <= 2:
            add_term(P, S, c)
    high = [(tuple(sorted(S)), float(c)) for S, c in hobo.terms.items() if len(S) >= 3]

    # choose the sub-aggregate T_S for each monomial (drop one vertex v_S)
    if share:
        freq = Counter()
        for S, _ in high:
            d = len(S)
            for i in range(d):
                freq[tuple(S[j] for j in range(d) if j != i)] += 1
    chosen = []                                   # (S, c, T_tuple, v)
    for S, c in high:
        d = len(S)
        if share:                                 # pick the most-shared (d-1)-subset
            best = max(range(d), key=lambda i: freq[tuple(S[j] for j in range(d) if j != i)])
        else:
            best = 0                              # drop the smallest index
        T = tuple(S[j] for j in range(d) if j != best)
        chosen.append((S, c, T, S[best]))

    chain_sum = defaultdict(float)                # per-chain coefficient mass
    for S, c, T, v in chosen:
        chain_sum[T] += abs(c)

    nxt = hobo.n_vars
    chain_out = {}                                # T -> output qubit index
    for S, c, T, v in chosen:
        if T not in chain_out:
            lam_T = lam if lam is not None else (chain_sum[T] + 1.0)   # per-chain local penalty
            out, nxt = build_chain(P, list(T), nxt, lam_T)
            chain_out[T] = out
        add_term(P, [v, chain_out[T]], c)         # objective coupler c*x_v*q^T
    P = {k: v for k, v in P.items() if abs(v) > 1e-9}
    return HOBO(P, n_vars=nxt), nxt - hobo.n_vars


def verify_spectrum(hobo: HOBO, qubo: HOBO, n_aux: int, tol=1e-6):
    """min over ancillas of qubo(x,anc) must equal hobo(x) for every x."""
    n = hobo.n_vars
    for xb in itertools.product((0, 1), repeat=n):
        e_true = hobo.evaluate(dict(enumerate(xb)))
        best = math.inf
        for ab in itertools.product((0, 1), repeat=n_aux):
            best = min(best, qubo.evaluate(dict(enumerate(list(xb) + list(ab)))))
        if abs(best - e_true) > tol:
            return False, (xb, e_true, best)
    return True, None


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Verify FerQ-Bc encoder (min over ancillas == original energy):")
    # (1) Appendix Case A
    hA = HOBO({frozenset({0, 1, 2, 3, 4}): 3.0}, n_vars=5)
    qA, naA = ferqbc_encode(hA, lam=4.0)
    print("  Case A (deg-5 domain wall): n_aux=%d, verified=%s" % (naA, verify_spectrum(hA, qA, naA)[0]))
    # (2) single monomials, both regimes
    for d in range(3, 10):
        h = HOBO({frozenset(range(d)): -2.0}, n_vars=d)
        q, na = ferqbc_encode(h)
        ok = verify_spectrum(h, q, na)[0] if d <= 9 else "skip"
        print("  deg %d: n_aux=%d verified=%s" % (d, na, ok))
    # (3) sharing: structured multi-term, small
    core = {0, 1, 2}
    terms = {frozenset(core | {3}): 2.0, frozenset(core | {4}): -3.0, frozenset(core | {5}): 1.0}
    hs = HOBO(terms, n_vars=6)
    qs, nas = ferqbc_encode(hs, share=True)
    qn, nan = ferqbc_encode(hs, share=False)
    print("  shared structured (3 monomials, core {0,1,2}): share n_aux=%d (verified=%s), "
          "no-share n_aux=%d" % (nas, verify_spectrum(hs, qs, nas)[0], nan))
