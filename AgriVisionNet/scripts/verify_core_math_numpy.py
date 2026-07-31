"""
scripts/verify_core_math_numpy.py

Standalone, torch-free re-implementation of the numerical claims in
models/evidential.py, so they can be ACTUALLY EXECUTED in an environment
without torch/GPU access (this authoring sandbox), rather than merely
asserted. This is not a substitute for tests/test_evidential.py (which
exercises the real nn.Module forward pass and must be run with
`pytest tests/` once torch is installed) — it validates the underlying
math in isolation so the design is checked BEFORE depending on a full
PyTorch install.

Run: python3 scripts/verify_core_math_numpy.py
"""

import numpy as np

rng = np.random.default_rng(42)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def evidential_forward(logits, num_classes):
    evidence = softplus(logits)
    alpha = evidence + 1.0
    strength = alpha.sum(axis=-1)
    belief = evidence / strength[..., None]
    prob = alpha / strength[..., None]
    uncertainty = num_classes / strength
    return dict(evidence=evidence, alpha=alpha, strength=strength,
                belief=belief, prob=prob, uncertainty=uncertainty)


def ds_discount(belief, uncertainty, discount_rate):
    r = discount_rate[..., None]
    discounted_belief = (1.0 - r) * belief
    discounted_uncertainty = discount_rate + (1.0 - discount_rate) * uncertainty
    return discounted_belief, discounted_uncertainty


def kl_annealing_coefficient(epoch, annealing_steps):
    if annealing_steps <= 0:
        return 1.0
    return float(min(1.0, epoch / annealing_steps))


results = []


def check(name, condition):
    results.append((name, bool(condition)))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")


print("=" * 70)
print("1. EvidentialHead math (models/evidential.py:EvidentialHead)")
print("=" * 70)

logits = rng.normal(size=(64, 10)) * 5  # wide range incl. large +/- values
out = evidential_forward(logits, num_classes=10)

check("evidence >= 0 for all samples/classes", np.all(out["evidence"] >= 0))
check("alpha >= 1 for all samples/classes", np.all(out["alpha"] >= 1.0))
check("uncertainty in (0, 1] for all samples", np.all((out["uncertainty"] > 0) & (out["uncertainty"] <= 1.0)))
check(
    "belief + uncertainty == 1 (valid mass function), max abs err < 1e-8",
    np.max(np.abs(out["belief"].sum(axis=-1) + out["uncertainty"] - 1.0)) < 1e-8,
)

# zero-evidence edge case: force logits very negative -> evidence ~ 0 -> u -> 1
zero_ev_logits = np.full((4, 6), -1000.0)
zero_out = evidential_forward(zero_ev_logits, num_classes=6)
check(
    "zero evidence -> uncertainty ~= 1.0 (untrained network defaults to 'unknown')",
    np.allclose(zero_out["uncertainty"], 1.0, atol=1e-6),
)

# high-evidence edge case: one class gets huge evidence -> u -> 0, belief concentrates
# NOTE: for K=6, u = K/S, so u < 1e-3 requires S > 6000 -> evidence0 > ~5994.
# An earlier version of this check used evidence0=500 (S~506, u~0.0119) and
# failed -- that was a test-scale bug (threshold too strict for the evidence
# magnitude used), not a bug in evidential_forward/evidential.py. Fixed by
# using an evidence magnitude consistent with the asserted threshold.
concentrated_logits = np.full((4, 6), -50.0)
concentrated_logits[:, 0] = 20000.0
conc_out = evidential_forward(concentrated_logits, num_classes=6)
check(
    "concentrated evidence on one class -> uncertainty near 0",
    np.all(conc_out["uncertainty"] < 1e-3),
)
check(
    "concentrated evidence -> belief for that class near 1",
    np.all(conc_out["belief"][:, 0] > 0.99),
)

print()
print("=" * 70)
print("2. Dempster-Shafer discounting (models/evidential.py:ds_discount)")
print("=" * 70)

K, B = 5, 200
raw_belief = rng.uniform(0, 1, size=(B, K))
raw_belief = raw_belief / (raw_belief.sum(axis=-1, keepdims=True) + rng.uniform(0.1, 2.0, size=(B, 1)))
raw_uncertainty = 1.0 - raw_belief.sum(axis=-1)
check(
    "constructed test beliefs are themselves valid (sanity check on the test harness)",
    np.all(raw_uncertainty > 0) and np.max(np.abs(raw_belief.sum(-1) + raw_uncertainty - 1.0)) < 1e-8,
)

r0 = np.zeros(B)
b0, u0 = ds_discount(raw_belief, raw_uncertainty, r0)
check("r=0 recovers identity (belief)", np.allclose(b0, raw_belief, atol=1e-10))
check("r=0 recovers identity (uncertainty)", np.allclose(u0, raw_uncertainty, atol=1e-10))

r1 = np.ones(B)
b1, u1 = ds_discount(raw_belief, raw_uncertainty, r1)
check("r=1 drives belief to exactly 0 (total ignorance)", np.allclose(b1, 0.0, atol=1e-10))
check("r=1 drives uncertainty to exactly 1 (total ignorance)", np.allclose(u1, 1.0, atol=1e-10))

all_mass_ok = True
for r_val in np.linspace(0, 1, 11):
    r = np.full(B, r_val)
    b, u = ds_discount(raw_belief, raw_uncertainty, r)
    mass = b.sum(axis=-1) + u
    if np.max(np.abs(mass - 1.0)) > 1e-8:
        all_mass_ok = False
check("mass budget (belief+uncertainty=1) preserved for r in [0,1] (11 points)", all_mass_ok)

# Monotonicity: discounted uncertainty must be non-decreasing in r, for fixed raw belief/u
mono_ok = True
r_grid = np.linspace(0, 1, 50)
for i in range(len(raw_uncertainty)):
    u_vals = [ds_discount(raw_belief[i:i+1], raw_uncertainty[i:i+1], np.array([r]))[1][0] for r in r_grid]
    if not np.all(np.diff(u_vals) >= -1e-12):
        mono_ok = False
        break
check("discounted uncertainty is monotonically non-decreasing in discount rate r", mono_ok)

print()
print("=" * 70)
print("3. KL annealing coefficient (models/evidential.py:kl_annealing_coefficient)")
print("=" * 70)

vals = [kl_annealing_coefficient(e, annealing_steps=10) for e in range(0, 25)]
check("annealing coefficient stays within [0, 1] for all epochs tested", all(0.0 <= v <= 1.0 for v in vals))
check("annealing coefficient is monotonically non-decreasing", vals == sorted(vals))
check("annealing coefficient saturates at exactly 1.0 after annealing_steps", vals[-1] == 1.0 and vals[10] == 1.0)
check("annealing_steps=0 returns 1.0 immediately (no ramp)", kl_annealing_coefficient(0, 0) == 1.0)

print()
print("=" * 70)
n_pass = sum(1 for _, ok in results if ok)
n_total = len(results)
print(f"SUMMARY: {n_pass}/{n_total} checks passed")
print("=" * 70)

if n_pass != n_total:
    raise SystemExit(1)
