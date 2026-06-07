# Warm-Start / Continuation Research: XFOIL + RL stepping

Date: 2026-06-06

## Short answer

Yes, this is a good idea and aligns with how XFOIL is already designed to be used. The key condition is that it is only safe when the new state is close to the previous state; otherwise force a cold re-initialization.

- XFOIL is an interactive, stateful solver with active data kept in memory during a run, including multiple stored polars and airfoils; it is not a purely stateless one-shot executable call pattern.[XFOIL User Guide, Data structure](https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt)
- In XFOIL viscous analysis, the Newton solver explicitly uses the last available solution as the starting guess, and convergence is better when alpha changes are small between runs.[XFOIL Caveats](https://v0xnihili.github.io/xfoil-docs/caveats/)
- This is exactly the continuum of numerical continuation: solve a nearby parameterized problem using the previous solution as initial guess, reducing work per step in nonlinear solvers.[Numerical continuation (COMSOL](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_solver.36.119.html), [Continuation methods overview](https://en.wikipedia.org/wiki/Numerical_continuation))

## What XFOIL already exposes that supports this

1. **Stateful execution model**
   - XFOIL stores data in RAM and supports current/buffer airfoils plus stored polar sets in one session, which is exactly the memory pattern you need for warm-start behavior.[XFOIL User Guide](https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt)

2. **Explicit continuation command path for alpha/Lift sweeps**
   - `ASEQ` is a built-in consecutive-angle run command in `OPER`; this indicates normal intended workflow is sequential solves over nearby points rather than independent isolated points.[OPER command list](https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt)
   - `ITER` controls max Newton iterations, and failed cases can be driven with additional `ALFA`/`CL` calls to perform extra Newton steps using the current state.[XFOIL Caveats](https://v0xnihili.github.io/xfoil-docs/caveats/)

3. **Solver restart controls**
   - `INIT` exists for boundary-layer reinitialization when large jumps are unavoidable or when the viscous solution has blown up. This is a hard control for avoiding false warm-start transfer.[XFOIL Caveats](https://v0xnihili.github.io/xfoil-docs/caveats/)

4. **Why this exists in XFOIL’s design**
   - XFOIL’s own caveats recommend small-step sequencing in alpha because the Newton method “works best if the change from the old to the new solutions is reasonably small,” and recommends `ASEQ` for difficult cases.[XFOIL Caveats](https://v0xnihili.github.io/xfoil-docs/caveats/)

## Expected benefits

- Higher effective throughput on smooth update sequences (e.g., sequential AOA moves for one airfoil shape) because the nonlinear solve starts from a nearby state instead of an arbitrary guess.[XFOIL Caveats](https://v0xnihili.github.io/xfoil-docs/caveats/)
- Better RL signal quality than random cold restarts, because many tiny geometric steps often remain on the same physical branch and remain numerically stable when continuation is respected.[Numerical continuation methods](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_solver.36.119.html)

## Limits and risk zones where warm-start should be disabled

- XFOIL warns about fragile behavior for large geometry/thick separation/resonant situations and suggests caution for aggressive steps; large jumps are explicitly flagged as requiring re-init and can still fail.[XFOIL Caveats](https://v0xnihili.github.io/xfoil-docs/caveats/)
- Transonic/supersonic-like flow is outside XFOIL’s strongest operating area and should be treated as low-confidence for continuation-based warm starts.[XFOIL User Guide](https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt)
- Near/stalled viscous regimes, the same caveats and accuracy section flag increased numerical risk; this is where you should tighten tolerances and be ready to fallback to conservative cold solves.[XFOIL Numerical Accuracy](https://v0xnihili.github.io/xfoil-docs/numerical-accuracy/), [XFOIL Caveats](https://v0xnihili.github.io/xfoil-docs/caveats/)

## Recommended RL-env warm-start policy for your project

### 1) Keep two solver tiers

- **Fast tier (warm path)**
  - Use warm start when geometry movement is small.
  - Reuse solver state from immediately preceding step in the same env instance.
  - Use reduced iteration budgets for first-pass convergence.
- **Safe tier (fallback)**
  - Force `INIT` / cold state when geometry delta is large or when previous step flagged unstable.
  - Increase `ITER` before retrying, then if still not converged, mark hard-fail and penalize deterministically.

### 2) Define a deterministic reuse gate

- Reuse only if all are true:
  - Re/`Mach` unchanged and turbulence/transition settings unchanged.
  - Geometry distance <= threshold (for example `L2` norm of control points).
  - Previous run converged and residual trend was monotonic.
  - No mode switch (AoA-parameterized vs CL-parameterized) changed.
- Otherwise cold-start and optionally re-panelize geometry for a clean initial state.

### 3) Session-level state cache

- Cache per env instance:
  - current geometry hash + control parameters + operation mode
  - last converged `(CL,CD,CM)` and any solver state proxy you can access
  - converged iterations used, residual trend, failure reasons
- Invalidate cache on any hard-fail or geometry topology change that breaks topology assumptions (excessively small/degenerate panels, TE discontinuity).

### 4) Deterministic telemetry

- Log decision (`warm`/`cold`/`retry`), distance metrics, and gate reasons in info.
- This makes the policy reproducible and allows post-hoc evaluation of success rate by regime.[AGENTS](/home/claude/xfoil/AGENTS.md)

## Practical performance expectation in your stack

- You should not expect 10x speedup from warm-start alone if you rebuild everything purely from scratch each step; the largest gains are from batch orchestration and Oracle throughput, not just one-step continuation state reuse.
- But continuation reuse can materially reduce per-step iterations in “easy regions” and is very low-hanging optimization value when actions are incremental because XFOIL already exposes this path.
- This is strongest when paired with your existing plan to keep a compiled XFOIL oracle and use Warp mostly for parallel pre/post orchestration, not for full solver replacement.

## References

- XFOIL User Primer (official): https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt
- XFOIL Caveats: https://v0xnihili.github.io/xfoil-docs/caveats/
- Numerical accuracy (XFOIL companion): https://v0xnihili.github.io/xfoil-docs/numerical-accuracy/
- COMSOL Parametric continuation and reuse of previous solution: https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_solver.36.119.html
- COMSOL Common Study Step settings (initial values from prior solution context): https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_solver.36.011.html
- Numerical continuation overview: https://en.wikipedia.org/wiki/Numerical_continuation
