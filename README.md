# Residual Worlds

**Keep the physics. Learn the mismatch.**

A simulation study of residual dynamics models for data-efficient
model-predictive control. No physical robot, no perception, no
sim-to-real claim. In progress; no results yet.

## Premise

A torque-controlled planar two-link arm must visit three targets in order
while avoiding an obstacle. Its rigid-body model is known but wrong in the
details: the hidden target world adds a payload, nonlinear friction, and
an actuator that does not deliver the torque it is told to. A planner that
imagines with the wrong model plans for a world that does not exist.

> Under a fixed, scarce budget of target-world transitions, should a
> controller keep the analytical model, re-identify its parameters,
> replace it with a neural network, or keep the physics and learn only
> the acceleration residual?

| Condition | Adaptation data | Structure |
|---|---|---|
| nominal physics | none | unchanged analytical two-link model |
| fitted physics | shared transitions | five bounded physical parameters (payload, damping, gains) |
| black box | shared transitions | MLP ensemble predicting the complete joint acceleration |
| residual | shared transitions | nominal acceleration plus a capacity-matched MLP correction |
| exact-dynamics reference | none (privileged) | true target dynamics under the same approximate planner |

The adapted methods receive identical data, matched capacity, the same
optimizer schedule, and paired noise streams; one frozen CEM controller
consumes every condition. Closed-loop task success is the primary
endpoint: a planner exploits exactly the regions where a model is
optimistic, and one-step error says little about that.

## Expected results

- The residual model wins at small budgets: gravity, inertia, and coupling
  are already in the equations, so the network learns only what is left.
  The advantage shrinks with data until the black box catches up.
- Fitted physics is competitive where the mismatch is parametric (the
  payload) and falls behind where it is not (Stribeck friction, a dead zone).
- The primary contrast (residual versus black box, composite world, 2,048
  transitions, eight pipelines by 24 scenario families) and its claim rule
  (crossed-bootstrap 95% interval above zero and at least +0.10 success
  probability) are fixed before evaluation. A null or opposite result is
  reported with the same prominence.

None of the ingredients are new. The contribution is the strictness of
the comparison.

## Status

- Built: simulator verified against analytic, energy, and reference
  integrator checks; target worlds; the five conditions; matched training;
  planner; evaluation; crossed bootstrap; figures and tables; a CPU smoke
  run through the whole chain.
- Open: task calibration. Under the draft task parameters even the
  exact-dynamics reference fails (see `docs/protocol_notes.md`); the task
  is being fixed before any learned model is trained on it.
- No protected results exist.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/); a GPU is needed only for
the final batched experiment.

```bash
uv sync --frozen --extra cpu --group dev     # CPU stack (CUDA: --extra cu130)
uv run residual-worlds doctor                # environment report
uv run pytest -m "not slow and not gpu" -q   # fast verification suite
uv run residual-worlds verify-simulator --config configs/experiment_contract.yaml
uv run residual-worlds smoke --config configs/smoke.yaml   # full chain on a miniature config
```

The site needs Node 22 and pnpm. One command builds it and serves it
locally, from the repository root:

```bash
pnpm go
```

## Layout

```
configs/            experiment contract (draft, with its decision ledger), smoke config
src/residual_worlds/
  physics/          kinematics, nominal dynamics, target worlds, RK4, verification   PyTorch, batched
  task/             collision geometry, task automaton, scenario generator          PyTorch
  data/             excitation signals, collection units, splits, datasets          PyTorch, NumPy
  models/           nominal / fitted / black-box / residual / ensemble               torch.nn, SciPy L-BFGS-B
  training/         shared losses, matched training runs                            PyTorch
  planning/         costs, imagined rollouts, CEM, receding-horizon controller     PyTorch
  evaluation/       manifest expansion, closed-loop runner, prediction metrics     PyTorch
  analysis/         crossed bootstrap, aggregation, figures, tables                 NumPy, Matplotlib
  release/          public result bundle: schema, builder, verifier, fixture
  reporting/        Typst report staging and build
  media/            the animated preview and its golden numbers
site/               single page with a live TypeScript mirror of the preview physics
report/             Typst report sources
tests/              unit / scientific / integration suites
docs/               protocol working notes
```

Every artifact is an immutable, checksummed directory; all randomness
flows from one root seed through named namespaces. Learned-model code
cannot import the target-world module; a static test enforces that.

## Limitations

Fully observed, synthetic, low-dimensional; designed mismatch, not natural
shift. The learned residual is an acceleration correction over the visited
distribution, not a recovered physical force. Ensemble spread is
disagreement, not calibrated uncertainty. The exact-dynamics reference
shares the approximate planner and is no performance ceiling.

## License

MIT; see `LICENSE`. Citation metadata in `CITATION.cff`.
