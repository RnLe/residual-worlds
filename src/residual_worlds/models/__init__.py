"""Planner-facing dynamics models: nominal, fitted, black-box, residual.

Nothing in this package may import ``residual_worlds.physics.target``;
the exact-dynamics reference is assembled by the evaluation harness,
which passes an opaque acceleration function into the same interfaces.
"""
