"""Common CEM-based model-predictive control.

One controller consumes every dynamics model. Nothing here may import
``residual_worlds.physics.target``: the exact-dynamics reference enters
as an opaque acceleration function supplied by the evaluation harness.
"""
