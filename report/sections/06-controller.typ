#import "../lib.typ": *

= Controller

One frozen receding-horizon controller serves every condition; only the
acceleration function differs. The planner is the cross-entropy method
@rubinstein1999cem over latent action-knot sequences: a Gaussian is
maintained in unconstrained latent space and candidates pass through
$u = u_max tanh(z)$, so torque bounds hold by construction rather than by
clipping inside the optimizer.

Deterministic numerical conventions are part of the scientific contract,
because every method must face bit-identical optimizer behavior given the
same primitive noise: all planner tensors are float32; candidate ranking
is a stable ascending sort of (invalid flag, total cost, candidate index)
with non-finite costs ordered last; the elite update uses a fixed
retention factor and a floored standard deviation; and after the final
iteration the latent mean itself is evaluated as one additional
deterministic candidate.

Warm starting lives entirely in latent space: the previous knot mean is
expanded to the horizon, shifted by the executed actions, its tail
repeated, and compressed back to knots; the latent standard deviation
resets at every call so only within-call elite updates shrink it.
Primitive noise is drawn per planning call from a method-free seed
namespace, so paired methods demonstrably receive identical randomness
while their updated sampling distributions remain free to diverge.

For ensemble conditions, each candidate is rolled through every member and
ranked by the arithmetic mean of member costs; the final plan is valid
only if every member's rollout is valid.
