#import "../lib.typ": *

= Data protocol

*Budgets.* A budget counts valid transitions from the target world, never
wall-clock interaction: aborted segments, resets, and discarded non-finite
endpoints are logged separately so that data efficiency never hides
collection cost. Budgets are nested --- every smaller budget is a prefix
subset of the larger ones under a balanced unit ordering --- so budget
comparisons are within-data, not across independent collections. The
confirmatory cell uses world #raw(results.primary.world_id) at a budget of
#results.primary.budget transitions.

*Collection.* The unit of collection is a fixed number of valid
transitions assigned to one excitation component: band-limited random
torque, phase-randomized multisine, or nominal-MPC task rollouts with a
small held perturbation. Command sequences are generated up front from the
unit's seed and never resampled in response to the trajectory.

*Splits.* Train/validation allocation operates on whole collection units,
never on single transitions: adjacent timesteps of one trajectory are
dependent, and letting them straddle the split would make validation
optimistic.

*Pairing.* At each budget, every learned condition trains from the
identical transition rows, and every condition is evaluated on the same
scenario families with paired planner noise. The primary cell crosses
#results.primary.pipelines pipeline replicates with
#results.primary.scenarios scenario families, and all contrasts are paired
at the pipeline-by-scenario level.
