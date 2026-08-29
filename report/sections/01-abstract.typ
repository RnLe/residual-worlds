#import "../lib.typ": *

= Abstract

We compare five dynamics-model conditions for model-predictive control of a
simulated two-link arm whose true dynamics deviate from the modelled
equations through controlled, hidden mechanisms: the unchanged nominal
physics, a bounded fitted-physics recalibration, a black-box neural
ensemble, a physics-residual ensemble (nominal physics plus a learned
acceleration correction), and an exact-dynamics reference that shares the
approximate planner. All learned conditions receive identical transition
data at fixed budgets, and every condition runs under one frozen CEM-MPC
controller, so observed differences are attributable to the dynamics model
alone. The preregistered primary endpoint is the paired closed-loop
success-rate difference between the residual and black-box conditions at
one primary world and data budget, with uncertainty from a crossed
stratified bootstrap and a practical-relevance threshold fixed in advance.
#interpretation-sentence
