#import "../lib.typ": *

= Dynamics

== Nominal model

The nominal plant is a planar two-link arm under gravity with viscous joint
damping, actuated by joint torques $u in RR^2$ and described by the
manipulator equation

$ M_0 (q) dot.double(q) + c_0 (q, dot(q)) + g_0 (q) + B_0 dot(q) = u, $

with joint angles $q = (q_1, q_2)$ measured from the positive horizontal
axis. With link lengths $l_i$, center-of-mass offsets $l_(c i)$, masses
$m_i$, rotational inertias $I_i$, and the shorthand
$beta = m_2 l_1 l_(c 2)$, $h(q_2) = -beta sin q_2$, the nominal terms are

$ M_0 (q) = mat(
  I_1 + I_2 + m_1 l_(c 1)^2 + m_2 (l_1^2 + l_(c 2)^2 + 2 l_1 l_(c 2) cos q_2),
  I_2 + m_2 (l_(c 2)^2 + l_1 l_(c 2) cos q_2);
  I_2 + m_2 (l_(c 2)^2 + l_1 l_(c 2) cos q_2),
  I_2 + m_2 l_(c 2)^2
), $

$ c_0 (q, dot(q)) = vec(
  h(q_2) (2 dot(q)_1 dot(q)_2 + dot(q)_2^2),
  -h(q_2) dot(q)_1^2
), quad
g_0 (q) = nabla_q V(q), $

with gravitational potential

$ V(q) = g [ (m_1 l_(c 1) + m_2 l_1) sin q_1 + m_2 l_(c 2) sin(q_1 + q_2) ], $

and $B_0$ a diagonal viscous damping matrix. Accelerations are obtained by
solving the $2 times 2$ linear system, never by forming an explicit
inverse; a scalar float64 reference implementation of the same equations
serves as an independent cross-check of the batched code.

== Target-world mechanisms

Every target world obeys one canonical equation,

$ M_w (q) dot.double(q) + c_w (q, dot(q)) + g_w (q) + tau_"fric" (dot(q))
  = tau_"act" (u) + tau_"el" (q), $

in which each component reduces exactly to its nominal counterpart when
disabled:

- *payload*: a point mass $m_p$ at the end effector contributes
  $M_p = m_p J_e^top J_e$ together with its Coriolis and gravity terms
  ($J_e$ the end-effector Jacobian);
- *nonlinear friction*: a smooth Stribeck-like law replaces the nominal
  viscous term entirely --- it carries its own viscous coefficient, so
  damping is never double-counted;
- *actuator*: commanded torque passes through a gain and a dead zone,
  then a componentwise clip to the physical torque limit;
- *elastic coupling*: the conservative torque $-nabla_q V_c$ of
  $V_c (q) = k_c (1 - cos(q_1 - q_2))$, a synthetic joint coupling held
  out as an unmodelled mechanism.

A magnitude scale interpolates each mechanism between the nominal world
and its full strength so that zero scale recovers the nominal dynamics
exactly. The world parameters stay hidden from every learned condition;
models see transitions $(x, u, x')$ only.
