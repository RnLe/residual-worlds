#import "../lib.typ": *

= Motivation and contribution

Learning a correction on top of an imperfect physical model is an old and
repeatedly rediscovered idea, and this study does not present it as a new
one. Residual physics has carried real-robot manipulation
@zeng2019tossingbot; structured decompositions of a physical model plus a
neural component have been formalized with explicit well-posedness
arguments @yin2021aphynity; and the same pattern appears in general form as
universal differential equations @rackauckas2020universal. Ensembles of
learned dynamics models driving sampling-based control are likewise
standard practice #cite(<chua2018pets>) #cite(<nagabandi2018neural>). None
of the five model classes evaluated here is our invention, and this report
claims no architectural novelty.

The contribution is narrower and, in our reading of the literature,
undersupplied: a controlled, preregistered comparison in which the model
class is the only manipulated factor. All learned conditions consume
identical transition rows at each budget; the residual and black-box
networks are capacity-matched twins differing in exactly one structural
bias; every condition plans through one frozen CEM-MPC controller with
paired primitive noise; and scenarios are shared across methods so that
every contrast is paired. Because model accuracy and control utility are
known to decouple @lambert2020objective, the primary endpoint is
closed-loop task success rather than prediction error; prediction quality
is reported separately as a diagnostic.

The study is a simulation study by design: target worlds are constructed
mismatch mechanisms with known ground truth, which is what makes an
exact-dynamics reference and controlled pairing possible at all. The
corresponding scope limits are stated in @sec-limitations.
