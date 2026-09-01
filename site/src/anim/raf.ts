// Motion helpers. The site has no animation library; components that
// animate do so with requestAnimationFrame and must consult motionOK()
// so a reduced-motion preference renders final states with manual
// stepping instead of autoplay. A missing matchMedia (test
// environments) counts as reduced motion, so the static path is the
// one every test exercises.

export function motionOK(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }
  return !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
