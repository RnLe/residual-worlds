# Residual Worlds site

One page: the premise, what I expect, where the study stands, and a live
preview of what the nominal model imagines against what the true world
does. The preview is computed in the browser from a TypeScript mirror of
the study's physics (`src/arm/`), pinned to the Python implementation by
`tests/fixtures/arm_golden.json`. Nothing on the page is a result.

From the repository root, one command builds the site and serves it:

```
pnpm go
```

Or work in this directory:

```
pnpm install
pnpm check      # type-check and unit tests
pnpm dev        # local dev server
pnpm build      # production build into dist/
pnpm preview    # serve the built site
```

`public/preview.webp` and `public/preview.gif` are the no-JavaScript
fallback and the card image; regenerate them (and the golden numbers)
from the repository root after any change to the arm, the world, or the
schedule:

```
uv run residual-worlds fixture-arm-golden --config configs/experiment_contract.yaml
uv run residual-worlds render-preview --config configs/experiment_contract.yaml
```
