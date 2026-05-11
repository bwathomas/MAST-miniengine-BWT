"""DistServe-style decode-latency model fitter and cudagraph-bucket grid
proposer (Milestone 2 improvement item 8).

Source: DistServe Appendix A — the closed-form ``T_decode(B, L) = α + β·B``
decomposition. We fit ``α, β`` from observed (batch_size, decode_step_seconds)
pairs and emit a bucket ladder whose worst-case "bucket-padding waste"
(``(B_{k+1} - live - 1) / B_{k+1}``) stays below a given tolerance for any
live batch ``live ≤ B_max``.

Why this is useful for B-02
---------------------------
Our default cudagraph ladder ``[1, 2, 4, 8, 16, 32]`` doubles at every
step. A live batch of 9 rounds up to bucket 16, so ~44% of the GPU work
the captured graph does is on dummy rows. On L4, where the kernel's
batch-marginal cost is roughly half the per-step constant, that waste
turns directly into a measured throughput regression vs the plain
flash-attn paged decode path — which is exactly the B-02 symptom in
``empirical-baseline.md``.

A denser ladder (e.g. ``[1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]``)
trades capture-time GPU memory + a longer warmup against a much higher
average fill ratio at runtime. This module computes such a ladder once
from profile data; the engine then accepts the ladder via the existing
``--cuda-graph-batch-sizes`` flag.

Usage
-----
As a library::

    from miniengine.bucket_grid import BucketGridFitter
    fitter = BucketGridFitter.from_measurements(
        [(1, 0.011), (4, 0.012), (16, 0.018), (32, 0.025)]
    )
    ladder = fitter.propose_ladder(max_b=32, max_waste=0.10)

As a CLI::

    python -m miniengine.bucket_grid --max-b 32 --max-waste 0.10 \
        --measurements 1:0.011 4:0.012 16:0.018 32:0.025
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class BucketGridFitter:
    """Linear decode-time model: ``T(B) = alpha + beta * B``."""

    alpha: float
    beta: float
    r2: float

    @classmethod
    def from_measurements(
        cls, samples: list[tuple[int, float]]
    ) -> BucketGridFitter:
        """Least-squares fit of ``T = α + β·B`` on observed batch-size/time pairs.

        ``samples`` is a list of ``(batch_size, mean_decode_step_seconds)``
        observations. Need ≥ 2 points; 4+ recommended for an honest R².
        """
        if len(samples) < 2:
            raise ValueError("need ≥2 measurements to fit a 2-parameter model")
        n = len(samples)
        sx = sum(b for b, _ in samples)
        sy = sum(t for _, t in samples)
        sxx = sum(b * b for b, _ in samples)
        sxy = sum(b * t for b, t in samples)
        denom = n * sxx - sx * sx
        if denom == 0:
            raise ValueError("all measurements share the same batch size")
        beta = (n * sxy - sx * sy) / denom
        alpha = (sy - beta * sx) / n
        y_mean = sy / n
        ss_tot = sum((t - y_mean) ** 2 for _, t in samples)
        ss_res = sum((t - (alpha + beta * b)) ** 2 for b, t in samples)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return cls(alpha=alpha, beta=beta, r2=r2)

    def predict(self, batch_size: int) -> float:
        return self.alpha + self.beta * batch_size

    def waste_fraction(self, live: int, bucket: int) -> float:
        """Fraction of GPU work that goes to dummy rows when live → bucket."""
        if live > bucket:
            raise ValueError(f"live ({live}) > bucket ({bucket})")
        return max(0.0, (bucket - live) / bucket)

    def propose_ladder(
        self,
        max_b: int,
        max_waste: float = 0.10,
        min_b: int = 1,
    ) -> list[int]:
        """Pick a bucket sequence so worst-case padding waste stays ≤ ``max_waste``.

        Greedy construction: starting from ``min_b``, at each step pick the
        largest next bucket ``B'`` such that the worst-case live count just
        below it (``live = prev + 1``) still has waste ``≤ max_waste``::

            (B' - (prev + 1)) / B' ≤ max_waste  ⇔  B' ≤ (prev + 1) / (1 - max_waste)

        This is purely combinatorial — the latency model isn't actually used
        here, but the fit is logged so the caller can sanity-check ``β`` is
        positive (i.e. there IS a per-batch cost worth optimising for).
        """
        if not (0.0 <= max_waste < 1.0):
            raise ValueError(f"max_waste must be in [0, 1), got {max_waste}")
        if min_b < 1 or max_b < min_b:
            raise ValueError(f"need 1 ≤ min_b ({min_b}) ≤ max_b ({max_b})")
        ladder: list[int] = [min_b]
        while ladder[-1] < max_b:
            prev = ladder[-1]
            cap = int((prev + 1) / max(1.0 - max_waste, 1e-9))
            nxt = min(max(prev + 1, cap), max_b)
            if nxt == prev:
                nxt = prev + 1
            ladder.append(nxt)
        # Dedupe and clamp
        out: list[int] = []
        for b in ladder:
            if b > max_b:
                break
            if not out or out[-1] != b:
                out.append(b)
        if out[-1] != max_b:
            out.append(max_b)
        return out


def _parse_measurement(s: str) -> tuple[int, float]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"measurement '{s}' must be batch:seconds (e.g. 16:0.018)"
        )
    b, t = s.split(":", 1)
    return int(b), float(t)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m miniengine.bucket_grid",
        description=(
            "Fit a DistServe-style decode-time model from observed "
            "measurements and emit a cudagraph bucket ladder. The "
            "ladder is printed comma-separated on stdout, suitable "
            "for --cuda-graph-batch-sizes."
        ),
    )
    p.add_argument(
        "--measurements",
        type=_parse_measurement,
        nargs="+",
        required=True,
        help="One or more batch:seconds samples, e.g. 1:0.011 16:0.018 32:0.025",
    )
    p.add_argument("--max-b", type=int, required=True)
    p.add_argument("--max-waste", type=float, default=0.10)
    p.add_argument("--min-b", type=int, default=1)
    args = p.parse_args(argv)

    fitter = BucketGridFitter.from_measurements(args.measurements)
    ladder = fitter.propose_ladder(
        max_b=args.max_b, max_waste=args.max_waste, min_b=args.min_b
    )
    print(
        f"# decode-time fit: T(B) = {fitter.alpha:.6f} + {fitter.beta:.6f}*B   "
        f"R^2 = {fitter.r2:.3f}"
    )
    print(",".join(str(b) for b in ladder))


if __name__ == "__main__":
    main()
