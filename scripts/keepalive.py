"""Hold a live CUDA context so the RTX 5090 never idles into RTD3.

The open kernel module can fail to wake the card from auto-suspend
during idle gaps; a trivial matmul every couple of seconds keeps it
busy. Launch alongside any long GPU job and keep it running for the
whole job. See the orion-server project memory.
"""

import sys
import time

import torch


# ponytail: fixed tiny matmul; the only job is to keep the context warm.
def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    a = torch.randn(512, 512, device="cuda")
    ticks = 0
    while True:
        try:
            (a @ a).sum().item()
        except Exception as exc:  # noqa: BLE001
            # Never die silently: a dead keepalive lets the GPU idle and
            # the card can fall off the bus on the next gap.
            print(f"keepalive error: {exc}", file=sys.stderr, flush=True)
        ticks += 1
        if ticks % 300 == 0:
            print(f"keepalive alive, tick {ticks}", flush=True)
        time.sleep(2.0)


if __name__ == "__main__":
    main()
