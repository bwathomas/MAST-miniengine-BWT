"""
Request scheduler — the core orchestrator of the serving engine.

The scheduler sits between the HTTP server and the model engine:

    Server  ──add_request()──▶  Scheduler  ──prefill/decode──▶  Engine
      ▲                            │
      └─── token_queue (stream) ◄──┘

It runs in a background thread, repeatedly calling step() which:
  1. Admits waiting requests and prefills them  (WAITING → RUNNING)
  2. Runs one decode step on every running request
  3. Retires finished requests                    (RUNNING → FINISHED)

"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from miniengine.core import Request, RequestStatus, TokenOutput
from miniengine.engine import Engine

logger = logging.getLogger(__name__)


class Scheduler:
    """
    FCFS scheduler with three modes:

      baseline : process one request to completion before the next.
      batched  : iteration-level batching — admit + prefill many requests,
                 then advance all running requests by one token in a
                 single batched forward pass.  New requests can join the
                 batch the same step they finish prefill.
      paged    : iteration-level batching backed by the engine's KV pool
                 — admission allocates the request's worst-case page
                 budget up front (prompt + max_new_tokens). Pool capacity
                 is the only admission constraint; requests that don't
                 fit right now wait, requests that can never fit are
                 deterministically rejected.

    Public API (thread-safe):
        add_request(req)   — enqueue a new request
        start()            — launch the background scheduling loop
        stop()             — gracefully shut down
    """

    def __init__(self, engine: Engine, max_running: int = 16, mode: str = "batched"):
        self.engine = engine
        self.max_running = max_running
        self.mode = mode

        # Queues
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        # Thread control
        self._lock = threading.Lock()
        self._running_flag = False
        self._thread: threading.Thread | None = None

        # Stats
        self.total_finished: int = 0
        self.total_generated_tokens: int = 0

    # ── Public API (thread-safe) ────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """Enqueue a request for scheduling."""
        with self._lock:
            self.waiting.append(request)
            logger.info(
                "Enqueued request %s  (prompt_len=%d, waiting=%d)",
                request.request_id,
                request.num_input_tokens,
                len(self.waiting),
            )

    def start(self) -> None:
        """Start the scheduler loop in a background daemon thread."""
        self._running_flag = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread to join."""
        self._running_flag = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped")

    # ── Main loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running_flag:
            has_work = bool(self.waiting) or bool(self.running)
            if not has_work:
                time.sleep(0.005)  # idle sleep to avoid busy-waiting
                continue
            try:
                self.step()
            except Exception:
                logger.exception("Scheduler step failed")

    # ── Scheduling step ─────────────────────────────────────────────────

    def step(self) -> list[Request]:
        """
        One scheduling iteration.  Behaviour depends on self.mode.

        Returns list of requests that finished in this step.
        """
        if self.mode == "baseline":
            return self._step_baseline()
        if self.mode == "paged":
            return self._step_paged()
        return self._step_batched()

    def _step_baseline(self) -> list[Request]:
        """One request to completion per step. Maximally naive."""
        finished: list[Request] = []

        with self._lock:
            if not self.waiting:
                return finished
            req = self.waiting.popleft()

        req.status = RequestStatus.RUNNING
        token_id = self.engine.prefill(req)
        req.output_ids.append(token_id)
        self._stream_token(req, token_id)

        while not self._check_finished(req, token_id):
            token_id = self.engine.decode_step(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

        self._finish_request(req, finished)
        return finished

    def _step_paged(self) -> list[Request]:
        """
        Paged iteration-level step.

        Admission asks the engine's KV pool for the request's worst-case
        page budget (prompt + max_new_tokens). Three outcomes:
          - budget exceeds the entire pool → reject deterministically.
          - budget fits but free pages are insufficient right now → leave
            the request at the head of the waiting queue (we admit FCFS
            so a single oversized request can't starve smaller ones
            indefinitely once others retire).
          - budget fits and is available → allocate, prefill, run.
        Decode is one batched forward pass through the paged engine path.
        Finished requests release their pages back to the pool.
        """
        assert self.engine.pool is not None, "paged mode requires an engine KV pool"
        pool = self.engine.pool
        finished: list[Request] = []

        # Packed-prefill token budget (DistServe §3.1 ``L_m`` / Sarathi-Serve
        # §4.3 ``τ``). 0 = no cap (default). Capping the total number of
        # prompt tokens admitted in a single packed prefill bounds the
        # prefill-step latency, which on L4 dominates TTFT spikes when a
        # large bursts admits 8+ long prompts in one step.
        prefill_budget = int(getattr(self.engine, "prefill_token_budget", 0))

        with self._lock:
            to_prefill: list[Request] = []
            packed_tokens = 0
            while (
                self.waiting and len(self.running) + len(to_prefill) < self.max_running
            ):
                req = self.waiting[0]
                budget = req.num_input_tokens + req.sampling_params.max_new_tokens
                needed = pool.pages_needed(budget)
                if needed > pool.num_pages:
                    self.waiting.popleft()
                    self._reject_request(
                        req,
                        f"request budget ({budget} tokens, {needed} pages) exceeds "
                        f"pool capacity ({pool.num_pages} pages × {pool.page_size} "
                        f"tokens)",
                        finished,
                    )
                    continue
                if needed > pool.num_free:
                    break
                if (
                    prefill_budget > 0
                    and to_prefill
                    and packed_tokens + req.num_input_tokens > prefill_budget
                ):
                    # Defer to next step. Keeps the packed prefill kernel's
                    # work-per-step below ``L_m``; never starves a single
                    # oversized request because the ``to_prefill`` list is
                    # non-empty only after the first admission succeeded.
                    break
                self.waiting.popleft()
                req.kv_cache = pool.allocate(needed)
                to_prefill.append(req)
                packed_tokens += req.num_input_tokens

        if to_prefill:
            for req in to_prefill:
                req.status = RequestStatus.RUNNING
            token_ids = self.engine.batched_prefill(to_prefill)
            for req, token_id in zip(to_prefill, token_ids):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    self.running.append(req)

        if self.running:
            token_ids = self.engine.batched_decode(self.running)
            still_running: list[Request] = []
            for req, token_id in zip(self.running, token_ids):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    still_running.append(req)
            self.running = still_running

        return finished

    def _step_batched(self) -> list[Request]:
        """
        Iteration-level batched step:
          Phase 1 — admit waiting requests and prefill them (per-request).
          Phase 2 — batched decode: one token for every running request.
        Newly prefilled requests join the decode batch in the same step.
        """
        finished: list[Request] = []

        # ── Phase 1: admit + prefill ────────────────────────────────────
        with self._lock:
            to_prefill: list[Request] = []
            while (
                self.waiting and len(self.running) + len(to_prefill) < self.max_running
            ):
                to_prefill.append(self.waiting.popleft())

        for req in to_prefill:
            req.status = RequestStatus.RUNNING
            token_id = self.engine.prefill(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)
            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                self.running.append(req)

        # ── Phase 2: batched decode ─────────────────────────────────────
        if self.running:
            token_ids = self.engine.batched_decode(self.running)
            still_running: list[Request] = []
            for req, token_id in zip(self.running, token_ids):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    still_running.append(req)
            self.running = still_running

        return finished

    # ── Helpers ─────────────────────────────────────────────────────────

    def _check_finished(self, req: Request, token_id: int) -> bool:
        """Decide whether a request should stop generating."""
        if req.is_finished:
            return True
        if self.engine.is_stop_token(token_id):
            return True
        return False

    def _stream_token(self, req: Request, token_id: int) -> None:
        """Push a generated token into the request's streaming queue."""
        text = self.engine.decode_token(token_id)
        req.token_queue.put(
            TokenOutput(token_id=token_id, token_text=text, finished=False)
        )

    def _finish_request(self, req: Request, finished_list: list[Request]) -> None:
        """Mark a request as finished and free its resources."""
        req.status = RequestStatus.FINISHED
        self._release_kv(req)
        req.token_queue.put(TokenOutput(token_id=-1, token_text="", finished=True))
        finished_list.append(req)

        self.total_finished += 1
        self.total_generated_tokens += req.num_output_tokens
        logger.info(
            "Finished request %s  (output_len=%d, running=%d, waiting=%d)",
            req.request_id,
            req.num_output_tokens,
            len(self.running),
            len(self.waiting),
        )

    def _reject_request(
        self, req: Request, reason: str, finished_list: list[Request]
    ) -> None:
        """Deterministically reject an admission-time impossible request."""
        logger.warning(
            "Rejecting request %s — %s",
            req.request_id,
            reason,
        )
        req.status = RequestStatus.FINISHED
        self._release_kv(req)
        req.token_queue.put(TokenOutput(token_id=-1, token_text="", finished=True))
        finished_list.append(req)
        self.total_finished += 1

    def _release_kv(self, req: Request) -> None:
        """Return KV resources held by the request to the pool / GC."""
        if (
            self.mode == "paged"
            and self.engine.pool is not None
            and isinstance(req.kv_cache, list)
            and req.kv_cache
            and isinstance(req.kv_cache[0], int)
        ):
            self.engine.pool.free(req.kv_cache)
        req.kv_cache = None
