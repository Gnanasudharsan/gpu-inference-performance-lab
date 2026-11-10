# Benchmark methodology

*Phase 0 draft. The full methodology is written in Phase 2 together with the harness that implements it.*

Principles that are already fixed:

1. Never time without synchronizing the device. Use device events for device time and host wall-clock
   time after a synchronize for end-to-end time.
2. Warm up before measuring (allocator, kernel caches, `torch.compile` compilation, Metal pipeline
   creation), and report compile time separately.
3. Report full latency distributions (p50, p90, p95, p99, min, max) with sample counts. Use
   bootstrap confidence intervals where samples are independent enough to justify them.
4. Record provenance (hardware, software, Git commit) inside every result file.
5. Goodput means work completed within an explicit deadline, not raw throughput.
