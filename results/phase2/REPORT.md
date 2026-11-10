| device | dtype | impl | rows | mode | host p50 µs | p95 | p99 | device p50 µs | rows/s (p50) | trial CV | speedup vs eager |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | float32 | eager | 1 | latency | 3,312 | 7,834 | 10,242 | n/a | 302 | 0.067 | n/a |
| cpu | float32 | eager | 1 | throughput | 2,506 | 2,786 | 2,857 | n/a | 399 | 0.061 | n/a |
| cpu | float32 | eager | 8 | latency | 9,327 | 23,226 | 27,736 | n/a | 858 | 0.379 | n/a |
| cpu | float32 | eager | 8 | throughput | 9,978 | 12,577 | 12,805 | n/a | 802 | 0.316 | n/a |
| mps | float16 | eager | 1 | e2e | 1,741 | 6,366 | 15,788 | n/a | 575 | 0.283 | n/a |
| mps | float16 | eager | 1 | latency | 1,826 | 6,626 | 11,016 | 1,696 | 548 | 0.121 | n/a |
| mps | float16 | eager | 1 | throughput | 913 | 1,207 | 1,262 | n/a | 1,095 | 0.158 | n/a |
| mps | float16 | eager | 8 | e2e | 3,070 | 7,699 | 18,272 | n/a | 2,606 | 0.216 | n/a |
| mps | float16 | eager | 8 | latency | 2,265 | 12,028 | 21,476 | 2,601 | 3,532 | 0.255 | n/a |
| mps | float16 | eager | 8 | latency | 2,656 | 10,118 | 12,303 | 1,212 | 3,013 | 0.216 | n/a |
| mps | float16 | eager | 8 | latency | 3,316 | 10,480 | 13,802 | 2,650 | 2,412 | 0.025 | n/a |
| mps | float16 | eager | 8 | latency | 2,948 | 6,417 | 10,185 | 2,698 | 2,713 | 0.182 | n/a |
| mps | float16 | eager | 8 | latency | 2,462 | 8,900 | 13,654 | 2,288 | 3,249 | 0.092 | n/a |
| mps | float16 | eager | 8 | throughput | 1,221 | 1,334 | 1,353 | n/a | 6,551 | 0.054 | n/a |
| mps | float16 | eager | 32 | e2e | 4,066 | 21,647 | 34,687 | n/a | 7,870 | 0.175 | n/a |
| mps | float16 | eager | 32 | latency | 2,803 | 10,776 | 14,419 | 2,022 | 11,417 | 0.393 | n/a |
| mps | float16 | eager | 32 | throughput | 1,894 | 2,323 | 2,404 | n/a | 16,897 | 0.121 | n/a |
