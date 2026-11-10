| device | dtype | impl | rows | mode | host p50 µs | p95 | p99 | device p50 µs | rows/s (p50) | trial CV | speedup vs eager |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | float32 | compile | 1 | latency | 3,325 | 6,013 | 9,434 | n/a | 301 | 0.196 | 0.78× [0.57–0.97] |
| cpu | float32 | eager | 1 | latency | 2,383 | 4,587 | 5,719 | n/a | 420 | 0.188 | n/a |
| cpu | float32 | fused_weights | 1 | latency | 2,302 | 5,198 | 5,706 | n/a | 434 | 0.163 | 1.04× [0.58–1.41] |
| cpu | float32 | compile | 1 | throughput | 3,700 | 5,337 | 6,007 | n/a | 270 | 0.297 | 0.86× [0.47–1.69] |
| cpu | float32 | eager | 1 | throughput | 2,860 | 4,181 | 4,255 | n/a | 350 | 0.223 | n/a |
| cpu | float32 | fused_weights | 1 | throughput | 2,898 | 4,872 | 5,609 | n/a | 345 | 0.350 | 1.02× [0.48–1.58] |
| cpu | float32 | compile | 8 | latency | 4,864 | 15,344 | 25,048 | n/a | 1,645 | 0.563 | 0.63× [0.30–0.92] |
| cpu | float32 | eager | 8 | latency | 3,909 | 10,935 | 14,738 | n/a | 2,046 | 0.068 | n/a |
| cpu | float32 | fused_weights | 8 | latency | 4,610 | 12,303 | 15,338 | n/a | 1,735 | 0.236 | 0.79× [0.48–0.90] |
| cpu | float32 | compile | 8 | throughput | 9,017 | 14,522 | 14,779 | n/a | 887 | 0.505 | 0.70× [0.29–0.89] |
| cpu | float32 | eager | 8 | throughput | 4,257 | 12,266 | 12,371 | n/a | 1,879 | 0.543 | n/a |
| cpu | float32 | fused_weights | 8 | throughput | 5,715 | 11,373 | 12,863 | n/a | 1,400 | 0.417 | 0.89× [0.49–2.52] |
| cpu | float32 | compile | 32 | latency | 12,102 | 17,177 | 28,668 | n/a | 2,644 | 0.463 | 1.47× [0.41–3.39] |
| cpu | float32 | eager | 32 | latency | 13,576 | 19,860 | 52,815 | n/a | 2,357 | 0.234 | n/a |
| cpu | float32 | fused_weights | 32 | latency | 12,978 | 16,847 | 21,151 | n/a | 2,466 | 0.398 | 1.28× [0.37–2.95] |
| cpu | float32 | compile | 32 | throughput | 11,652 | 15,400 | 15,558 | n/a | 2,746 | 0.372 | 1.40× [0.43–3.79] |
| cpu | float32 | eager | 32 | throughput | 15,296 | 18,907 | 19,856 | n/a | 2,092 | 0.271 | n/a |
| cpu | float32 | fused_weights | 32 | throughput | 14,404 | 23,066 | 27,445 | n/a | 2,222 | 0.381 | 1.01× [0.28–3.48] |
| mps | bfloat16 | compile | 1 | latency | 1,553 | 3,126 | 11,111 | 1,329 | 644 | 0.350 | 1.15× [0.38–3.32] |
| mps | bfloat16 | eager | 1 | latency | 1,784 | 3,396 | 9,109 | 1,493 | 561 | 0.361 | n/a |
| mps | bfloat16 | fused_weights | 1 | latency | 1,447 | 3,683 | 9,048 | 1,317 | 691 | 0.499 | 1.21× [0.30–2.74] |
| mps | bfloat16 | metal | 1 | latency | 1,403 | 2,832 | 8,394 | 1,224 | 713 | 0.241 | 1.27× [0.55–3.25] |
| mps | bfloat16 | compile | 1 | throughput | 751 | 885 | 897 | n/a | 1,332 | 0.066 | 1.15× [0.93–1.40] |
| mps | bfloat16 | eager | 1 | throughput | 857 | 1,039 | 1,042 | n/a | 1,167 | 0.083 | n/a |
| mps | bfloat16 | fused_weights | 1 | throughput | 760 | 857 | 893 | n/a | 1,316 | 0.058 | 1.15× [0.95–1.38] |
| mps | bfloat16 | metal | 1 | throughput | 762 | 884 | 916 | n/a | 1,313 | 0.066 | 1.14× [0.92–1.40] |
| mps | bfloat16 | compile | 8 | latency | 2,611 | 5,388 | 12,506 | 2,972 | 3,064 | 0.360 | 0.86× [0.42–3.17] |
| mps | bfloat16 | eager | 8 | latency | 1,882 | 4,631 | 11,504 | 2,642 | 4,251 | 0.400 | n/a |
| mps | bfloat16 | fused_weights | 8 | latency | 2,268 | 4,591 | 11,643 | 2,401 | 3,527 | 0.414 | 0.93× [0.30–1.71] |
| mps | bfloat16 | metal | 8 | latency | 1,704 | 4,124 | 12,719 | 1,166 | 4,694 | 0.610 | 1.22× [0.52–3.70] |
| mps | bfloat16 | compile | 8 | throughput | 1,167 | 2,232 | 2,376 | n/a | 6,853 | 0.294 | 0.95× [0.45–1.53] |
| mps | bfloat16 | eager | 8 | throughput | 1,162 | 1,509 | 1,527 | n/a | 6,883 | 0.139 | n/a |
| mps | bfloat16 | fused_weights | 8 | throughput | 1,156 | 1,366 | 1,373 | n/a | 6,923 | 0.094 | 1.02× [0.84–1.42] |
| mps | bfloat16 | metal | 8 | throughput | 877 | 1,126 | 1,130 | n/a | 9,126 | 0.118 | 1.31× [0.94–1.87] |
| mps | bfloat16 | compile | 32 | latency | 2,787 | 5,895 | 15,330 | 3,121 | 11,484 | 0.420 | 1.02× [0.50–2.30] |
| mps | bfloat16 | eager | 32 | latency | 2,902 | 6,614 | 12,404 | 2,078 | 11,025 | 0.374 | n/a |
| mps | bfloat16 | fused_weights | 32 | latency | 3,433 | 12,843 | 17,658 | 2,352 | 9,323 | 0.364 | 0.91× [0.53–1.78] |
| mps | bfloat16 | metal | 32 | latency | 3,569 | 12,555 | 20,534 | 2,674 | 8,966 | 0.385 | 0.87× [0.38–2.82] |
| mps | bfloat16 | compile | 32 | throughput | 1,438 | 1,756 | 1,788 | n/a | 22,259 | 0.088 | 1.10× [0.87–1.28] |
| mps | bfloat16 | eager | 32 | throughput | 1,594 | 1,737 | 1,827 | n/a | 20,072 | 0.055 | n/a |
| mps | bfloat16 | fused_weights | 32 | throughput | 1,402 | 1,561 | 1,799 | n/a | 22,819 | 0.087 | 1.13× [0.86–1.34] |
| mps | bfloat16 | metal | 32 | throughput | 1,485 | 1,714 | 1,763 | n/a | 21,555 | 0.087 | 1.07× [0.90–1.35] |
| mps | float16 | compile | 1 | latency | 725 | 3,323 | 9,026 | 462 | 1,380 | 0.819 | 1.22× [0.25–2.40] |
| mps | float16 | eager | 1 | latency | 998 | 1,677 | 10,014 | 683 | 1,002 | 0.245 | n/a |
| mps | float16 | fused_weights | 1 | latency | 722 | 2,504 | 10,522 | 475 | 1,385 | 1.266 | 1.27× [0.14–2.37] |
| mps | float16 | metal | 1 | latency | 612 | 3,042 | 10,507 | 411 | 1,634 | 1.370 | 1.44× [0.14–2.94] |
| mps | float16 | compile | 1 | throughput | 311 | 501 | 546 | n/a | 3,211 | 0.253 | 1.36× [0.60–2.51] |
| mps | float16 | eager | 1 | throughput | 417 | 695 | 741 | n/a | 2,397 | 0.240 | n/a |
| mps | float16 | fused_weights | 1 | throughput | 282 | 360 | 401 | n/a | 3,548 | 0.138 | 1.57× [1.02–2.26] |
| mps | float16 | metal | 1 | throughput | 257 | 344 | 436 | n/a | 3,897 | 0.196 | 1.66× [1.13–2.83] |
| mps | float16 | compile | 8 | latency | 1,486 | 1,984 | 8,038 | 1,481 | 5,384 | 0.238 | 1.11× [0.46–2.92] |
| mps | float16 | eager | 8 | latency | 1,592 | 2,117 | 7,477 | 1,106 | 5,025 | 0.480 | n/a |
| mps | float16 | fused_weights | 8 | latency | 1,338 | 7,009 | 9,689 | 1,308 | 5,979 | 0.765 | 1.01× [0.30–3.27] |
| mps | float16 | metal | 8 | latency | 1,185 | 1,905 | 6,422 | 1,034 | 6,751 | 0.140 | 1.27× [0.61–3.84] |
| mps | float16 | compile | 8 | throughput | 511 | 955 | 1,091 | n/a | 15,667 | 0.353 | 0.99× [0.40–1.82] |
| mps | float16 | eager | 8 | throughput | 550 | 763 | 802 | n/a | 14,550 | 0.215 | n/a |
| mps | float16 | fused_weights | 8 | throughput | 409 | 632 | 658 | n/a | 19,559 | 0.198 | 1.27× [0.82–1.76] |
| mps | float16 | metal | 8 | throughput | 317 | 540 | 552 | n/a | 25,223 | 0.264 | 1.62× [0.96–2.73] |
| mps | float16 | compile | 1 | latency | 1,460 | 6,643 | 10,536 | 1,096 | 685 | 0.790 | 0.98× [0.26–2.39] |
| mps | float16 | eager | 1 | latency | 1,516 | 4,486 | 10,487 | 1,589 | 659 | 0.343 | n/a |
| mps | float16 | fused_weights | 1 | latency | 1,409 | 2,877 | 7,781 | 1,250 | 710 | 0.145 | 1.18× [0.77–2.40] |
| mps | float16 | metal | 1 | latency | 1,325 | 5,788 | 8,488 | 1,214 | 755 | 0.645 | 1.05× [0.31–2.81] |
| mps | float16 | compile | 1 | throughput | 778 | 928 | 1,037 | n/a | 1,286 | 0.101 | 1.14× [0.82–1.77] |
| mps | float16 | eager | 1 | throughput | 879 | 1,016 | 1,304 | n/a | 1,138 | 0.130 | n/a |
| mps | float16 | fused_weights | 1 | throughput | 761 | 799 | 803 | n/a | 1,315 | 0.029 | 1.19× [1.06–1.79] |
| mps | float16 | metal | 1 | throughput | 778 | 841 | 888 | n/a | 1,286 | 0.048 | 1.16× [0.95–1.80] |
| mps | float16 | compile | 2 | latency | 2,008 | 5,080 | 7,325 | 1,314 | 996 | 0.354 | 0.93× [0.51–2.13] |
| mps | float16 | eager | 2 | latency | 1,991 | 3,490 | 6,036 | 1,954 | 1,005 | 0.322 | n/a |
| mps | float16 | fused_weights | 2 | latency | 1,848 | 4,304 | 10,554 | 1,232 | 1,082 | 0.426 | 1.00× [0.47–2.87] |
| mps | float16 | metal | 2 | latency | 1,417 | 2,823 | 5,258 | 906 | 1,412 | 0.141 | 1.33× [0.85–2.97] |
| mps | float16 | compile | 2 | throughput | 948 | 1,195 | 1,232 | n/a | 2,110 | 0.101 | 0.96× [0.75–1.27] |
| mps | float16 | eager | 2 | throughput | 927 | 1,059 | 1,276 | n/a | 2,158 | 0.100 | n/a |
| mps | float16 | fused_weights | 2 | throughput | 921 | 1,230 | 1,559 | n/a | 2,173 | 0.178 | 0.98× [0.77–1.16] |
| mps | float16 | metal | 2 | throughput | 780 | 877 | 948 | n/a | 2,564 | 0.065 | 1.21× [0.94–1.66] |
| mps | float16 | compile | 4 | latency | 2,059 | 4,012 | 9,684 | 1,583 | 1,943 | 0.233 | 1.03× [0.54–1.72] |
| mps | float16 | eager | 4 | latency | 2,160 | 3,519 | 8,854 | 1,596 | 1,852 | 0.183 | n/a |
| mps | float16 | fused_weights | 4 | latency | 1,912 | 5,222 | 10,067 | 1,453 | 2,092 | 0.469 | 1.00× [0.21–1.46] |
| mps | float16 | metal | 4 | latency | 1,609 | 5,142 | 8,919 | 874 | 2,486 | 0.455 | 1.17× [0.48–1.95] |
| mps | float16 | compile | 4 | throughput | 1,003 | 1,383 | 1,996 | n/a | 3,988 | 0.247 | 0.96× [0.46–1.26] |
| mps | float16 | eager | 4 | throughput | 989 | 1,160 | 1,226 | n/a | 4,043 | 0.070 | n/a |
| mps | float16 | fused_weights | 4 | throughput | 951 | 1,081 | 1,085 | n/a | 4,206 | 0.050 | 1.04× [0.91–1.30] |
| mps | float16 | metal | 4 | throughput | 795 | 851 | 860 | n/a | 5,029 | 0.033 | 1.26× [1.14–1.52] |
| mps | float16 | compile | 8 | latency | 3,216 | 4,932 | 10,663 | 1,507 | 2,487 | 0.310 | 1.04× [0.39–3.96] |
| mps | float16 | eager | 8 | latency | 2,586 | 8,988 | 11,876 | 3,086 | 3,093 | 0.559 | n/a |
| mps | float16 | fused_weights | 8 | latency | 2,528 | 5,832 | 11,013 | 1,219 | 3,164 | 0.584 | 1.10× [0.19–4.34] |
| mps | float16 | metal | 8 | latency | 1,804 | 4,083 | 10,006 | 1,057 | 4,434 | 0.263 | 1.36× [0.65–4.14] |
| mps | float16 | compile | 8 | throughput | 1,051 | 1,100 | 1,119 | n/a | 7,612 | 0.042 | 1.02× [0.89–1.34] |
| mps | float16 | eager | 8 | throughput | 1,053 | 1,284 | 1,380 | n/a | 7,600 | 0.099 | n/a |
| mps | float16 | fused_weights | 8 | throughput | 978 | 1,102 | 1,111 | n/a | 8,184 | 0.055 | 1.07× [0.93–1.31] |
| mps | float16 | metal | 8 | throughput | 845 | 1,132 | 1,222 | n/a | 9,468 | 0.132 | 1.20× [0.90–1.56] |
| mps | float16 | compile | 16 | latency | 3,212 | 7,529 | 11,695 | 2,295 | 4,982 | 0.338 | 0.88× [0.44–1.87] |
| mps | float16 | eager | 16 | latency | 2,492 | 5,907 | 11,021 | 1,903 | 6,420 | 0.322 | n/a |
| mps | float16 | fused_weights | 16 | latency | 3,599 | 7,703 | 11,443 | 1,657 | 4,445 | 0.393 | 0.84× [0.25–1.93] |
| mps | float16 | metal | 16 | latency | 2,855 | 5,207 | 10,104 | 3,465 | 5,604 | 0.360 | 0.93× [0.31–1.89] |
| mps | float16 | compile | 16 | throughput | 1,181 | 1,477 | 1,492 | n/a | 13,550 | 0.084 | 1.02× [0.82–1.21] |
| mps | float16 | eager | 16 | throughput | 1,220 | 1,405 | 1,413 | n/a | 13,114 | 0.057 | n/a |
| mps | float16 | fused_weights | 16 | throughput | 1,136 | 1,303 | 1,337 | n/a | 14,087 | 0.058 | 1.08× [0.99–1.23] |
| mps | float16 | metal | 16 | throughput | 1,204 | 1,582 | 1,804 | n/a | 13,284 | 0.138 | 1.00× [0.67–1.22] |
| mps | float16 | compile | 32 | latency | 2,844 | 10,276 | 12,096 | 1,914 | 11,251 | 0.543 | 1.10× [0.49–5.02] |
| mps | float16 | eager | 32 | latency | 3,163 | 10,216 | 11,785 | 3,317 | 10,117 | 0.534 | n/a |
| mps | float16 | fused_weights | 32 | latency | 3,375 | 11,028 | 15,207 | 2,276 | 9,482 | 0.568 | 1.06× [0.41–2.43] |
| mps | float16 | metal | 32 | latency | 2,743 | 9,748 | 12,240 | 4,597 | 11,667 | 0.388 | 1.21× [0.53–5.37] |
| mps | float16 | compile | 32 | throughput | 1,346 | 1,611 | 1,677 | n/a | 23,769 | 0.072 | 1.08× [0.86–1.24] |
| mps | float16 | eager | 32 | throughput | 1,475 | 1,598 | 1,665 | n/a | 21,690 | 0.043 | n/a |
| mps | float16 | fused_weights | 32 | throughput | 1,344 | 1,533 | 1,609 | n/a | 23,809 | 0.065 | 1.08× [0.93–1.26] |
| mps | float16 | metal | 32 | throughput | 1,344 | 1,613 | 1,802 | n/a | 23,817 | 0.093 | 1.07× [0.90–1.18] |
| mps | float16 | compile | 64 | latency | 3,973 | 10,018 | 13,315 | 4,088 | 16,107 | 0.481 | 1.01× [0.37–4.82] |
| mps | float16 | eager | 64 | latency | 3,844 | 9,754 | 11,241 | 2,511 | 16,650 | 0.411 | n/a |
| mps | float16 | fused_weights | 64 | latency | 3,705 | 8,269 | 10,763 | 2,265 | 17,276 | 0.389 | 1.08× [0.34–2.43] |
| mps | float16 | metal | 64 | latency | 3,694 | 10,485 | 18,360 | 3,881 | 17,326 | 0.483 | 0.95× [0.39–4.37] |
| mps | float16 | compile | 64 | throughput | 1,839 | 2,082 | 2,104 | n/a | 34,797 | 0.055 | 1.13× [0.96–1.38] |
| mps | float16 | eager | 64 | throughput | 2,048 | 2,301 | 2,440 | n/a | 31,249 | 0.057 | n/a |
| mps | float16 | fused_weights | 64 | throughput | 1,791 | 2,106 | 2,178 | n/a | 35,727 | 0.061 | 1.14× [0.92–1.39] |
| mps | float16 | metal | 64 | throughput | 1,881 | 2,120 | 2,233 | n/a | 34,021 | 0.064 | 1.10× [0.98–1.33] |
| mps | float16 | compile | 1 | latency | 3,321 | 6,963 | 10,994 | 2,809 | 301 | 0.266 | 1.07× [0.47–3.13] |
| mps | float16 | eager | 1 | latency | 3,455 | 9,148 | 11,448 | 3,057 | 289 | 0.483 | n/a |
| mps | float16 | fused_weights | 1 | latency | 3,424 | 6,328 | 8,934 | 2,865 | 292 | 0.204 | 1.08× [0.53–3.03] |
| mps | float16 | metal | 1 | latency | 3,300 | 5,294 | 11,460 | 2,880 | 303 | 0.098 | 1.13× [0.92–3.12] |
| mps | float16 | compile | 1 | throughput | 2,866 | 2,963 | 2,985 | n/a | 349 | 0.024 | 1.04× [0.98–1.18] |
| mps | float16 | eager | 1 | throughput | 2,973 | 3,162 | 3,224 | n/a | 336 | 0.037 | n/a |
| mps | float16 | fused_weights | 1 | throughput | 2,893 | 3,030 | 3,054 | n/a | 346 | 0.028 | 1.03× [0.94–1.10] |
| mps | float16 | metal | 1 | throughput | 2,937 | 3,156 | 3,311 | n/a | 340 | 0.047 | 1.01× [0.94–1.10] |
| mps | float16 | compile | 8 | latency | 3,432 | 8,584 | 10,181 | 3,071 | 2,331 | 0.248 | 1.06× [0.59–3.06] |
| mps | float16 | eager | 8 | latency | 3,848 | 8,888 | 10,353 | 3,169 | 2,079 | 0.352 | n/a |
| mps | float16 | fused_weights | 8 | latency | 3,666 | 7,358 | 9,420 | 3,100 | 2,183 | 0.238 | 1.06× [0.55–2.46] |
| mps | float16 | metal | 8 | latency | 3,201 | 8,213 | 12,335 | 2,930 | 2,500 | 0.255 | 1.18× [0.60–2.98] |
| mps | float16 | compile | 8 | throughput | 3,608 | 4,262 | 4,467 | n/a | 2,217 | 0.105 | 1.03× [0.88–1.22] |
| mps | float16 | eager | 8 | throughput | 3,718 | 4,199 | 4,470 | n/a | 2,152 | 0.096 | n/a |
| mps | float16 | fused_weights | 8 | throughput | 3,390 | 3,894 | 3,900 | n/a | 2,360 | 0.076 | 1.08× [0.83–1.28] |
| mps | float16 | metal | 8 | throughput | 3,023 | 3,474 | 3,618 | n/a | 2,646 | 0.062 | 1.20× [1.03–1.36] |
| mps | float32 | compile | 1 | latency | 2,364 | 4,279 | 10,222 | 1,817 | 423 | 0.107 | 1.06× [0.79–2.07] |
| mps | float32 | eager | 1 | latency | 2,311 | 4,395 | 8,696 | 2,329 | 433 | 0.314 | n/a |
| mps | float32 | fused_weights | 1 | latency | 2,251 | 3,523 | 5,858 | 1,579 | 444 | 0.132 | 1.11× [0.84–2.04] |
| mps | float32 | metal | 1 | latency | 2,102 | 3,536 | 7,645 | 1,593 | 476 | 0.163 | 1.15× [0.73–2.55] |
| mps | float32 | compile | 1 | throughput | 1,442 | 1,610 | 1,669 | n/a | 693 | 0.056 | 1.03× [0.89–1.14] |
| mps | float32 | eager | 1 | throughput | 1,507 | 1,615 | 1,628 | n/a | 664 | 0.033 | n/a |
| mps | float32 | fused_weights | 1 | throughput | 1,448 | 1,613 | 1,834 | n/a | 691 | 0.072 | 1.03× [0.78–1.14] |
| mps | float32 | metal | 1 | throughput | 1,463 | 1,564 | 1,602 | n/a | 684 | 0.032 | 1.03× [0.97–1.11] |
| mps | float32 | compile | 8 | latency | 3,913 | 10,598 | 11,883 | 4,190 | 2,044 | 0.456 | 0.81× [0.32–2.48] |
| mps | float32 | eager | 8 | latency | 3,611 | 7,840 | 10,442 | 3,925 | 2,215 | 0.375 | n/a |
| mps | float32 | fused_weights | 8 | latency | 3,583 | 10,005 | 11,278 | 3,919 | 2,233 | 0.378 | 1.01× [0.49–1.98] |
| mps | float32 | metal | 8 | latency | 2,642 | 9,684 | 10,308 | 1,725 | 3,028 | 0.582 | 1.20× [0.40–2.79] |
| mps | float32 | compile | 8 | throughput | 2,237 | 3,229 | 3,349 | n/a | 3,575 | 0.156 | 1.08× [0.68–1.57] |
| mps | float32 | eager | 8 | throughput | 2,490 | 3,803 | 4,025 | n/a | 3,212 | 0.199 | n/a |
| mps | float32 | fused_weights | 8 | throughput | 2,315 | 3,175 | 4,511 | n/a | 3,456 | 0.249 | 1.07× [0.82–1.27] |
| mps | float32 | metal | 8 | throughput | 1,814 | 2,502 | 2,851 | n/a | 4,410 | 0.165 | 1.36× [0.92–1.79] |
| mps | float32 | compile | 32 | latency | 3,733 | 11,551 | 17,433 | 2,422 | 8,572 | 0.452 | 0.87× [0.33–1.65] |
| mps | float32 | eager | 32 | latency | 3,362 | 8,741 | 11,828 | 4,338 | 9,518 | 0.281 | n/a |
| mps | float32 | fused_weights | 32 | latency | 3,394 | 8,883 | 14,786 | 3,081 | 9,429 | 0.386 | 0.98× [0.55–1.66] |
| mps | float32 | metal | 32 | latency | 3,967 | 11,718 | 17,784 | 3,408 | 8,067 | 0.407 | 0.81× [0.37–1.27] |
| mps | float32 | compile | 32 | throughput | 2,197 | 2,674 | 2,934 | n/a | 14,567 | 0.114 | 1.09× [0.79–1.76] |
| mps | float32 | eager | 32 | throughput | 2,418 | 2,952 | 3,460 | n/a | 13,235 | 0.143 | n/a |
| mps | float32 | fused_weights | 32 | throughput | 2,235 | 2,669 | 2,693 | n/a | 14,321 | 0.104 | 1.10× [0.88–1.61] |
| mps | float32 | metal | 32 | throughput | 2,170 | 2,660 | 2,721 | n/a | 14,745 | 0.105 | 1.09× [0.82–1.74] |
