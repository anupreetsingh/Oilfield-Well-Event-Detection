| model | macro-F1 (pooled, held-out wells) | macro-F1 per fold | detection rate | median lead (min) | alerted before steady state | false alerts / normal day |
|---|---|---|---|---|---|---|
| rf | 0.213 | 0.266 ± 0.051 | 0.13 | 1076.1 | 0.12 | 0.54 |
| hgb | 0.279 | 0.357 ± 0.099 | 0.29 | 216.0 | 0.16 | 10.71 |
| cnn | 0.143 | 0.227 ± 0.042 | 0.26 | 232.7 | 0.19 | 27.35 |
| ensemble | 0.218 | 0.317 ± 0.093 | 0.25 | 182.2 | 0.16 | 9.63 |

Window-level F1 by class (support = held-out windows):

| class | support | rf | hgb | cnn | ensemble |
|---|---|---|---|---|---|
| NORMAL | 196129 | 0.65 | 0.68 | 0.53 | 0.64 |
| ABRUPT_INCREASE_OF_BSW | 1211 | 0.00 | 0.00 | 0.00 | 0.00 |
| SPURIOUS_CLOSURE_OF_DHSV | 1845 | 0.35 | 0.28 | 0.07 | 0.16 |
| SEVERE_SLUGGING | 9496 | 0.00 | 0.00 | 0.03 | 0.00 |
| FLOW_INSTABILITY | 34135 | 0.20 | 0.39 | 0.25 | 0.37 |
| RAPID_PRODUCTIVITY_LOSS | 5558 | 0.00 | 0.00 | 0.00 | 0.00 |
| QUICK_RESTRICTION_IN_PCK | 141 | 0.00 | 0.00 | 0.00 | 0.00 |
| SCALING_IN_PCK | 105274 | 0.00 | 0.37 | 0.01 | 0.25 |
| HYDRATE | 66811 | 0.72 | 0.79 | 0.40 | 0.54 |

Alerts by fault class for `hgb` (held-out runs; lead = minutes before steady state, delay = minutes after the labelled event start):

| class | runs | detected | median lead | median delay |
|---|---|---|---|---|
| ABRUPT_INCREASE_OF_BSW | 4 | 0.00 | - | - |
| SPURIOUS_CLOSURE_OF_DHSV | 22 | 0.95 | 86.2 | 5.2 |
| SEVERE_SLUGGING | 32 | 0.00 | - | - |
| FLOW_INSTABILITY | 303 | 0.26 | - | 3.0 |
| RAPID_PRODUCTIVITY_LOSS | 11 | 0.00 | - | - |
| QUICK_RESTRICTION_IN_PCK | 5 | 0.00 | - | - |
| SCALING_IN_PCK | 35 | 0.40 | - | 62.3 |
| HYDRATE | 71 | 0.35 | 1964.7 | 4.8 |
