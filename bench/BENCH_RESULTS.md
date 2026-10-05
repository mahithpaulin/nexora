# Bench results (v2, 2026-10-05)

Generated: `python bench/run_bench.py` (floors enforced) and `python bench/scaling.py --save bench/scaling_results.json`. All datasets seeded; scores deterministic.

```
dataset           | metric              | score
------------------+---------------------+------
periodic          | discovery_precision | 1.000
periodic          | discovery_recall    | 1.000
periodic          | discovery_f1        | 1.000
noise             | discovery_f1        | 1.000
planted_anomalies | anomaly_f1          | 1.000
periodic          | predict_accuracy    | 1.000
periodic          | baseline_accuracy   | 0.320
note: periodic discovery: 3 frequent-sequence bigrams found
note: noise discovery: 0 spurious frequent-sequence bigrams (expected none)
note: anomaly tolerance +-1: tp=2 fp=0 fn=0
note: prediction note: order-1 engine (Markov/context predict_next, acc 1.000) vs order-0 baseline (always train mode 'A', acc 0.320)
PASS: all 7 metrics >= floors.
```

```
     n    discover(s)  find_anomalies(s)    predict(s)     stream(s)
--------------------------------------------------------------------
   200         0.1002             0.0021        0.0057        0.0006
   500         0.2641             0.0051        0.0138        0.0014
  1000         0.4199             0.0100        0.0277        0.0029
  2000         1.1679             0.0207        0.0553        0.0059
  5000         3.0485             0.0604        0.1466        0.0159
--------------------------------------------------------------------
 slope          1.063              1.034         1.007         1.037
(log-log least-squares slope of time vs n; cap {'discover': 1.6, 'find_anomalies': 1.6, 'predict': 1.6, 'stream': 1.5})
slope[discover] = 1.063 (max 1.6)
slope[find_anomalies] = 1.034 (max 1.6)
slope[predict] = 1.007 (max 1.6)
slope[stream] = 1.037 (max 1.5)
```
