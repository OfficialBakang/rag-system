# Windows synthetic benchmark results

Windows 11 build 26200, Python 3.12.10, 8 logical CPUs, Intel family 6 model 142 stepping 12. CIM access to the model name and physical RAM was denied.

| Points | Reopen seconds | Points/second | Query p50/p95/p99 ms | Filter p50/p95/p99 ms | 3-thread p95 ms | RSS MiB | Disk MiB |
|---|---|---|---|---|---|---|---|
| 1000 | 0.32 | 157.7 | 11.6/14.0/14.6 | 29.7/35.7/38.5 | 26.1 | 159.0 | 13.7 |
| 5000 | 2.46 | 144.4 | 86.4/123.2/136.3 | 184.3/318.5/374.6 | 133.2 | 420.4 | 68.5 |
| 10000 | 3.39 | 143.8 | 107.4/146.7/160.0 | 258.7/346.9/446.3 | 289.8 | 725.6 | 137.1 |

Seed 1729, 1,536 dimensions, 100 measured requests per mode after five warmups. Every query checked self-nearest retrieval and cosine score. Counts survived close/reopen. The declared prototype limits were 1 second p95, 3 GiB RSS and 30 seconds reopen at 10,000 points. All were met.

The benchmark calls Qdrant directly. Concurrent numbers exclude the application wrapper's serialisation and HTTP admission behavior. Reopen timing uses a warm OS cache within the same process. Resident memory includes the generated dataset and allocations retained from earlier sizes. No controlled Chroma throughput comparison or 20,000-point run was performed. These measurements support further prototype review and do not establish production capacity.
