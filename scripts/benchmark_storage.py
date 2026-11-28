"""Synthetic Qdrant Local capacity measurements, excluded from timing-sensitive CI."""

import argparse
import concurrent.futures
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models


def benchmark(size, samples):
    rng = np.random.default_rng(1729)
    vectors = rng.normal(size=(size, 1536)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    with tempfile.TemporaryDirectory() as root:
        client = QdrantClient(path=root)
        client.create_collection(
            "bench",
            vectors_config=models.VectorParams(
                size=1536, distance=models.Distance.COSINE
            ),
        )
        started = time.perf_counter()
        for offset in range(0, size, 100):
            client.upsert(
                "bench",
                [
                    models.PointStruct(
                        id=i, vector=vectors[i].tolist(), payload={"group": i % 10}
                    )
                    for i in range(offset, min(offset + 100, size))
                ],
            )
        ingestion = time.perf_counter() - started
        client.close()
        started = time.perf_counter()
        client = QdrantClient(path=root)
        cold = time.perf_counter() - started
        assert client.count("bench", exact=True).count == size

        def query(index, filtered=False):
            condition = (
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="group", match=models.MatchValue(value=index % 10)
                        )
                    ]
                )
                if filtered
                else None
            )
            start = time.perf_counter()
            result = client.query_points(
                "bench", query=vectors[index].tolist(), limit=6, query_filter=condition
            ).points
            assert result[0].id == index and abs(result[0].score - 1) < 1e-5
            return (time.perf_counter() - start) * 1000

        for i in range(5):
            query(i)
        latency = [query(i % size) for i in range(samples)]
        filtered = [query(i % size, True) for i in range(samples)]
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            concurrent_times = list(pool.map(query, range(samples)))
        storage = sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Memory(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t)
                    for name in (
                        "peak",
                        "working",
                        "quota_peak",
                        "quota",
                        "page_peak",
                        "page",
                        "pagefile",
                        "peak_pagefile",
                    )
                ]

            memory = Memory()
            memory.cb = ctypes.sizeof(memory)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.c_void_p(-1),
                ctypes.byref(memory),
                memory.cb,
            )
            rss = memory.working
        else:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        client.close()
        return {
            "points": size,
            "dimensions": 1536,
            "samples": samples,
            "cold_seconds": cold,
            "points_per_second": size / ingestion,
            "query_ms": np.percentile(latency, [50, 95, 99]).tolist(),
            "filtered_ms": np.percentile(filtered, [50, 95, 99]).tolist(),
            "concurrent_p95_ms": float(np.percentile(concurrent_times, 95)),
            "rss_bytes": rss,
            "storage_bytes": storage,
            "correctness": "self nearest, score and restart count passed",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "machine": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "results": [],
    }
    for size in args.sizes:
        report["results"].append(benchmark(size, args.samples))
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(size, "completed", flush=True)
