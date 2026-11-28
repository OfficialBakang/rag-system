# Qdrant Local compatibility prototype

Status is implemented for disposable data, pending container and release verification.

Qdrant Local removes the affected Chroma runtime while preserving single-process
persistent storage and offline smoke testing. PostgreSQL would introduce an additional
service and extension lifecycle. Qdrant Server is the next option when measured local
capacity is insufficient. Local mode loads vectors into memory and recommends a server
for larger collections, including those above 20,000 points.

## Migration and rollback

Freeze writes and stop the legacy application before taking a complete backup of its
database and indexes. Hash the backup and verify restoration into a disposable directory.
Keep the original volume intact. The exporter requires a disposable copy and runs in an
isolated legacy environment with network blocked and provider credentials removed.

Run export_chroma.py with --disposable-copy, --output and --collection. It writes batches
of data-only JSONL records and a count, dimension and SHA-256 manifest. The importer runs
in the new runtime using python -m scripts.migration.import_bundle --bundle PATH --target
NEW_PATH. It validates the bundle before opening storage. A migration marker rejects
unrelated targets. Retrying replays deterministic upserts. The current importer loads
the bundle into memory, so large exports require a streaming-validation extension before
production approval.

Preserve the original export vectors because cosine storage normalises them. Verify
complete IDs, document counts, text and metadata checksums, dimensions, source lists,
filtered scores within 1e-5, restart persistence and deletion on a disposable clone.
Treat equal-score ordering separately. No embedding regeneration is required.

Keep writes frozen through cutover verification. Rollback before new writes restores
the prior image and original volume mapping. After new writes, reconcile updates and
deletions before rollback. Running the vulnerable legacy image requires a separate
security decision. The prototype performs no live cutover.

## Benchmarks and release gates

scripts/benchmark_storage.py uses seed 1729, 1,536-dimensional float32 vectors,
1,000/5,000/10,000 records, five warmups and 100 samples per query mode. It records
p50/p95/p99, filtered queries, three-thread calls, startup, ingestion, resident memory
and disk size. Correctness checks self retrieval, score and restart counts.
Provisional targets declared before measuring were p95 below 1 second, resident memory
below 3 GiB and startup below 30 seconds at 10,000 records. These are local prototype
targets. Dataset generation contributes to resident memory. Performance is not a CI
assertion. Separate-process cold OS-cache and large production workloads need further
measurement.

Retain all security gates, the 85% statement floor and role-based HTTP controls.
Linux runtime audits, exact-image Trivy and Docker recreation are mandatory before
release. Windows benchmark results are provided with the implementation report.

## Verified prototype results

147 automated tests passed on Windows with Python 3.12.10. Coverage including the
migration importer was 92.78%, above the 85% gate and below the prior 94.06% baseline.
The isolated runtime contains no Chroma package. Strict pip-audit of its 62-package
resolved runtime graph returned zero known vulnerabilities. Runtime Bandit returned
zero findings. An expanded scan including the rehearsal launcher reports three low
severity subprocess findings. The launcher executes explicit local Python paths,
constant source strings and no shell; operators must trust the interpreter paths.
These findings have not been suppressed.

Synthetic legacy export was executed with socket connection and DNS blocked in a
temporary directory, without provider keys. Complete IDs, text, typed metadata,
document IDs, filtered cosine scores (1.0 and 0.6), repeated import, restart count,
source listing and deletion passed. Automated tests verify recovery after an injected
import failure and rejection of a checksum-corrupted export. Large live datasets,
Linux behavior, Docker image scans and remote CI remain unverified.

Reproduce the full synthetic round trip with the new runtime interpreter:
python scripts/migration/rehearse.py ABSOLUTE_PATH_TO_ISOLATED_LEGACY_PYTHON
Install chromadb==1.5.9 only in that disposable legacy environment.
