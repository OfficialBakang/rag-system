# Security and supported deployment

The supported deployment is a single tenant, one Uvicorn worker and a local persistent Chroma database. Every reader can query the whole corpus. The service credentials identify applications sharing a role. Individual user identities and document-level permissions require a different access model.

## Threat model

| Asset and trust boundary | Finding in the previous API | Control in this release | Residual exposure |
|---|---|---|---|
| Documents and administrative actions at the HTTP boundary | Anonymous reads, writes and deletion | Mandatory header credentials, role policy, minimal public probes | A stolen role key has that role's full corpus access |
| Provider budget and service capacity | Unlimited queries, uploads and evaluations, including through the query flag | Per-role quotas, shared anonymous quota, AI concurrency, byte, page, chunk, context and embedding-call limits | Trusted callers can consume their entire quota. Slow clients need gateway controls |
| Files crossing into the parser | Whole-file reads, permissive decoding, temporary files left after errors | Bounded body and file reads, strict decoding, generated temporary paths, PDF signature/page checks, cleanup | PDF decompression and parser defects can consume CPU and memory before extraction limits take effect |
| Source passages crossing into AI prompts | Documents could impersonate instructions | Escaped passage boundaries, separate system instructions, bounded context, no model tools or privileged actions | Models can still follow embedded instructions, disclose corpus content or hallucinate |
| Requests and provider failures crossing into observability | Raw exceptions, prompts, answers and source labels reached responses or logs | Generic errors, server-generated correlation IDs, numeric span attributes, limited evaluation history | Existing files, backups, provider-side records and independently configured agents have separate retention policies |
| Source, packages and base image crossing into deployment | CI exercised behavior without scanning for known vulnerabilities | Separate source, dependency and runtime image scans | Scanner coverage and advisory databases are incomplete |

## Service credentials

Generate **three independent** 32-byte keys using Python's cryptographic generator. Run the following once per role in a private local terminal, then place the output directly in the corresponding secret manager entry or restricted local environment file. Do not include values in tickets, shell command arguments, image layers or source control.

```cmd
python -c "from secrets import token_hex; print(token_hex(32))"
```

Set `RAG_READER_API_KEY`, `RAG_EDITOR_API_KEY` and `RAG_ADMIN_API_KEY`. Startup requires distinct lower-case 64-character hex values in both operating modes. Format checks can reject obvious weak values. Only the generation procedure establishes randomness. Rotate a credential by replacing its environment value and recreating the service. Restrict access to Docker and its host because administrators can inspect runtime environment values. OpenAI and Anthropic credentials serve a separate purpose.

Send a single `X-API-Key` header. Missing or invalid credentials receive 401, insufficient permissions receive 403. URLs, cookies and forwarding headers do not establish identity. Swagger's Authorize dialog supports the header. Browser streaming clients should use fetch with request headers.

| Minimum role | Method and endpoints |
|---|---|
| Public | GET /live, GET /health |
| Public when documentation is enabled | GET /docs, GET /docs/oauth2-redirect, GET /redoc, GET /openapi.json |
| Reader | POST /query, POST /query/stream |
| Editor | POST /ingest/text, POST /ingest/file, GET /sources |
| Admin | DELETE /documents/{doc_id}, POST /evaluate, GET /evals, GET /metrics, GET /traces, GET /ready |
| Admin | POST /query with evaluate=true, charged against both query and evaluation quotas |

Roles inherit the permissions above them. Streaming returns tokens only. Use the non-streaming endpoint for evaluation. Streaming rejects evaluation requests. Public health returns only status. Detailed storage statistics are available through authenticated metrics. Authorised AI operations in smoke mode return 503.

## Configuration and limits

The complete configuration is in `.env.example`. Each `RAG_LIMIT_*` value is a positive integer. Limits default to 60 general requests, 10 queries, 5 ingestions and 5 evaluations per role per minute, plus three concurrent AI requests across roles. One shared 120/minute anonymous/public budget bounds memory independently of client-controlled identities. Public probes share that budget. Rate and concurrency refusals carry HTTP 429 and Retry-After. Restarting the worker resets its windows. Multiple workers or replicas require shared quotas or a gateway.

The defaults allow 2,000 query characters, top-k 20, 100,000 raw characters, 12 MiB per HTTP body, 10 MiB per file, 100 PDF pages, 200,000 extracted characters, 256 chunks and three embedding batches per ingestion (including retry calls). The retrieved context budget is 32,000 characters. MAX_TOKENS defaults to 1,024 with a startup ceiling of 8,192. Invalid requests receive 422, unsupported media 415, and oversized documents or bodies 413. Malformed or encrypted PDFs are rejected. PDF, UTF-8 text, Markdown and HTML uploads are parsed locally. There is no malware scanner or parser sandbox.

Configure exact comma-separated HTTP(S) origins in `RAG_CORS_ORIGINS` if a separate browser origin needs access. The default allows same-origin use. CORS controls browser access to responses. Service authentication applies independently. Set `RAG_PUBLIC_DOCS=false` to remove public Swagger, ReDoc and schema routes. Restart after configuration changes.

## Deployment and diagnostics

Keep Compose bound to loopback. Before public exposure, use a TLS-terminating gateway, configure connection, header, body and idle timeouts, set memory/CPU and disk quotas, protect backups and apply network access controls. Configure HSTS at the HTTPS gateway. Apply connection limits and coarse rate limits before requests reach Python. Disable proxy logging of raw URLs, headers, prompts and bodies. Configure explicit trusted proxy peers if adding forwarded-header support. Restrict data directory access and encrypt storage and backups according to your data classification.

Validation errors omit rejected values. Generic upstream errors use 502 or 503. A stream already sending HTTP 200 reports a generic SSE error event if generation fails. Use X-Request-ID to correlate failures with server logs. Historical trace and evaluation files may contain plaintext from previous versions. Restrict and review them and their backups before enabling administrative access. Deployments needing stronger PDF isolation should parse in a resource-limited sandbox. Generated answers and source labels remain untrusted when displayed in other applications.

## Vulnerability management

CI uses Bandit 1.9.4 for runtime Python, pip-audit 2.10.1 for the resolved runtime dependency graph, and Trivy v0.76.2 through the SHA-pinned v0.36.0 action for OS and language packages in the built image. All Bandit findings and known pip-audit vulnerabilities block the job. High and critical Trivy findings block it, including those without fixes. Trivy uses exit code 42 for policy findings. Report generation, advisory database, package-resolution and registry failures require rerunning or repairing the scanner. Treat an incomplete scan as unverified.

Triage critical issues immediately and high issues before release. Verify affected configurations and compatible fixed versions, rerun the suite, rebuild and rescan. Track lower-severity findings along with upstream fixes. Document any exceptional release decision with an owner, affected versions, reason and expiry. Passing these gates covers only the threats the configured tools detect.

Report suspected vulnerabilities privately to the repository maintainer through GitHub private vulnerability reporting if enabled, or request a private channel from the maintainer. Include the affected revision, deployment model and impact. Omit live credentials, private documents and raw production traces.


### Audit of the Windows runtime on 28 November 2026

pip-audit inspected the 101 installed distributions in the Python 3.12 runtime dependency graph. It reported five entries for chromadb 1.5.9, covering four unique advisories. The advisory feed returned the first identifier twice. GitHub's advisories give the following severity ratings.

| Advisory | Severity | Patched release listed |
|---|---|---|
| [CVE-2026-45829](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c) | Critical | None |
| [CVE-2026-45833](https://github.com/advisories/GHSA-36p7-vc44-83pf) | Critical | None |
| [CVE-2026-45831](https://github.com/advisories/GHSA-xph7-9rjv-w5fr) | High | None |
| [CVE-2026-45830](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr) | High | None |

The app constructs an embedded PersistentClient and does not expose a Chroma server or delegate authorisation to Chroma RBAC. That reduces exposure to server-specific paths. The vulnerable distribution remains in the image and these findings block the dependency gate. Keep the service restricted while tracking upstream. Validate storage compatibility, existing collections and container recreation against a disposable copy before adopting a fixed release. An alternate database would require a separate migration.
