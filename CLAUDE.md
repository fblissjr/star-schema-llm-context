# star-schema-llm-context

Pattern library for Kimball-style dimensional modeling in LLM agent systems. This repo is documentation and design specs -- not a code library (yet).

## what this repo is

Design documents and reference implementation specs for using DuckDB star schemas as agent state persistence. The working implementation lives in [fb-claude-skills/skill-maintainer/scripts/store.py](https://github.com/fblissjr/fb-claude-skills/blob/main/skill-maintainer/scripts/store.py).

## what this repo is NOT

- Not a production system
- Not a framework or SDK
- Not the old knowledge graph prototype (that code has been deleted)

## key files

| File | Purpose |
|------|---------|
| `docs/library_design.md` | Reference implementation spec: primitives, schema patterns, DAG model, consumer patterns, design rationale |
| `docs/abstraction_analogies.md` | Conceptual framework: selection under constraint, five invariant operations, database analogy |
| `README.md` | Vision statement and pattern summary |

## three-repo architecture

| Repo | Role | Database analog |
|------|------|----------------|
| star-schema-llm-context (this) | Storage engine spec | I/O, schema, key generation |
| fb-claude-skills | Stored procedures | Skills, CDC pipeline, DuckDB store |
| ccutils | Client application | Session analytics, dashboards |

## core patterns (reference)

### key generation
- `dimension_key(*natural_keys)` -- MD5 surrogate from natural key components
- `hash_diff(**attributes)` -- MD5 of non-key attributes for SCD Type 2 change detection

### SCD Type 2 on all dimensions
- hash_key (no PK constraint -- multiple rows per entity)
- effective_from / effective_to / is_current
- hash_diff for change detection
- record_source, session_id, created_at

### fact tables
- No PKs, no sequences
- Grain = composite dimension keys + event timestamp
- Metadata: inserted_at, record_source, session_id

### design decisions
- MD5 for surrogates: deterministic, no coordination, portable
- No FK constraints: Kimball convention, ETL controls quality
- No fact table PKs: append-only event logs, composite grain is sufficient
- DuckDB: embedded, columnar, analytical SQL, zero config

## conventions

- No emojis in code or docs
- Use `uv` for Python (never pip)
- Use `orjson` for JSON serialization
- Docs use lowercase filenames with underscores
