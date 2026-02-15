# star-schema-llm-context

**star-schema-llm-context** is a repo to both research and document patterns for using Kimball-style / star schema dimensional modeling as a way to abstract without breaking the lower level via a data/business logic/state persistence layer for LLM agent systems.

The core insight: abstract the data, not the behavior. LLM frameworks that abstract interaction patterns (chains, agents, retrievers) break when research moves faster than the abstraction. Dimensional modeling abstracts what happened (facts) in what context (dimensions) -- patterns that are 30+ years old and model-agnostic.

This repo provides:
- A reference implementation spec (how to structure a DuckDB star schema for agent state)
- Conceptual framework (the database analogy for skills, attention, and context management)
- Working on a POC in [fb-claude-skills](https://github.com/fblissjr/fb-claude-skills) store.py (v0.6.0): CDC tracking, validation history, token budgets, session journaling

This is a research & pattern library, not a code library. Each consumer builds its own domain-specific schema following these patterns.

## what's here

```
docs/
  library_design.md          Reference implementation spec
                             Core primitives, schema patterns, DAG hierarchy model,
                             consumer import patterns, design decisions with rationale
  abstraction_analogies.md   Conceptual framework
                             Selection under constraint, five invariant operations,
                             database analogy for skills, progressive disclosure model
```

## the database analogy

| Role | Repo | Database analog |
|------|------|----------------|
| Storage Engine / Kernel | `star-schema-llm-context/` | I/O, schema, key generation. No business logic. |
| Stored Procedures / System Catalog | `fb-claude-skills/` | Business logic. Loading a skill = CREATE OR REPLACE PROCEDURE. |
| Client Application | `ccutils/` and consumers | Orchestration. Initiates sessions, calls procedures. |

## core patterns

### key generation

MD5 hash surrogate keys from natural key components. Deterministic, NULL-safe, composable. No sequences, no coordination.

```python
import hashlib

def dimension_key(*natural_keys) -> str:
    parts = [str(k) if k is not None else "-1" for k in natural_keys]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

def hash_diff(**attributes) -> str:
    parts = [f"{k}={v}" for k, v in sorted(attributes.items()) if v is not None]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
```

### SCD Type 2 dimensions

Every dimension table: hash_key (no PK), business columns, effective_from/effective_to, is_current, hash_diff, record_source, session_id, created_at. Multiple rows per entity -- the whole point of SCD Type 2.

### fact tables

No PKs, no sequences. Grain = composite dimension keys + event timestamp. Append-only event logs. Metadata: inserted_at, record_source, session_id.

### meta tables

meta_schema_version (schema evolution tracking) and meta_load_log (operational visibility).

## working proof

[fb-claude-skills/skill-maintainer/scripts/store.py](https://github.com/fblissjr/fb-claude-skills/blob/main/skill-maintainer/scripts/store.py) implements the full Kimball schema (v0.6.0):

- 3 dimension tables (dim_source, dim_skill, dim_page) with SCD Type 2
- 6 fact tables (watermark checks, changes, validations, update attempts, content measurements, session events)
- Pre-built analytical views (freshness, budgets, trends)
- Automatic schema migration (v1 -> v2)
- Backward-compatible state.json export

## use cases

### primary: agent task decomposition DAG

Track agent execution as a data pipeline DAG. Goals decompose to tasks, tasks route to subagents, subagents execute tool chains, tool chains produce outputs that get synthesized. The five invariant operations (decompose, route, prune, synthesize, verify) become the fact table grain.

### secondary: skill quality lifecycle

Track how skills evolve from creation to maturity. Trigger accuracy, validation trends, reference freshness, token budget trajectory.

### future: cross-project pattern mining

Identify which conventions work across projects, surface candidates for promotion to shared skills.

See [docs/library_design.md](docs/library_design.md) for full schema designs and the expansion roadmap.

## related

- [fb-claude-skills](https://github.com/fblissjr/fb-claude-skills) -- stored procedures (skills, CDC pipeline, DuckDB store)
- [ccutils](https://github.com/fblissjr/ccutils) -- client application (session analytics)
