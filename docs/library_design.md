last updated: 2026-02-14

# star-schema-llm-context: shared library design

## vision

A reusable Python library providing star schema-style dimensional modeling primitives for LLM agent systems. Three repos converge on one common pattern: **the same DAG structure -- decompose, route, prune, synthesize, verify -- appears in database query planning, distributed search, sparse MoE transformers, and agent hierarchies**. The common primitive is routing: given limited compute, which subset of the search space to activate.

This library is the storage engine layer. It handles I/O, schema management, key generation, and connection lifecycle. It does not know business logic -- that lives in consumer repos that import and extend it.

## motivation: why this matters

### the gap in the ecosystem

Using a star schema modeling approach with DuckDB as a persistent state store for LLM agent operations**

DuckDB fills a specific gap: **local-first, analytical, columnar, embedded, zero-dependency OLAP** that runs in the same process as the agent. No network hop, no auth, no cloud dependency.

### three repos, three roles

```
star-schema-llm-context/           THE LIBRARY (the "what")
  Core dimensional modeling primitives
  Hash key generation, DuckDB connection management
  Common dimension patterns (dim_date, dim_session, dim_project)
  The DAG hierarchy model (goal -> task -> branch -> session -> agent -> tool chain)
  Routing/pruning/synthesis/verification event model
  Schema migration framework

fb-claude-skills/                  THE SKILL (the "how to use it")
  <dimensional-modeling>/
    SKILL.md                       Teaches Claude agents dimensional modeling
    references/                    Each ref = an abstraction loaded on demand

ccutils/                           A CONSUMER (session analytics)
fb-claude-skills/skill-maintainer/ A CONSUMER (skill maintenance state)
```

### the database analogy (why it's not just a metaphor)

| Role | Repo | Database analog |
|------|------|----------------|
| Storage Engine / Kernel | `star-schema-llm-context/` | Handles I/O, memory management, locking. No business logic. |
| Stored Procedures / System Catalog | `fb-claude-skills/` | Business logic. Loading a skill = `CREATE OR REPLACE PROCEDURE`. |
| Client Application | `ccutils/` and consumers | Orchestration layer that initiates sessions and calls procedures. |

---

## core primitives

### key generation

Single implementation, used everywhere. Extracted from fb-claude-skills `store.py` (v0.6.0) and ccutils `generate_dimension_key()`:

```python
# star_schema.keys

import hashlib

def dimension_key(*natural_keys) -> str:
    """MD5 surrogate key from natural key components. Kimball-style.

    Deterministic: same inputs always produce the same key.
    NULL-safe: None values are replaced with '-1' sentinel.
    Composable: dimension_key(source_key, url) works for composite keys.
    """
    parts = [str(k) if k is not None else "-1" for k in natural_keys]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def hash_diff(**attributes) -> str:
    """MD5 of non-key attributes for SCD Type 2 change detection.

    Pass the mutable attributes of a dimension row. If the hash changes,
    the dimension has changed and needs a new SCD Type 2 row.
    """
    parts = [f"{k}={v}" for k, v in sorted(attributes.items()) if v is not None]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
```

### connection management

```python
# star_schema.connection

import duckdb
from pathlib import Path

class DuckDBStore:
    """Managed DuckDB connection with schema versioning."""

    def __init__(self, path: Path, schema_version: int):
        self.path = path
        self.expected_version = schema_version
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self._check_version()

    def _check_version(self):
        """Check and migrate schema if needed."""
        # ... migration logic

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

### common dimensions

Reusable dimension patterns that every consumer needs:

```python
# star_schema.dimensions

# dim_date: calendar date with hierarchies (year, quarter, month, week, day_of_week)
# dim_time: time of day (hour, minute, period)
# dim_session: Claude Code session (degenerate dimension -- carried in fact rows as session_id)
# dim_project: project directory with git metadata
```

`dim_session` is intentionally a **degenerate dimension** -- the session UUID is the natural key and it's carried directly in fact rows. No separate dimension table. This is a Kimball pattern for high-cardinality dimensions where the natural key IS the interesting attribute.

### SCD Type 2 pattern

Every dimension table follows this template:

```sql
CREATE TABLE IF NOT EXISTS dim_<name> (
    hash_key         TEXT NOT NULL,       -- MD5 of natural keys (no PK: SCD Type 2 needs multiple rows per entity)
    -- business columns here --
    effective_from   TIMESTAMP NOT NULL DEFAULT current_timestamp,
    effective_to     TIMESTAMP,           -- NULL = current row
    is_current       BOOLEAN NOT NULL DEFAULT TRUE,
    hash_diff        TEXT,                -- MD5 of non-key attributes
    record_source    TEXT NOT NULL,       -- lineage: who wrote this row
    created_at       TIMESTAMP NOT NULL DEFAULT current_timestamp,
    session_id       TEXT,                -- Claude Code session (if applicable)
    last_verified_at TIMESTAMP            -- staleness signal
);
```

### fact table pattern

Fact tables have **no primary keys, no sequences**. The grain is the composite of dimension surrogate keys + event timestamp:

```sql
CREATE TABLE IF NOT EXISTS fact_<name> (
    -- dimension keys (TEXT, referencing hash_key by convention) --
    -- measure columns --
    -- event timestamp --
    inserted_at      TIMESTAMP NOT NULL DEFAULT current_timestamp,
    record_source    TEXT NOT NULL,
    session_id       TEXT
);
```

### meta tables

Every database gets:

```sql
-- Schema versioning
CREATE TABLE IF NOT EXISTS meta_schema_version (
    version      INTEGER NOT NULL,
    applied_at   TIMESTAMP NOT NULL DEFAULT current_timestamp,
    description  TEXT
);

-- Load logging (operational visibility)
CREATE TABLE IF NOT EXISTS meta_load_log (
    script_name  TEXT NOT NULL,
    started_at   TIMESTAMP NOT NULL DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    rows_inserted INTEGER DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'running',
    error_message TEXT,
    session_id   TEXT
);
```

---

## the DAG hierarchy model

The universal decomposition that every consumer shares:

```
Goal / Objective     (why -- spans days, sessions)
  Task               (what -- decomposed unit)
    Branch / Attempt (how -- specific approach, may be pruned)
      Session        (where -- execution boundary)
        Agent        (delegation -- routed sub-problem)
          Tool chain (sequence -- ordered operations)
            Tool call(atomic -- read/write/search/execute)
```

### mapping to the five invariant operations

| Phase | DB Query Planning | Sparse MoE | Agent Hierarchy | CDC Pipeline |
|-------|------------------|------------|----------------|-------------|
| Decompose | Parse SQL into logical plan | Tokenize input | Break request into subtasks | Split pages by delimiter |
| Route | Optimizer selects indexes | Router picks top-k experts | Spawn subagents per subtask | Hash pages, match to skills |
| Prune | Predicate pushdown | Gate zeros non-selected | Kill unproductive agents | Skip unchanged (hash match) |
| Synthesize | Join/aggregate results | Weighted sum of outputs | Merge subagent results | Collect into unified report |
| Verify | Check constraints, return | Output guardrails | Quality check merged answer | Classify severity, validate |

### fact tables for the DAG

Future fact tables that complete the hierarchy:

```sql
-- The EXPLAIN ANALYZE output for agents
CREATE TABLE IF NOT EXISTS fact_tool_call (
    session_id       TEXT NOT NULL,       -- degenerate dimension
    agent_key        TEXT,                -- which agent/subagent
    tool_name        TEXT NOT NULL,
    called_at        TIMESTAMP NOT NULL DEFAULT current_timestamp,
    duration_ms      INTEGER,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    status           TEXT,                -- success/error/timeout
    target_path      TEXT,
    metadata         TEXT,
    inserted_at      TIMESTAMP NOT NULL DEFAULT current_timestamp,
    record_source    TEXT NOT NULL,
    session_id_      TEXT                 -- capturing session
);

-- Token usage per session (cost tracking)
CREATE TABLE IF NOT EXISTS fact_token_usage (
    session_id       TEXT NOT NULL,
    measured_at      TIMESTAMP NOT NULL DEFAULT current_timestamp,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    model            TEXT,
    cost_usd         REAL,
    inserted_at      TIMESTAMP NOT NULL DEFAULT current_timestamp,
    record_source    TEXT NOT NULL
);
```

`fact_tool_call` is critical: without logging routing decisions, you can't optimize the agent. You don't know why it chose chain-of-thought (nested loop join) over direct tool call (hash join).

---

## how consumers import and extend

### fb-claude-skills (skill maintenance)

Currently implements its own store.py with the full Kimball schema (v0.6.0). Future: import core primitives from this library, extend with skill-specific dimensions and facts.

```python
# Future consumer pattern
from star_schema.keys import dimension_key, hash_diff
from star_schema.connection import DuckDBStore

class SkillStore(DuckDBStore):
    """Skill-maintainer-specific store."""

    def __init__(self, path, config_path):
        super().__init__(path, schema_version=2)
        self._create_skill_tables()
        self._sync_from_config(config_path)
```

### ccutils (session analytics)

Already uses `generate_dimension_key()` for session tracking. Would import the canonical implementation.

### new consumers

Any agent system that needs to track state over time can import the library:

```python
from star_schema.keys import dimension_key
from star_schema.connection import DuckDBStore

# Create a store for your domain
store = DuckDBStore(Path("my_agent/state/agent.duckdb"), schema_version=1)

# Use the key generation
user_key = dimension_key(user_id)
project_key = dimension_key(org, project_name)
```

---

## implementation status

### done (in fb-claude-skills v0.6.0)

- Hash-based surrogate keys (`_hash_key()`)
- Hash-diff change detection (`_hash_diff()`)
- SCD Type 2 on all dimensions
- No fact table PKs (dropped all 6 sequences)
- Metadata columns on all tables (record_source, session_id, inserted_at)
- Meta tables (meta_schema_version, meta_load_log)
- Schema migration (v1 -> v2 automatic)
- Merged session model (fact_session_event replaces fact_session + fact_session_event)
- All consumer scripts updated and verified

### future (library extraction)

- Extract core primitives into `src/star_schema/` package
- Add `pyproject.toml` with uv build system
- Migrate from `pip`/`requirements.txt` to `uv`
- Common dimension patterns (dim_date, dim_time, dim_project)
- The DAG hierarchy fact tables (fact_tool_call, fact_token_usage)
- Schema migration framework (numbered SQL files)
- Tests

---

## package structure (planned)

```
star-schema-llm-context/
  pyproject.toml                # uv build system
  src/
    star_schema/
      __init__.py
      keys.py                   # dimension_key(), hash_diff()
      connection.py             # DuckDBStore base class
      dimensions.py             # Common dimension patterns
      migration.py              # Schema migration framework
      patterns/
        scd_type2.py            # SCD Type 2 helper
        degenerate.py           # Degenerate dimension pattern
  docs/
    library_design.md           # This file
    abstraction_analogies.md    # The unified framework
  tests/
    test_keys.py
    test_connection.py
    test_scd.py
```

---

## design decisions and rationale

### why MD5 for surrogate keys?

- Deterministic: same natural key always produces same surrogate. No sequence coordination.
- Portable: keys are meaningful across databases and environments.
- Collision-resistant enough: MD5 has known collision weaknesses for cryptographic use, but for surrogate key generation in a bounded domain (hundreds of dimensions, not billions), the 2^64 collision resistance is more than sufficient.
- Human-readable-ish: 32 hex chars. Not great, but better than UUIDs for debugging.

### why no FK constraints?

Kimball convention. FKs enforce referential integrity at write time, which:
1. Slows down bulk loads (the primary write pattern for fact tables)
2. Creates coupling between tables that makes schema evolution harder
3. Is unnecessary when the ETL pipeline controls data quality

Instead: join by convention (fact.source_key = dim_source.hash_key) and validate at the application layer.

### why SCD Type 2 (not Type 1)?

Type 1 (overwrite) loses history. When a skill path changes, you want to know both the old and new path, and when the change happened. SCD Type 2 preserves the full history with `effective_from`/`effective_to` ranges.

### why no PRIMARY KEY on dimension tables?

SCD Type 2 requires multiple rows per entity: one current row (`is_current = TRUE`) and N historical rows (`is_current = FALSE`). All rows share the same `hash_key` (derived from the natural key). A `PRIMARY KEY` constraint on `hash_key` would prevent inserting the new row when closing the old one. DuckDB's columnar storage doesn't benefit from PK indexes for join performance the way row stores do -- predicate pushdown and zone maps handle that. All queries filter on `is_current = TRUE` to find the current row.

### why no fact table PKs?

Fact tables are append-only event logs. Their grain is the composite of dimension keys + timestamp. A synthetic PK adds no information and wastes space. If you need to identify a specific fact row, the composite grain is unique enough (and if it's not, that's a data quality issue, not a schema issue).

### why DuckDB (not SQLite, PostgreSQL, etc.)?

- **Embedded**: runs in the same process. No server, no network, no auth.
- **Columnar**: OLAP queries (aggregates, group-by, window functions) are fast.
- **Analytical SQL**: proper window functions, CTEs, INTERVAL arithmetic.
- **WAL mode**: concurrent readers don't block writers.
- **Zero config**: just a file path. `.gitignore` it, reconstruct from state.json + config.yaml.

---

## expansion roadmap (v3+)

### primary: DAG execution model

Track agent execution as a data pipeline DAG. Agents ARE data pipelines. A Claude Code session executing a complex task IS a DAG: goal decomposes to tasks, tasks route to subagents, subagents execute tool chains, tool chains produce outputs that get synthesized.

The five invariant operations aren't just a communication framework here -- they're literally the fact table grain:

```sql
fact_task_decomposition  -- tracks decompose phase (goal -> tasks, what was decomposed and how)
fact_routing_decision    -- tracks route phase (task -> agent/tool, what was routed where)
fact_execution_step      -- tracks individual stage execution (atomic: tool call with timing, tokens, status)
fact_pruning_event       -- tracks what was killed/abandoned
fact_synthesis_result    -- tracks merge/synthesis (merged output with quality signal)
fact_verification        -- tracks quality checks
```

**What it enables**: Not "how much did it cost" but "what routing decisions did the agent make, and which patterns succeed?"
- "This decomposition pattern succeeds 80% of the time; that one fails 60%"
- "When the agent spawns >3 subagents, synthesis quality drops -- prune earlier"
- "Read-first-then-write routing outperforms write-then-fix by 2x in tool calls"

**Capture mechanism**: Claude Code hooks (PostToolUse, SubagentStart, SubagentStop, Stop, TaskCompleted). Journal.py pattern: JSONL buffer -> batch ingest into DuckDB. Already implemented in fb-claude-skills v0.5.0.

**Write-back**: Because dimensions are mutable (SCD Type 2), you can fix routing problems by modifying the data, not the code. Update a skill description in dim_skill, change a source URL in dim_source, deprecate a tool in dim_agent_node -- the agent picks up the change on next read without a redeploy. The schema is the stable layer; behavior changes by changing what's in it.

### secondary: patterns that enable automation

Multiple patterns focused on what enables automating the next level of abstraction:

- **Consistency metrics**: Track how consistently an agent follows patterns across sessions. Consistent behavior is automatable; inconsistent behavior isn't.
- **Pattern extraction**: Identify recurring tool call sequences that could become skills. "You do this 5-step sequence in every code review -- make it a skill."
- **Abstraction readiness scoring**: Given a pattern's consistency, frequency, and success rate, score whether it's ready to be automated as a skill. This is the bridge from "interesting data" to "automated behavior."

The key insight: just like models went from needing 10+ few-shot examples to zero-shot as they internalized patterns, agent workflows go from manual sequences to automated skills as the patterns become consistent enough to codify.

### what's NOT in the roadmap (and why)

- **Embeddings / vector search**: Not ready until multivector retrieval is built into the harness/LLM. Adding it now creates a premature dependency on unstable embedding infrastructure.
- **Full graph database**: Overcomplicated for 99% of use cases. Recursive CTEs in DuckDB handle the dependency DAG use case without adding another database.
- **Knowledge graph code analysis**: The old star-schema-llm-context prototype. Deleted. Language servers and tree-sitter do this better.
