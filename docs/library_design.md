last updated: 2026-02-14

# star-schema-llm-context: shared library design

## vision

A reusable Python library providing Kimball-style dimensional modeling primitives for LLM agent systems. Three repos converge on one insight: **the same DAG structure -- decompose, route, prune, synthesize, verify -- appears in database query planning, distributed search, sparse MoE transformers, and agent hierarchies**. The universal primitive is routing: given limited compute, which subset of the search space to activate.

This library is the storage engine layer. It handles I/O, schema management, key generation, and connection lifecycle. It does not know business logic -- that lives in consumer repos that import and extend it.

## motivation: why this matters

### the gap in the ecosystem

Comprehensive research (see `docs/abstraction_analogies.md`) found that **no existing project uses DuckDB as a persistent state store for LLM agent operations**. The landscape has:

- **Vector DBs** (Pinecone, Chroma, Qdrant): semantic search over embeddings. No relational queries, no temporal tracking, no star schema.
- **Conversation memory** (MemGPT/Letta, Zep, Mem0): stores chat history and summaries. Complementary to operational data but different use case.
- **Observability platforms** (LangSmith, Braintrust, Phoenix/Arize): SaaS, cloud-first, no local-first option. Expensive at scale.
- **OpenTelemetry GenAI**: spans and traces. Good for observability, not for dimensional analysis.

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

Three architectural expansions that move the system from "smart storage" to a "self-optimizing organism." These build on the existing star schema foundation and are informed by cross-analysis with Gemini Deep Think (source: `fb-claude-skills/internal/research/20260214-gemini-feedback.md`).

### expansion 1: runtime layer (observability / self-optimization)

**Current state**: tracks skill definitions (what the agent knows). **Gap**: doesn't track execution (what the agent does).

`fact_tool_call` (already designed in the DAG hierarchy section above) is the foundation. It's the **EXPLAIN ANALYZE output** for agent execution. Without it, you can measure wall-clock time but can't identify which routing decision was expensive.

Additional schema:

```sql
-- Cost-aware routing: token usage per session
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

**Capability unlocked**: cost-aware routing. An agent can query its own history: "My top 5 most expensive tools by token count are X. I should switch from `read_file` (full scan) to `grep` (index scan) to respect the token constraint." This is the agent equivalent of a database query optimizer learning from past execution plans.

**Capture mechanism**: Claude Code hooks write JSONL events -> `journal.py` batch-ingests into DuckDB. Already implemented in fb-claude-skills v0.5.0.

### expansion 2: semantic layer (hybrid search / fuzzy routing)

**Current state**: routing is sparse/exact (MD5 hash matching). If the agent searches for "dataframe help" but the skill is named `pandas_guide`, the hash join fails.

**Gap**: no fuzzy/semantic routing. Need to complement exact matching with vector similarity.

```sql
-- Embedding dimension for fuzzy skill routing
CREATE TABLE IF NOT EXISTS dim_skill_embedding (
    skill_key        TEXT NOT NULL,       -- references dim_skill.hash_key
    model_version    TEXT NOT NULL,       -- e.g. 'text-embedding-3-small'
    embedding        FLOAT[1536],        -- vector representation
    content_summary  TEXT,               -- the text chunk that was embedded
    effective_from   TIMESTAMP NOT NULL DEFAULT current_timestamp,
    effective_to     TIMESTAMP,
    is_current       BOOLEAN NOT NULL DEFAULT TRUE,
    record_source    TEXT NOT NULL
);
```

**Capability unlocked**: hybrid search (SQL filter + vector rank). Filter dimensions with structured predicates (`WHERE auto_update = TRUE`), then rank by `array_cosine_similarity(embedding, $query_vec)`. DuckDB's `vss` extension makes this feasible without adding another database.

**Design note**: this is complementary to the existing hash-based routing. Exact matching handles known entities; vector search handles discovery of unknown but relevant entities. The two compose: filter first (cheap), rank second (expensive).

### expansion 3: graph layer (dependency DAG / impact analysis)

**Current state**: `skill_source_dep` maps sources to skills (flat routing table). **Gap**: no recursive dependency tracking between skills.

Skills depend on other skills (mlx-lm references mlx, dimensional-modeling would reference star-schema-llm-context). When a breaking change is detected in a leaf skill, the impact propagates up the DAG.

```sql
-- Recursive skill dependency graph
CREATE TABLE IF NOT EXISTS dim_skill_dependency (
    parent_skill_key TEXT NOT NULL,       -- the importer
    child_skill_key  TEXT NOT NULL,       -- the imported
    dependency_type  TEXT NOT NULL,       -- 'imports', 'calls', 'extends', 'references'
    is_critical      BOOLEAN DEFAULT TRUE, -- if child breaks, does parent break?
    effective_from   TIMESTAMP NOT NULL DEFAULT current_timestamp,
    effective_to     TIMESTAMP,
    is_current       BOOLEAN NOT NULL DEFAULT TRUE,
    record_source    TEXT NOT NULL
);
```

**Capability unlocked**: impact analysis via recursive CTEs. When `fact_change` detects a breaking change in `logging_utils`, query the dependency graph:

```sql
WITH RECURSIVE impact AS (
    SELECT child_skill_key AS skill_key, 1 AS depth
    FROM dim_skill_dependency
    WHERE child_skill_key = ? AND is_current = TRUE AND is_critical = TRUE
    UNION ALL
    SELECT d.parent_skill_key, i.depth + 1
    FROM dim_skill_dependency d
    JOIN impact i ON d.child_skill_key = i.skill_key
    WHERE d.is_current = TRUE AND d.is_critical = TRUE
)
SELECT DISTINCT s.skill_name, i.depth
FROM impact i
JOIN dim_skill s ON s.hash_key = i.skill_key AND s.is_current = TRUE
ORDER BY i.depth;
```

This is the "exploded view diagram" -- knowing that removing the alternator kills the battery.

### expansion summary

| Layer | Current (v2) | Expanded (v3+) | Capability Gained |
|-------|-------------|----------------|-------------------|
| Runtime | Static definitions | Execution logs (`fact_tool_call`, `fact_token_usage`) | Self-optimization (cost/speed awareness) |
| Semantic | Exact hash matching | Vector embeddings (`dim_skill_embedding`) | Fuzzy routing (finding unknown-but-relevant tools) |
| Graph | Flat source-skill map | Recursive dependency DAG (`dim_skill_dependency`) | Impact analysis (predicting cascading breakage) |

The progression: fact tables are sensory inputs, dimension tables are the entity model, views are reflexes. `fact_tool_call` adds proprioception (the system observing its own execution). Embeddings add intuition (finding things by meaning). The dependency graph adds foresight (predicting consequences).
