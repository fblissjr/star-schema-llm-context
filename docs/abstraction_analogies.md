last updated: 2026-02-14

# abstraction analogies: selection under constraint

## the unifying insight

Three repos converge on one observation: **the same DAG structure -- decompose, route, prune, synthesize, verify -- appears in database query planning, distributed search, sparse MoE transformers, Gemini Deep Think, and agent hierarchies**. The universal primitive is routing: given limited compute, which subset of the search space to activate.

This document is the canonical treatment of the analogy, synthesizing analysis from Claude agent research (fb-claude-skills session af5bb1d) and Gemini analysis (internal/research/20260214-gemini-analysis-analogies.md).

---

## the corrected mental model

### what breaks: the vertical compilation analogy

The initial analogy mapped tokens to bytecode and skills to Python bindings in a vertical compilation stack. This breaks because:

- **Skills are data, not code.** A skill gets loaded into the context window (a mutable data structure). It doesn't get compiled. Model weights are the compiled binary -- they don't change per session.
- **The context window is a temporary table**, not a program counter. It has a fixed capacity (`[1, seq_len, hidden_dim]`), and loading a skill is an `INSERT INTO context_buffer`.
- **Attention is a query operation**, not instruction execution. QK^T computes similarity scores (join condition), softmax normalizes them (WHERE clause), and multiplication by V aggregates results.

### what works: the database analogy

The corrected model maps the LLM system to a database system:

| LLM/Agent Concept | Database Analog | Why |
|-------------------|----------------|-----|
| Tokens | Rows in a temporary table | Context window = `[1, seq_len, hidden_dim]` table in RAM |
| Model weights | Compiled binary / storage engine | Don't change per session. Define capabilities. |
| Inference + tokenization | Query executor | Scan context, compute attention (fuzzy self-join), produce output |
| Skills | View definitions + stored procedures | Define projections (what to show) and control execution graphs |
| References | Materialized subqueries | Pre-computed knowledge loaded on demand |
| SKILL.md frontmatter | Routing metadata / index entries | Controls WHEN skill loads (not loaded = not scanned) |
| Context window | Working memory / temp table | Fixed capacity, must be managed explicitly |
| Skill loading | INSERT INTO context_buffer | Adds rows to the working set |
| Progressive disclosure | Lazy materialization | Load detail only when needed (layers 0-5) |
| The architecture | External query planner | Optimizes LLM I/O by selecting what context to load |

### attention as a fuzzy self-join

In SQL terms:
```
-- Attention mechanism
SELECT V_weighted
FROM context_tokens t1
CROSS JOIN context_tokens t2               -- QK^T: compute all pairwise similarities
WHERE softmax(Q(t1) * K(t2)^T) > threshold -- softmax: normalize to probability distribution
```

Agent routing (skills, references, tool selection) is an **explicit index** replacing the model's implicit scan of training data. Instead of relying on attention to find relevant knowledge in the model's weights, skills inject the relevant knowledge directly into context.

---

## parallel abstraction stacks

These stacks map **side by side**, not vertically mixed:

| Layer | Programming | Database | LLM/Agent | This Project |
|-------|------------|----------|-----------|-------------|
| Storage | Machine code | Disk pages, column segments | Tokens (rows in memory table) | DuckDB file on disk |
| Execution | CPU decode + execute | Query executor (scan, join) | Transformer forward pass (attention = fuzzy self-join) | `Store` class methods |
| Structure | Type system, structs | Schema (DDL) | Attention mask + learned associations (soft schema) | Star schema DDL (hard schema the soft schema populates) |
| Composition | Libraries, modules | Views, CTEs, stored procedures | Skills, tools, MCP servers | Shared library primitives |
| Interface | Python REPL | SQL client, BI dashboards | SKILL.md + references/ (view defs injected into context) | fb-claude-skills skill |

Key insight: the DDL in this project is a **hard schema that the model's soft schema populates**. The model's learned associations determine what data goes into the star schema. The star schema gives that data structure for analytical queries.

---

## the routing spine: five invariant operations

Every system that processes more possibilities than it can evaluate implements these five operations:

| Phase | DB Query Planning | Sparse MoE | Agent Hierarchy | CDC Pipeline | Gemini Deep Think |
|-------|------------------|------------|----------------|-------------|------------------|
| **Decompose** | Parse SQL into logical plan | Tokenize input | Break request into subtasks | Split pages by delimiter | Parse question into reasoning steps |
| **Route** | Optimizer selects indexes | Router picks top-k experts | Spawn subagents per subtask | Hash pages, match to skills | Select verification strategies |
| **Prune** | Predicate pushdown | Gate zeros non-selected | Kill unproductive agents | Skip unchanged (hash match) | Discard low-confidence paths |
| **Synthesize** | Join/aggregate results | Weighted sum of outputs | Merge subagent results | Collect into unified report | Combine verified conclusions |
| **Verify** | Check constraints, return | Output guardrails | Quality check merged answer | Classify severity, validate | Cross-check against source |

### concrete examples from this codebase

**CDC pipeline (docs_monitor.py)**:
1. **Decompose**: split `llms-full.txt` into per-page sections by `Source:` delimiter
2. **Route**: hash each page, compare to stored hashes, identify which pages changed
3. **Prune**: skip unchanged pages (hash match = predicate pushdown)
4. **Synthesize**: collect all changes into a classified report
5. **Verify**: run keyword heuristic (BREAKING/ADDITIVE/COSMETIC), validate skill

**Skill loading (progressive disclosure)**:
1. **Decompose**: skill split into layers (frontmatter, body, references, cross-refs, CLAUDE.md, coderef)
2. **Route**: frontmatter description determines whether skill loads at all
3. **Prune**: only load layers needed for current prompt (don't load all references)
4. **Synthesize**: compose skill content + loaded references into working context
5. **Verify**: validation against Agent Skills spec

---

## the six-layer progressive disclosure model

Drawn from mlx-skills (the best example of layered skill abstractions):

```
Layer 0: Frontmatter        (routing metadata -- controls WHEN skill loads)
Layer 1: SKILL.md body      (always loaded -- compressed essentials + pointers)
Layer 2: References          (on demand -- deep knowledge by category)
Layer 3: Cross-skill refs    (runtime -- skills compose: mlx -> mlx-lm -> fast-mlx)
Layer 4: CLAUDE.md           (project-level orchestration guide)
Layer 5: coderef/ upstream   (external ground truth)
```

fast-mlx SKILL.md (~350 words) is a pure router: it dispatches to domain-specific references. The skill system IS selection under constraint: limited context window, select what to load.

Reference files categorized by function:
- **Concept references** (explain "how it works"): schema_patterns, hierarchy_model
- **Pattern references** (show "how to do it"): query_patterns, key_generation
- **Problem references** (help "what went wrong"): anti_patterns, troubleshooting

Cross-skill composition: mlx-lm references mlx ("Load the `mlx` skill first for core concepts"). Similarly, a future `dimensional-modeling` skill would reference this library's docs.

---

## the DAG hierarchy

The universal decomposition that all consumers share:

```
Goal / Objective     (why -- spans days, sessions)
  Task               (what -- decomposed unit)
    Branch / Attempt (how -- specific approach, may be pruned)
      Session        (where -- execution boundary)
        Agent        (delegation -- routed sub-problem)
          Tool chain (sequence -- ordered operations)
            Tool call(atomic -- read/write/search/execute)
```

Each level maps to the five operations:
- **Goal -> Tasks**: decompose
- **Tasks -> Agents**: route
- **Agents evaluated**: prune (kill unproductive branches)
- **Agent results -> Answer**: synthesize
- **Answer -> Output**: verify

### fact_tool_call as EXPLAIN ANALYZE

Without logging routing decisions (`fact_tool_call`), you can't optimize the agent. You don't know why it chose chain-of-thought (nested loop join) over direct tool call (hash join). This is the equivalent of running SQL queries without EXPLAIN ANALYZE -- you can measure wall-clock time but can't identify which operation was expensive.

---

## three repos as database components

### star-schema-llm-context (Storage Engine / Kernel)

Handles I/O, memory management, locking. Does not know business logic.

Provides:
- Key generation (`dimension_key`, `hash_diff`)
- Connection management (`DuckDBStore`)
- Common dimension patterns (`dim_date`, `dim_session`, `dim_project`)
- Schema migration framework
- The DAG hierarchy model as schema patterns

### fb-claude-skills (Stored Procedures / System Catalog)

Business logic. Loading a skill is `CREATE OR REPLACE PROCEDURE`.

Provides:
- `dimensional-modeling` skill (teaches Claude agents the patterns)
- `skill-maintainer` store.py (concrete implementation of the library patterns)
- Reference documents (loaded into context on demand = procedure invocation)
- Validation against Agent Skills spec

### ccutils and consumers (Client Application)

Orchestration layer that initiates sessions and calls procedures.

Provides:
- Session analytics (who ran what, when, how much it cost)
- Dashboard queries against the star schema
- Hook integration (capture events from Claude Code into JSONL, flush to DuckDB)

---

## selection under constraint: the unified framework

The unified principle: **given more possibilities than you can evaluate, select the subset that matters, process it, combine results.**

This appears at every level:
- **Token level**: attention selects which context tokens influence the output
- **Skill level**: frontmatter routing selects which skills load
- **Reference level**: progressive disclosure selects which references load
- **Agent level**: task decomposition selects which subagents to spawn
- **CDC level**: hash comparison selects which pages to fetch and classify
- **Schema level**: views select which dimension rows are current (`is_current = TRUE`)

The star schema is the persistent record of these selections. fact tables capture what was selected (changes detected, validations run, content measured). dimension tables capture the entities being selected from (sources, skills, pages). Views compose the two to answer analytical questions (which skills are stale? which are over budget?).

---

## implementation status across repos

### star-schema-llm-context (this repo)

**Current state**: prototype with flat schema, requirements.txt, raw SQL.

**Target state** (future sessions):
- Python package with `src/star_schema/`
- uv-based build system
- Core primitives extracted from fb-claude-skills store.py
- Common dimensions and fact patterns
- Tests

### fb-claude-skills (v0.6.0, completed)

- Kimball dimensional model in `store.py`
- Hash-based surrogate keys (`_hash_key()`, `_hash_diff()`)
- SCD Type 2 on all dimensions
- No fact PKs, no sequences
- Metadata columns on all tables
- Meta tables (schema versioning, load logging)
- Merged session model (boundaries as events)
- 10 consumer scripts updated and verified
- Full schema documentation in `docs/internals/duckdb_schema.md`

### ccutils

- Uses `generate_dimension_key()` for session tracking
- Would import canonical key generation from this library

---

## source attribution

- Claude agent analysis: fb-claude-skills session af5bb1d (2026-02-14)
- Gemini Deep Think analysis: internal/research/20260214-gemini-analysis-analogies.md
- Data-centric agent state research: fb-claude-skills/docs/analysis/data_centric_agent_state_research.md
- DuckDB dimensional model strategy: fb-claude-skills/docs/analysis/duckdb_dimensional_model_strategy.md
