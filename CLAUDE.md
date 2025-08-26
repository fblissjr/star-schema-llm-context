# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code and Writing Style Guidelines

- **No emojis** in code, display names, or documentation
- Keep all naming and display text professional
- Avoid "Pure", "Enhanced", "Advanced", "Ultimate" type prefixes - use descriptive names instead
- Seriously, no emojis, not even checkboxes
- Avoid hyping up what this repo does. It's purely intended for testing new ideas and seeing what works. This should NOT be used in production scenarios, nor should it be advertised as such.

## Project Overview

This is an LLM Memory System using DuckDB and a star schema, dimensional modeling data model. It is **RESEARCH ORIENTED** and and aims to preserve context versioning and relationships in a different way.

This project should not anchor too far into the knowledge graph / graph side of things. Keep it simple.

## Background

This project came about from three places:

1. My anchoring on 20+ years in the data space, where star schema models reign king
2. 3 years of thinking about ways to better model data for consumption by LLMs. Initially, this was for retrieval use cases and embeddings, which was further validated by Meta's LCM paper, which was not a star schema, but had similar goals as my own research (Large Concept Models)[https://github.com/facebookresearch/large_concept_model]
3. Better data access controls and row-level security - star schemas make row-level security significantly easier to manage and extend and scale

## Problem Statement

Like computers of the 80s and 90s, working with LLMs today (or any transformer model) requires not just managing context length, but also managing the actual *content* that goes into that context. Those of us working with LLMs for the past few years take for granted how intuitive this is to us, but for many others, a single wrong input can cascade into a downward spiral - and that's bad for adoption and bad for consistency and bad for security.

Perhaps more importantly, reusing context - such as the app you built a few weeks ago, synthesized with a new idea, synthesized with new data inputs, synthesized with a colleague's context or another's team's context - is something that few people know you can do effectively. That means having the LLM you're using to build the 'new' thing rewrite the 'old' thing in it's native "language". And, of course, pulling in the right data at the right time.

So we should aim to make it easier to manage and reuse context. Some more thoughts on the "why" explore storing context in a data lake / database:

- **Context Across Sessions Across Teams**: It's frustrating enough to lose your context on your own hobbyist project - imagine what's possible for a whole team.
- **Token Size Limitations and Context Confusion**: Loading entire codebases causes a good chunk of (what's in the middle to be lost)[https://arxiv.org/abs/2307.03172]. More importantly, though, flooding the LLM with tokens that are unhelpful at best or misleading at worst will lead you down a path of wasted time and less than wonderful results.
- **Tracking Versions, Evals, and Auditing**: Can't track what was previously understood, can't audit what was used for inputs, and can't evaluate how things are getting better, worse, same, etc.
- **File-based Scalability**: Linear scanning of files doesn't scale. Hopping between markdown files is painful. As wonderful as grep is for Claude Code, it can't hop between relationships
- **Relationship Understanding**: A star schema creates connections between code elements or other context that look like your business or domain
- **Data Access Controls**: This won't solve prompt injections or other LLM security issues, but it does move us in the right step toward ensuring that users leveraging these tools have access to only the data they are permissioned to have. This makes having your context in a data lake much more reliable than hitting the source systems directly - which lines up nicely with the same best practices prior to LLMs.

## What This Repo's For

This is a *research-oriented* repo - not a production one. If anyone's even looking at this, it's mostly intended for my own uses and explorations, building in open to stay accountable.

Given the research-oriented approach, this readme will likely be out of date quite often. That said, here's an initial stab at a schema:

## Common Development Commands

### Initial Setup
```bash
make install    # Install Python dependencies
make setup      # Initialize database with schema
```

### Development
```bash
make run        # Start MCP server
make analyze    # Analyze current directory
make demo       # Run quick start demo
```

### Database Operations
```bash
make db-stats    # Show database statistics
make db-optimize # Optimize database (vacuum, analyze, refresh views)
```

### Testing & Maintenance
```bash
make test       # Run basic tests (DuckDB connection, schema verification)
make clean      # Clean generated files (database, logs, cache)
```

### Code Quality
```bash
make format     # Format code with black
make lint       # Run pylint on Python files
```

## Architecture Overview

The system uses a **Knowledge Graph + Star Schema** architecture:

1. **Knowledge Graph**: Represents code as nodes (projects→files→entities) with edges (imports, calls, extends)
2. **Star Schema**: Optimized fact/dimension tables for efficient context retrieval
3. **DuckDB Backend**: Columnar storage handling millions of nodes and billions of edges
4. **MCP Protocol**: Standard interface for LLM integration

### Core Components

- **mcp_server.py**: Main MCP server implementation with context management
- **graph_algorithms.py**: Graph algorithms (PageRank, community detection, anti-patterns)
- **schema.sql**: Complete database schema with tables, views, and functions
- **setup.py**: Database initialization and quick start demo

### Database Schema

The system uses the `llm_memory` schema with:
- **Node tables**: `node_project`, `node_file`, `node_entity`, `node_concept`
- **Edge tables**: `edge_relationship`, `edge_fast_lookup`
- **Fact tables**: `fact_interaction`, `fact_understanding`, `fact_graph_metrics`
- **Dimension tables**: `dim_session`, `dim_task`

## Key Implementation Details

### Context Management
The system prioritizes context based on relevance scores:
- Active tasks (0.95 weight)
- Recent interactions (0.85)
- Critical files (0.75)
- Hot spots (0.70)
- Code understanding (0.60)

Context is optimized to fit within token limits through intelligent compression and relevance-based filtering.

### Graph Traversal
Uses optimized SQL functions for BFS/DFS operations with configurable depth limits. Materialized views (`mv_file_dependencies`, `mv_call_graph`) accelerate common queries.

### Pattern Detection
Automatically identifies:
- Circular dependencies
- God classes (>20 methods, >15 dependencies)
- Dead code
- Code clones (>0.8 similarity)
- Long methods (>50 lines)

## Configuration

Main configuration is in `config.yaml`:
- Database settings (path, memory limits, threads)
- Context management (max tokens, prioritization weights)
- Graph algorithms (PageRank damping, community resolution)
- Analysis settings (languages, file types, ignore patterns)

## Testing Approach

Currently uses basic validation tests via Makefile. No formal test framework is set up yet. To test:
1. Run `make test` for basic DuckDB and schema validation
2. Run `make demo` for end-to-end functionality demonstration
3. Check database integrity with `make db-stats`

## Performance Considerations

- Handles up to 1M nodes and 10M edges with <100ms context retrieval
- Uses columnar storage for 10-100x compression
- Implements relevance decay (5% per day) to maintain fresh context
- Supports sampling for expensive algorithms on large graphs (>10K nodes)

## Recent Updates and Fixes (Major Refactor)

### Simplified to Hash-Based IDs
- **Removed auto-increment complexity**: All primary keys now use SHA256 hashes of natural keys
- **No foreign key constraints**: Removed FK constraints for better performance
- **Deterministic IDs**: IDs are predictable based on content (e.g., project_id = SHA256(project_path))

### Schema Simplification
- **Using default schema**: Removed custom schema to avoid naming conflicts
- **VARCHAR(64) keys everywhere**: All IDs are 64-character hex strings
- **No sequences or identity columns**: Simplified to hash-based approach

### Key Design Decisions
- **project_id**: SHA256 hash of project_path
- **file_id**: SHA256 hash of project_id + file_path
- **entity_id**: SHA256 hash of file_id + entity_name + line_start
- **edge_id**: SHA256 hash of source_id + target_id + relationship_type
- **session_id**: SHA256 hash of project_id + timestamp

### Connection Management
- **Created DuckDBManager**: Thread-safe connection manager in `db_manager.py`
- **Improved transaction handling**: Automatic transaction management for DML operations
- **Better error handling**: Proper handling of DuckDB-specific exceptions

### Best Practices Documentation
- **Added DuckDB guide**: See `.claude/DUCKDB_BEST_PRACTICES.md` for DuckDB usage patterns
- **Parameterized queries**: All queries use proper parameter binding
- **Simplified SQL splitting**: Better handling of multi-statement SQL files

### Known Issues Resolved
- Fixed "NOT NULL constraint failed" errors
- Resolved "GENERATED ALWAYS AS IDENTITY not supported" issue
- Fixed schema namespace ambiguity
- Simplified the entire ID generation system
