-- =====================================================
-- LLM Memory System - DuckDB Schema
-- =====================================================
-- Scalable knowledge graph and context management for LLMs
-- Optimized for millions of nodes and billions of edges

-- Using default schema (main) to avoid ambiguity
-- Using SHA256 hash keys based on natural keys for simplicity

-- =====================================================
-- CORE NODE TABLES (Vertices in Knowledge Graph)
-- =====================================================

-- Projects are the root nodes
CREATE TABLE IF NOT EXISTS node_project (
    project_id VARCHAR(64) PRIMARY KEY, -- SHA256 hash of project_path
    project_uuid UUID DEFAULT gen_random_uuid(),
    project_name VARCHAR NOT NULL,
    project_path VARCHAR NOT NULL UNIQUE,

    -- Metadata
    primary_language VARCHAR,
    framework VARCHAR,
    project_type VARCHAR,  -- 'web', 'cli', 'library', 'api', 'mobile', 'data'
    complexity_score DECIMAL(3,2),  -- 0.00 to 9.99

    -- Temporal
    created_at TIMESTAMPTZ DEFAULT current_timestamp,
    last_accessed TIMESTAMPTZ,
    last_modified TIMESTAMPTZ,

    -- State
    status VARCHAR DEFAULT 'active',  -- 'active', 'archived', 'deleted'
    understanding_level DECIMAL(3,2) DEFAULT 0.00,  -- 0-1 scale

    -- Structured metadata
    tech_stack VARCHAR[],
    key_patterns VARCHAR[],
    architectural_style VARCHAR,  -- 'microservices', 'monolith', 'serverless', etc

    -- Statistics
    total_files INTEGER DEFAULT 0,
    total_lines INTEGER DEFAULT 0,
    total_entities INTEGER DEFAULT 0
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_project_path ON node_project(project_path);
CREATE INDEX IF NOT EXISTS idx_project_status ON node_project(status);
CREATE INDEX IF NOT EXISTS idx_project_accessed ON node_project(last_accessed);

-- Files are primary code nodes
CREATE TABLE IF NOT EXISTS node_file (
    file_id VARCHAR(64) PRIMARY KEY, -- SHA256 hash of project_id + file_path
    project_id VARCHAR(64) NOT NULL, -- Reference to node_project

    -- Identity
    file_path VARCHAR NOT NULL,
    file_name VARCHAR NOT NULL,
    file_hash VARCHAR(64),

    -- Classification
    file_type VARCHAR,  -- 'source', 'config', 'test', 'doc', 'data'
    file_category VARCHAR,  -- 'frontend', 'backend', 'database', 'infrastructure'
    language VARCHAR,

    -- Metrics
    size_bytes BIGINT,
    lines_total INTEGER,
    lines_code INTEGER,
    lines_comment INTEGER,
    complexity_score DECIMAL(4,2),

    -- Importance scoring
    criticality_score DECIMAL(3,2) DEFAULT 0.50,  -- 0-1
    change_frequency DECIMAL(3,2) DEFAULT 0.00,  -- 0-1
    dependency_count INTEGER DEFAULT 0,

    -- Temporal
    first_seen TIMESTAMPTZ DEFAULT current_timestamp,
    last_modified TIMESTAMPTZ,
    last_analyzed TIMESTAMPTZ,

    -- Understanding
    understanding_level DECIMAL(3,2) DEFAULT 0.00,
    confidence_score DECIMAL(3,2) DEFAULT 0.50,

    UNIQUE (project_id, file_path)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_file_project ON node_file(project_id);
CREATE INDEX IF NOT EXISTS idx_file_path ON node_file(project_id, file_path);
CREATE INDEX IF NOT EXISTS idx_file_criticality ON node_file(criticality_score);

-- Code entities (classes, functions, etc)
CREATE TABLE IF NOT EXISTS node_entity (
    entity_id VARCHAR(64) PRIMARY KEY, -- SHA256 hash of file_id + entity_name + line_start
    file_id VARCHAR(64) NOT NULL, -- Reference to node_file
    parent_entity_id VARCHAR(64), -- Self-reference

    -- Identity
    entity_name VARCHAR NOT NULL,
    entity_type VARCHAR NOT NULL,  -- 'class', 'function', 'method', 'interface', 'type', 'constant'
    entity_subtype VARCHAR,  -- 'constructor', 'getter', 'setter', 'async', 'generator'

    -- Code location
    line_start INTEGER,
    line_end INTEGER,
    column_start INTEGER,
    column_end INTEGER,

    -- Properties
    visibility VARCHAR,  -- 'public', 'private', 'protected', 'internal'
    is_exported BOOLEAN DEFAULT false,
    is_async BOOLEAN DEFAULT false,
    is_static BOOLEAN DEFAULT false,
    is_abstract BOOLEAN DEFAULT false,
    is_deprecated BOOLEAN DEFAULT false,

    -- Signature and documentation
    signature TEXT,
    return_type VARCHAR,
    parameter_types VARCHAR[],
    documentation TEXT,

    -- Metrics
    complexity_score DECIMAL(4,2),
    lines_of_code INTEGER,
    parameter_count INTEGER,

    -- Usage tracking
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMPTZ
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_entity_file ON node_entity(file_id);
CREATE INDEX IF NOT EXISTS idx_entity_parent ON node_entity(parent_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_type ON node_entity(entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_usage ON node_entity(usage_count);

-- Concepts and patterns (higher-level nodes)
CREATE TABLE IF NOT EXISTS node_concept (
    concept_id VARCHAR(64) PRIMARY KEY, -- SHA256 hash of concept_name + concept_type,
    concept_name VARCHAR NOT NULL,
    concept_type VARCHAR,  -- 'pattern', 'algorithm', 'architecture', 'principle'
    concept_category VARCHAR,  -- 'creational', 'structural', 'behavioral', 'security'

    -- Description
    description TEXT,
    implementation_notes TEXT,

    -- Relationships
    related_concepts VARCHAR[],
    alternative_concepts VARCHAR[],

    -- Quality metrics
    complexity_level INTEGER,  -- 1-10
    effectiveness_score DECIMAL(3,2),  -- 0-1

    -- Usage tracking across projects
    usage_count INTEGER DEFAULT 0,
    project_ids VARCHAR(64)[]
);

CREATE INDEX IF NOT EXISTS idx_concept_type ON node_concept(concept_type);
CREATE INDEX IF NOT EXISTS idx_concept_usage ON node_concept(usage_count);

-- =====================================================
-- EDGE TABLES (Relationships in Knowledge Graph)
-- =====================================================

-- Main edge table - stores all relationships
CREATE TABLE IF NOT EXISTS edge_relationship (
    edge_id VARCHAR(64) PRIMARY KEY, -- SHA256 hash of source_id + target_id + relationship_type

    -- Source and target nodes (polymorphic)
    source_type VARCHAR NOT NULL,  -- 'file', 'entity', 'concept'
    source_id VARCHAR(64) NOT NULL,
    target_type VARCHAR NOT NULL,
    target_id VARCHAR(64) NOT NULL,

    -- Relationship details
    relationship_type VARCHAR NOT NULL,  -- 'imports', 'calls', 'extends', 'implements', etc
    relationship_subtype VARCHAR,

    -- Strength and confidence
    strength DECIMAL(3,2) DEFAULT 1.00,  -- 0-1, how strong is this relationship
    confidence DECIMAL(3,2) DEFAULT 0.50,  -- 0-1, how confident are we
    weight DECIMAL(4,2) DEFAULT 1.00,  -- For weighted graph algorithms

    -- Directionality
    is_bidirectional BOOLEAN DEFAULT false,

    -- Temporal
    discovered_at TIMESTAMPTZ DEFAULT current_timestamp,
    last_validated TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,

    -- Context
    context_data JSON  -- Additional relationship-specific data
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_edge_source ON edge_relationship(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_edge_target ON edge_relationship(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_edge_type ON edge_relationship(relationship_type);
CREATE INDEX IF NOT EXISTS idx_edge_active ON edge_relationship(is_active);
CREATE INDEX IF NOT EXISTS idx_edge_strength ON edge_relationship(strength);

-- Specialized edge table for fast traversal
CREATE TABLE IF NOT EXISTS edge_fast_lookup (
    source_id VARCHAR(64) NOT NULL,
    target_id VARCHAR(64) NOT NULL,
    relationship_type VARCHAR NOT NULL,
    distance INTEGER DEFAULT 1,  -- For multi-hop queries
    path_count INTEGER DEFAULT 1,  -- Number of paths between nodes

    PRIMARY KEY (source_id, target_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_fast_source ON edge_fast_lookup(source_id);
CREATE INDEX IF NOT EXISTS idx_fast_target ON edge_fast_lookup(target_id);

-- =====================================================
-- DIMENSION TABLES
-- =====================================================

-- Sessions dimension
CREATE TABLE IF NOT EXISTS dim_session (
    session_id VARCHAR(64) PRIMARY KEY, -- SHA256 hash of project_id + timestamp
    session_uuid UUID DEFAULT gen_random_uuid(),
    project_id VARCHAR(64) NOT NULL, -- Reference to node_project

    -- Temporal
    started_at TIMESTAMPTZ DEFAULT current_timestamp,
    ended_at TIMESTAMPTZ,
    duration_seconds INTEGER,

    -- Context
    llm_model VARCHAR,
    interaction_mode VARCHAR,  -- 'coding', 'debugging', 'reviewing', 'learning'
    initial_context_size INTEGER,
    max_context_size INTEGER,

    -- Metrics
    interactions_count INTEGER DEFAULT 0,
    tokens_consumed BIGINT DEFAULT 0,
    successful_interactions INTEGER DEFAULT 0,
    failed_interactions INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_session_project ON dim_session(project_id);
CREATE INDEX IF NOT EXISTS idx_session_started ON dim_session(started_at);

-- Tasks dimension
CREATE TABLE IF NOT EXISTS dim_task (
    task_id VARCHAR(64) PRIMARY KEY,
    project_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NOT NULL,

    -- Task details
    task_type VARCHAR,  -- 'feature', 'bugfix', 'refactor', 'test', 'documentation'
    task_description TEXT,
    task_goal TEXT,

    -- Status
    status VARCHAR DEFAULT 'pending',  -- 'pending', 'active', 'blocked', 'completed'
    priority INTEGER DEFAULT 5,  -- 1-10
    complexity_estimate INTEGER,

    -- Temporal
    created_at TIMESTAMPTZ DEFAULT current_timestamp,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    deadline_at TIMESTAMPTZ,

    -- Dependencies
    depends_on_tasks VARCHAR(64)[],
    blocks_tasks VARCHAR(64)[],

    -- Affected areas
    affected_files VARCHAR(64)[],
    affected_entities VARCHAR(64)[]
);

CREATE INDEX IF NOT EXISTS idx_task_project ON dim_task(project_id);
CREATE INDEX IF NOT EXISTS idx_task_status ON dim_task(status);
CREATE INDEX IF NOT EXISTS idx_task_priority ON dim_task(priority);

-- =====================================================
-- FACT TABLES
-- =====================================================

-- Core interaction facts
CREATE TABLE IF NOT EXISTS fact_interaction (
    interaction_id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    task_id VARCHAR(64) NOT NULL,

    -- Temporal
    timestamp TIMESTAMPTZ DEFAULT current_timestamp,
    response_time_ms INTEGER,

    -- Interaction details
    interaction_type VARCHAR,
    user_prompt TEXT,
    user_prompt_embedding FLOAT[],  -- For semantic search
    llm_response_summary TEXT,

    -- Nodes involved (for graph updates)
    nodes_read VARCHAR(64)[],
    nodes_modified VARCHAR(64)[],
    nodes_created VARCHAR(64)[],
    edges_created VARCHAR(64)[],

    -- Changes made
    files_read INTEGER DEFAULT 0,
    files_modified INTEGER DEFAULT 0,
    lines_added INTEGER DEFAULT 0,
    lines_deleted INTEGER DEFAULT 0,

    -- Quality metrics
    confidence_score DECIMAL(3,2),
    success_score DECIMAL(3,2),
    user_satisfaction INTEGER,  -- 1-5 scale

    -- Token usage
    prompt_tokens INTEGER,
    response_tokens INTEGER,
    total_tokens INTEGER,

    -- Context used
    context_strategy VARCHAR,  -- 'full', 'filtered', 'summary'
    context_nodes_included VARCHAR(64)[]
);

CREATE INDEX IF NOT EXISTS idx_interaction_session ON fact_interaction(session_id);
CREATE INDEX IF NOT EXISTS idx_interaction_timestamp ON fact_interaction(timestamp);
CREATE INDEX IF NOT EXISTS idx_interaction_type ON fact_interaction(interaction_type);

-- Code understanding facts
CREATE TABLE IF NOT EXISTS fact_understanding (
    understanding_id VARCHAR(64) PRIMARY KEY,
    node_type VARCHAR NOT NULL,  -- 'file', 'entity', 'concept'
    node_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NOT NULL,

    -- Understanding metrics
    understanding_level DECIMAL(3,2),  -- 0-1
    confidence_score DECIMAL(3,2),  -- 0-1
    completeness_score DECIMAL(3,2),  -- 0-1

    -- What we understand
    summary TEXT,
    key_insights TEXT[],
    identified_patterns TEXT[],
    potential_issues TEXT[],
    improvement_suggestions TEXT[],

    -- Dependencies understood
    dependencies_identified VARCHAR(64)[],
    dependency_depth INTEGER,

    -- Temporal
    first_analyzed TIMESTAMPTZ DEFAULT current_timestamp,
    last_updated TIMESTAMPTZ DEFAULT current_timestamp,
    times_analyzed INTEGER DEFAULT 1,

    -- Relevance management
    relevance_score DECIMAL(3,2) DEFAULT 1.00,
    staleness_factor DECIMAL(3,2) DEFAULT 0.00
);

CREATE INDEX IF NOT EXISTS idx_understanding_node ON fact_understanding(node_type, node_id);
CREATE INDEX IF NOT EXISTS idx_understanding_relevance ON fact_understanding(relevance_score);
CREATE INDEX IF NOT EXISTS idx_understanding_updated ON fact_understanding(last_updated);

-- Graph metrics facts
CREATE TABLE IF NOT EXISTS fact_graph_metrics (
    metric_id VARCHAR(64) PRIMARY KEY, -- SHA256 hash of project_id + timestamp
    project_id VARCHAR(64) NOT NULL, -- Reference to node_project
    calculated_at TIMESTAMPTZ DEFAULT current_timestamp,

    -- Graph size
    total_nodes INTEGER,
    total_edges INTEGER,
    node_types JSON,  -- Count by type
    edge_types JSON,  -- Count by type

    -- Graph properties
    average_degree DECIMAL(6,2),
    max_degree INTEGER,
    density DECIMAL(6,4),
    diameter INTEGER,

    -- Centrality measures (for top nodes)
    most_central_nodes JSON,  -- {node_id: centrality_score}
    most_connected_nodes JSON,

    -- Clustering
    clustering_coefficient DECIMAL(4,3),
    connected_components INTEGER,
    strongly_connected_components INTEGER,

    -- Patterns detected
    cycles_detected INTEGER,
    pattern_counts JSON  -- {pattern_type: count}
);

CREATE INDEX IF NOT EXISTS idx_graph_metrics_project ON fact_graph_metrics(project_id);
CREATE INDEX IF NOT EXISTS idx_graph_metrics_time ON fact_graph_metrics(calculated_at);

-- =====================================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- =====================================================

-- Pre-computed file dependencies
CREATE OR REPLACE VIEW mv_file_dependencies AS
SELECT
    f1.file_id as source_file_id,
    f1.file_path as source_path,
    f2.file_id as target_file_id,
    f2.file_path as target_path,
    COUNT(*) as dependency_count,
    MAX(e.strength) as max_strength,
    ARRAY_AGG(DISTINCT e.relationship_type) as relationship_types
FROM node_file f1
JOIN edge_relationship e
    ON e.source_type = 'file' AND e.source_id = f1.file_id
JOIN node_file f2
    ON e.target_type = 'file' AND e.target_id = f2.file_id
WHERE e.is_active = true
GROUP BY f1.file_id, f1.file_path, f2.file_id, f2.file_path;

-- Pre-computed entity call graph
CREATE OR REPLACE VIEW mv_call_graph AS
SELECT
    e1.entity_id as caller_id,
    e1.entity_name as caller_name,
    e2.entity_id as callee_id,
    e2.entity_name as callee_name,
    er.strength,
    er.confidence
FROM node_entity e1
JOIN edge_relationship er
    ON er.source_type = 'entity'
    AND er.source_id = e1.entity_id
    AND er.relationship_type = 'calls'
JOIN node_entity e2
    ON er.target_type = 'entity'
    AND er.target_id = e2.entity_id
WHERE er.is_active = true;

-- Hot spots (frequently modified areas)
CREATE OR REPLACE VIEW mv_hot_spots AS
SELECT
    f.file_id,
    f.file_path,
    COUNT(DISTINCT fi.interaction_id) as interaction_count,
    SUM(fi.lines_added + fi.lines_deleted) as total_changes,
    MAX(fi.timestamp) as last_modified,
    AVG(fi.success_score) as avg_success_score
FROM node_file f
JOIN fact_interaction fi
    ON f.file_id = ANY(fi.nodes_modified)
GROUP BY f.file_id, f.file_path
HAVING COUNT(DISTINCT fi.interaction_id) > 3
ORDER BY interaction_count DESC;

-- =====================================================
-- GRAPH TRAVERSAL VIEWS (Functions not supported in DuckDB)
-- =====================================================

-- Create a view for dependencies that can be filtered
CREATE OR REPLACE VIEW v_dependencies AS
WITH RECURSIVE dependency_graph AS (
    -- Base case
    SELECT
        e.source_id,
        e.target_id as node_id,
        1 as depth
    FROM edge_relationship e
    WHERE e.is_active = true

    UNION ALL

    -- Recursive case  
    SELECT
        dg.source_id,
        e.target_id as node_id,
        dg.depth + 1 as depth
    FROM dependency_graph dg
    JOIN edge_relationship e
        ON e.source_id = dg.node_id
    WHERE dg.depth < 5
      AND e.is_active = true
)
SELECT
    source_id,
    node_id,
    MIN(depth) as depth,
    COUNT(*) as path_count
FROM dependency_graph
GROUP BY source_id, node_id;

-- View for analyzing impact of changes
CREATE OR REPLACE VIEW v_impact_analysis AS
SELECT
    e.target_id,
    e.target_type,
    e.source_id as impacted_node_id,
    e.source_type as impacted_node_type,
    e.strength * e.confidence as impact_score,
    e.relationship_type as impact_reason
FROM edge_relationship e
WHERE e.is_active = true
  AND e.relationship_type IN ('depends_on', 'calls', 'imports', 'extends');

-- View for path finding (simplified)
CREATE OR REPLACE VIEW v_paths AS
WITH RECURSIVE path_search AS (
    -- Start from all nodes
    SELECT
        e.source_id as start_node,
        e.target_id as end_node,
        ARRAY[e.source_id, e.target_id] as path,
        (2.0 - e.strength) as distance
    FROM edge_relationship e
    WHERE e.is_active = true

    UNION ALL

    -- Extend paths
    SELECT
        ps.start_node,
        e.target_id as end_node,
        ARRAY_APPEND(ps.path, e.target_id) as path,
        ps.distance + (2.0 - e.strength) as distance
    FROM path_search ps
    JOIN edge_relationship e
        ON e.source_id = ps.end_node
    WHERE NOT ARRAY_CONTAINS(ps.path, e.target_id)
      AND e.is_active = true
      AND ARRAY_LENGTH(ps.path) < 5
)
SELECT
    start_node,
    end_node,
    path,
    distance
FROM path_search;

-- View for detecting circular dependencies
CREATE OR REPLACE VIEW v_cycles AS
WITH RECURSIVE cycle_search AS (
    -- Start from each node
    SELECT
        e.source_id as start_node,
        ARRAY[e.source_id, e.target_id] as path,
        e.target_id as current_node,
        e.relationship_type,
        f.project_id
    FROM edge_relationship e
    JOIN node_file f ON f.file_id = e.source_id
    WHERE e.source_type = 'file'
      AND e.is_active = true

    UNION ALL

    -- Continue path
    SELECT
        cs.start_node,
        ARRAY_APPEND(cs.path, e.target_id) as path,
        e.target_id as current_node,
        cs.relationship_type,
        cs.project_id
    FROM cycle_search cs
    JOIN edge_relationship e
        ON e.source_id = cs.current_node
        AND e.relationship_type = cs.relationship_type
    WHERE NOT ARRAY_CONTAINS(cs.path[2:], e.target_id)  -- Can revisit start
      AND ARRAY_LENGTH(cs.path) < 20  -- Prevent infinite loops
      AND e.is_active = true
)
SELECT
    project_id,
    path as cycle_nodes,
    relationship_type as cycle_type
FROM cycle_search
WHERE current_node = start_node
  AND ARRAY_LENGTH(path) > 2;

-- =====================================================
-- CONTEXT RETRIEVAL VIEWS
-- =====================================================

-- View for optimal context retrieval
CREATE OR REPLACE VIEW v_context AS
WITH ranked_context AS (
    -- Active tasks
    SELECT
        project_id,
        'active_task' as context_type,
        'Task: ' || task_type || ' - ' || task_description as content,
        0.95 as relevance_score,
        LENGTH(task_description) / 4 as token_estimate,
        affected_files as node_ids
    FROM dim_task
    WHERE status IN ('active', 'blocked')

    UNION ALL

    -- Recent interactions
    SELECT
        s.project_id,
        'recent_interaction' as context_type,
        'Previous: ' || SUBSTRING(user_prompt, 1, 200) || ' -> ' || SUBSTRING(llm_response_summary, 1, 200) as content,
        0.85 as relevance_score,
        LENGTH(user_prompt) / 4 + LENGTH(llm_response_summary) / 4 as token_estimate,
        nodes_modified as node_ids
    FROM fact_interaction fi
    JOIN dim_session s ON fi.session_id = s.session_id

    UNION ALL

    -- Critical file understanding
    SELECT
        f.project_id,
        'file_understanding' as context_type,
        fu.summary as content,
        fu.relevance_score * f.criticality_score as relevance_score,
        LENGTH(fu.summary) / 4 as token_estimate,
        ARRAY[f.file_id] as node_ids
    FROM fact_understanding fu
    JOIN node_file f
        ON fu.node_type = 'file'
        AND fu.node_id = f.file_id
    WHERE f.criticality_score > 0.7
      AND fu.staleness_factor < 0.3

    UNION ALL

    -- Hot spots
    SELECT
        nf.project_id,
        'hot_spot' as context_type,
        'Frequently modified: ' || hs.file_path || ' (' || hs.interaction_count || ' changes)' as content,
        0.75 as relevance_score,
        50 as token_estimate,
        ARRAY[hs.file_id] as node_ids
    FROM mv_hot_spots hs
    JOIN node_file nf ON hs.file_id = nf.file_id
),
cumulative_tokens AS (
    SELECT
        *,
        SUM(token_estimate) OVER (PARTITION BY project_id ORDER BY relevance_score DESC) as running_total
    FROM ranked_context
)
SELECT
    project_id,
    context_type,
    content,
    relevance_score,
    token_estimate,
    node_ids,
    running_total
FROM cumulative_tokens
ORDER BY project_id, relevance_score DESC;

-- =====================================================
-- MAINTENANCE VIEWS (Instead of procedures)
-- =====================================================

-- View for graph metrics
CREATE OR REPLACE VIEW v_graph_metrics AS
SELECT
    f.project_id,
    COUNT(DISTINCT f.file_id) as total_nodes,
    COUNT(DISTINCT e.edge_id) as total_edges,
    CASE 
        WHEN COUNT(DISTINCT f.file_id) > 0 
        THEN (COUNT(DISTINCT e.edge_id)::DECIMAL / COUNT(DISTINCT f.file_id)) * 2
        ELSE 0
    END as average_degree
FROM node_file f
LEFT JOIN edge_relationship e
    ON e.source_id = f.file_id 
    AND e.source_type = 'file'
    AND e.is_active = true
GROUP BY f.project_id;

-- View for understanding with decayed relevance
CREATE OR REPLACE VIEW v_understanding_with_decay AS
SELECT
    *,
    CASE
        WHEN last_updated < CURRENT_TIMESTAMP - INTERVAL '7 days'
        THEN GREATEST(0.1, relevance_score * 0.95)
        ELSE relevance_score
    END as decayed_relevance_score,
    CASE
        WHEN last_updated < CURRENT_TIMESTAMP - INTERVAL '7 days'
        THEN LEAST(1.0, staleness_factor + 0.05)
        ELSE staleness_factor
    END as decayed_staleness_factor
FROM fact_understanding;

-- End of schema
