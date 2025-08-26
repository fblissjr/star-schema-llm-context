#!/usr/bin/env python3
"""
MCP Server for LLM Memory Management with DuckDB Backend
Scalable knowledge graph and context management
"""

import asyncio
import json
import hashlib
import duckdb
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
from collections import defaultdict
import pickle
import ast
import re
from db_manager import DuckDBManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# Data Models
# =====================================================

class NodeType(Enum):
    """Types of nodes in the knowledge graph"""
    PROJECT = "project"
    FILE = "file"
    ENTITY = "entity"
    CONCEPT = "concept"
    TASK = "task"

class EdgeType(Enum):
    """Types of edges in the knowledge graph"""
    # Syntactic relationships
    IMPORTS = "imports"
    CALLS = "calls"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    INSTANTIATES = "instantiates"
    CONTAINS = "contains"
    
    # Semantic relationships  
    SIMILAR_TO = "similar_to"
    ALTERNATIVE_TO = "alternative_to"
    REPLACES = "replaces"
    CONFLICTS_WITH = "conflicts_with"
    DEPENDS_ON = "depends_on"
    
    # Temporal relationships
    MODIFIED_BEFORE = "modified_before"
    CREATED_WITH = "created_with"
    DEPRECATED_BY = "deprecated_by"

@dataclass
class Node:
    """Represents a node in the knowledge graph"""
    node_id: int
    node_type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    
@dataclass
class Edge:
    """Represents an edge in the knowledge graph"""
    edge_id: int
    source_node: Node
    target_node: Node
    edge_type: EdgeType
    strength: float = 1.0
    confidence: float = 0.5
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphContext:
    """Encapsulates graph context for LLM"""
    nodes: List[Node]
    edges: List[Edge]
    subgraph_metrics: Dict[str, Any]
    relevance_scores: Dict[int, float]
    token_count: int

# =====================================================
# Knowledge Graph Engine
# =====================================================

class KnowledgeGraphEngine:
    """
    Manages the knowledge graph with DuckDB backend
    Handles millions of nodes and billions of edges efficiently
    """
    
    def __init__(self, db_path: str = "data/llm_memory.duckdb"):
        self.db_path = db_path
        self.db = DuckDBManager(db_path)
        self.conn = self.db._get_connection()  # Keep conn for compatibility
        self.cache = {}
        self.init_database()
        
    def init_database(self):
        """Initialize DuckDB connection and schema"""
        # Load the schema
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
                # Execute schema creation statements
                try:
                    self.db.execute(schema_sql)
                except duckdb.CatalogException as e:
                    # Schema might already exist
                    logger.debug(f"Schema setup note: {e}")
        
        logger.info(f"Initialized DuckDB at {self.db_path}")
    
    async def add_node(self, 
                       node_type: NodeType,
                       name: str,
                       properties: Dict[str, Any]) -> Node:
        """Add a node to the graph"""
        
        if node_type == NodeType.FILE:
            # Generate hash ID from project_id + file_path
            project_id = properties.get('project_id')
            file_path = properties.get('file_path', name)
            file_id = hashlib.sha256(f"{project_id}:{file_path}".encode()).hexdigest()
            
            # Insert file node
            self.conn.execute("""
                INSERT INTO node_file 
                (file_id, project_id, file_path, file_name, file_type, language, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                file_id,
                project_id,
                file_path,
                name,
                properties.get('file_type', 'source'),
                properties.get('language'),
                properties.get('size_bytes', 0)
            ])
            
            node_id = file_id
            
        elif node_type == NodeType.ENTITY:
            # Generate hash ID from file_id + entity_name + line_start
            file_id = properties.get('file_id')
            line_start = properties.get('line_start', 0)
            entity_id = hashlib.sha256(f"{file_id}:{name}:{line_start}".encode()).hexdigest()
            
            # Insert entity node
            self.conn.execute("""
                INSERT INTO node_entity
                (entity_id, file_id, entity_name, entity_type, line_start, line_end, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                entity_id,
                file_id,
                name,
                properties.get('entity_type', 'function'),
                line_start,
                properties.get('line_end'),
                properties.get('signature')
            ])
            
            node_id = entity_id
            
        else:
            # Generic node insertion
            node_id = hash(f"{node_type.value}:{name}") % (10**8)
        
        return Node(
            node_id=node_id,
            node_type=node_type,
            name=name,
            properties=properties
        )
    
    async def add_edge(self,
                      source_node: Node,
                      target_node: Node,
                      edge_type: EdgeType,
                      strength: float = 1.0,
                      confidence: float = 0.5) -> Edge:
        """Add an edge to the graph"""
        
        # Generate hash ID for edge
        edge_id = hashlib.sha256(
            f"{source_node.node_id}:{target_node.node_id}:{edge_type.value}".encode()
        ).hexdigest()
        
        # Insert edge
        self.db.execute("""
            INSERT INTO edge_relationship
            (edge_id, source_type, source_id, target_type, target_id, 
             relationship_type, strength, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            edge_id,
            source_node.node_type.value,
            source_node.node_id,
            target_node.node_type.value,
            target_node.node_id,
            edge_type.value,
            strength,
            confidence
        ])
        
        # Also update fast lookup table
        self.db.execute("""
            INSERT OR REPLACE INTO edge_fast_lookup
            (source_id, target_id, relationship_type)
            VALUES (?, ?, ?)
        """, [source_node.node_id, target_node.node_id, edge_type.value])
        
        return Edge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            edge_type=edge_type,
            strength=strength,
            confidence=confidence
        )
    
    async def find_neighbors(self,
                           node_id: int,
                           edge_types: Optional[List[EdgeType]] = None,
                           max_distance: int = 1) -> List[Tuple[Node, float]]:
        """Find neighboring nodes within max_distance hops"""
        
        edge_filter = ""
        if edge_types:
            types_str = "','".join([e.value for e in edge_types])
            edge_filter = f"AND relationship_type IN ('{types_str}')"
        
        results = self.conn.execute(f"""
            WITH RECURSIVE neighbors AS (
                -- Direct neighbors
                SELECT 
                    target_id as node_id,
                    target_type as node_type,
                    strength,
                    1 as distance
                FROM edge_relationship
                WHERE source_id = ? {edge_filter}
                  AND is_active = true
                
                UNION ALL
                
                -- Recursive neighbors
                SELECT 
                    e.target_id as node_id,
                    e.target_type as node_type,
                    n.strength * e.strength as strength,
                    n.distance + 1 as distance
                FROM neighbors n
                JOIN edge_relationship e
                    ON e.source_id = n.node_id
                WHERE n.distance < ?
                  AND e.is_active = true
                  {edge_filter}
            )
            SELECT DISTINCT
                node_id,
                node_type,
                MAX(strength) as max_strength,
                MIN(distance) as min_distance
            FROM neighbors
            GROUP BY node_id, node_type
            ORDER BY min_distance, max_strength DESC
        """, [node_id, max_distance]).fetchall()
        
        neighbors = []
        for row in results:
            # Create node object (simplified)
            node = Node(
                node_id=row[0],
                node_type=NodeType(row[1]),
                name=f"Node_{row[0]}",  # Would fetch actual name
                properties={}
            )
            neighbors.append((node, row[2]))  # node, strength
        
        return neighbors
    
    async def find_path(self,
                       source_id: int,
                       target_id: int,
                       max_length: int = 5) -> Optional[List[Node]]:
        """Find shortest path between two nodes"""
        
        # Simplified path finding - would need proper implementation
        result = None
        
        if result and result[0]:
            path_ids = result[0]
            path_nodes = []
            
            for node_id in path_ids:
                # Fetch node details (simplified)
                node = Node(
                    node_id=node_id,
                    node_type=NodeType.FILE,  # Would determine actual type
                    name=f"Node_{node_id}",
                    properties={}
                )
                path_nodes.append(node)
            
            return path_nodes
        
        return None
    
    async def detect_patterns(self,
                            project_id: int,
                            pattern_types: List[str]) -> Dict[str, List]:
        """Detect patterns in the graph"""
        
        patterns = {}
        
        if "circular_dependency" in pattern_types:
            # Detect circular dependencies
            # Simplified cycle detection - would need proper implementation
            cycles = []
            
            patterns["circular_dependencies"] = [
                {
                    "nodes": cycle[0],
                    "type": cycle[1]
                }
                for cycle in cycles
            ]
        
        if "god_class" in pattern_types:
            # Detect god classes (too many dependencies)
            god_classes = self.conn.execute("""
                SELECT 
                    e.entity_id,
                    e.entity_name,
                    COUNT(DISTINCT er.target_id) as dependency_count
                FROM node_entity e
                JOIN edge_relationship er
                    ON er.source_id = e.entity_id
                    AND er.source_type = 'entity'
                WHERE e.entity_type = 'class'
                GROUP BY e.entity_id, e.entity_name
                HAVING COUNT(DISTINCT er.target_id) > 20
                ORDER BY dependency_count DESC
            """).fetchall()
            
            patterns["god_classes"] = [
                {
                    "entity_id": gc[0],
                    "name": gc[1],
                    "dependencies": gc[2]
                }
                for gc in god_classes
            ]
        
        return patterns
    
    async def calculate_centrality(self,
                                  project_id: int,
                                  centrality_type: str = "degree") -> Dict[int, float]:
        """Calculate node centrality measures"""
        
        if centrality_type == "degree":
            # Degree centrality (number of connections)
            results = self.conn.execute("""
                SELECT 
                    node_id,
                    in_degree + out_degree as total_degree
                FROM (
                    SELECT 
                        f.file_id as node_id,
                        COUNT(DISTINCT e1.edge_id) as out_degree,
                        COUNT(DISTINCT e2.edge_id) as in_degree
                    FROM node_file f
                    LEFT JOIN edge_relationship e1
                        ON e1.source_id = f.file_id AND e1.source_type = 'file'
                    LEFT JOIN edge_relationship e2
                        ON e2.target_id = f.file_id AND e2.target_type = 'file'
                    WHERE f.project_id = ?
                    GROUP BY f.file_id
                ) degree_calc
                ORDER BY total_degree DESC
            """, [project_id]).fetchall()
            
            max_degree = max(r[1] for r in results) if results else 1
            return {r[0]: r[1] / max_degree for r in results}
        
        elif centrality_type == "betweenness":
            # Simplified betweenness centrality
            # (In practice, would use more sophisticated algorithm)
            return {}
        
        return {}

# =====================================================
# Context Management Engine
# =====================================================

class ContextManager:
    """
    Manages context optimization and retrieval for LLMs
    """
    
    def __init__(self, graph_engine: KnowledgeGraphEngine):
        self.graph = graph_engine
        self.context_cache = {}
        self.token_estimator = lambda text: len(text) // 4
        
    async def build_context(self,
                          project_id: int,
                          max_tokens: int = 100000,
                          focus_nodes: Optional[List[int]] = None) -> GraphContext:
        """Build optimized context for LLM"""
        
        # Get context using the SQL function
        # Simplified context retrieval - would need proper implementation
        context_items = []
        
        nodes = []
        edges = []
        relevance_scores = {}
        total_tokens = 0
        
        for item in context_items:
            context_type, content, relevance, tokens, node_ids = item
            
            # Track tokens
            total_tokens += tokens
            
            # Extract nodes from context
            if node_ids:
                for node_id in node_ids:
                    if node_id not in relevance_scores:
                        relevance_scores[node_id] = relevance
                        
                        # Create simplified node
                        node = Node(
                            node_id=node_id,
                            node_type=NodeType.FILE,  # Would determine actual type
                            name=f"Node_{node_id}",
                            properties={"content": content}
                        )
                        nodes.append(node)
        
        # Get edges between included nodes
        if nodes:
            node_ids = [n.node_id for n in nodes]
            edges_data = self.graph.conn.execute("""
                SELECT 
                    edge_id, source_id, target_id, 
                    relationship_type, strength, confidence
                FROM edge_relationship
                WHERE source_id = ANY(?) AND target_id = ANY(?)
                  AND is_active = true
                ORDER BY strength DESC
                LIMIT 100
            """, [node_ids, node_ids]).fetchall()
            
            # Create edge objects
            node_map = {n.node_id: n for n in nodes}
            for edge_data in edges_data:
                if edge_data[1] in node_map and edge_data[2] in node_map:
                    edge = Edge(
                        edge_id=edge_data[0],
                        source_node=node_map[edge_data[1]],
                        target_node=node_map[edge_data[2]],
                        edge_type=EdgeType(edge_data[3]),
                        strength=edge_data[4],
                        confidence=edge_data[5]
                    )
                    edges.append(edge)
        
        # Calculate subgraph metrics
        subgraph_metrics = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "density": len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0,
            "avg_relevance": sum(relevance_scores.values()) / len(relevance_scores) if relevance_scores else 0
        }
        
        return GraphContext(
            nodes=nodes,
            edges=edges,
            subgraph_metrics=subgraph_metrics,
            relevance_scores=relevance_scores,
            token_count=total_tokens
        )
    
    async def update_relevance(self,
                              node_ids: List[int],
                              boost: float = 0.1):
        """Update relevance scores for accessed nodes"""
        
        for node_id in node_ids:
            self.graph.conn.execute("""
                UPDATE fact_understanding
                SET relevance_score = LEAST(1.0, relevance_score + ?),
                    last_updated = CURRENT_TIMESTAMP
                WHERE node_id = ? AND node_type = 'file'
            """, [boost, node_id])
    
    async def compress_context(self,
                             context: GraphContext,
                             target_tokens: int) -> GraphContext:
        """Compress context to fit within token limit"""
        
        if context.token_count <= target_tokens:
            return context
        
        # Sort nodes by relevance
        sorted_nodes = sorted(
            context.nodes,
            key=lambda n: context.relevance_scores.get(n.node_id, 0),
            reverse=True
        )
        
        # Keep only most relevant nodes
        compressed_nodes = []
        current_tokens = 0
        
        for node in sorted_nodes:
            node_tokens = self.token_estimator(str(node.properties))
            if current_tokens + node_tokens <= target_tokens:
                compressed_nodes.append(node)
                current_tokens += node_tokens
            else:
                break
        
        # Filter edges
        node_ids = {n.node_id for n in compressed_nodes}
        compressed_edges = [
            e for e in context.edges
            if e.source_node.node_id in node_ids 
            and e.target_node.node_id in node_ids
        ]
        
        return GraphContext(
            nodes=compressed_nodes,
            edges=compressed_edges,
            subgraph_metrics=context.subgraph_metrics,
            relevance_scores=context.relevance_scores,
            token_count=current_tokens
        )

# =====================================================
# Code Analysis Engine
# =====================================================

class CodeAnalyzer:
    """
    Analyzes code to build and update the knowledge graph
    """
    
    def __init__(self, graph_engine: KnowledgeGraphEngine):
        self.graph = graph_engine
        
    async def analyze_file(self,
                         file_path: str,
                         project_id: int) -> Dict[str, Any]:
        """Analyze a code file and update the graph"""
        
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return {"error": "File not found"}
        
        # Basic file properties
        file_stats = file_path_obj.stat()
        file_content = file_path_obj.read_text()
        
        # Create or update file node
        file_node = await self.graph.add_node(
            node_type=NodeType.FILE,
            name=file_path_obj.name,
            properties={
                "project_id": project_id,
                "file_path": str(file_path_obj),
                "file_type": self._determine_file_type(file_path_obj),
                "language": self._detect_language(file_path_obj),
                "size_bytes": file_stats.st_size,
                "lines": len(file_content.splitlines())
            }
        )
        
        # Parse file based on language
        if file_path_obj.suffix == '.py':
            entities = await self._parse_python(file_content, file_node)
        elif file_path_obj.suffix in ['.js', '.ts']:
            entities = await self._parse_javascript(file_content, file_node)
        else:
            entities = []
        
        # Detect imports and create edges
        imports = self._detect_imports(file_content, file_path_obj.suffix)
        for import_path in imports:
            # Create edge for import relationship
            # (In real implementation, would resolve import to actual file node)
            pass
        
        return {
            "file_id": file_node.node_id,
            "entities_found": len(entities),
            "imports_found": len(imports)
        }
    
    async def _parse_python(self,
                          content: str,
                          file_node: Node) -> List[Node]:
        """Parse Python code and extract entities"""
        
        entities = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                entity_node = None
                
                if isinstance(node, ast.FunctionDef):
                    # Function entity
                    entity_node = await self.graph.add_node(
                        node_type=NodeType.ENTITY,
                        name=node.name,
                        properties={
                            "file_id": file_node.node_id,
                            "entity_type": "function",
                            "line_start": node.lineno,
                            "line_end": node.end_lineno,
                            "signature": ast.unparse(node.args) if hasattr(ast, 'unparse') else str(node.args)
                        }
                    )
                    
                elif isinstance(node, ast.ClassDef):
                    # Class entity
                    entity_node = await self.graph.add_node(
                        node_type=NodeType.ENTITY,
                        name=node.name,
                        properties={
                            "file_id": file_node.node_id,
                            "entity_type": "class",
                            "line_start": node.lineno,
                            "line_end": node.end_lineno
                        }
                    )
                
                if entity_node:
                    entities.append(entity_node)
                    
                    # Create contains edge
                    await self.graph.add_edge(
                        source_node=file_node,
                        target_node=entity_node,
                        edge_type=EdgeType.CONTAINS
                    )
        
        except SyntaxError:
            logger.warning(f"Failed to parse Python file: {file_node.name}")
        
        return entities
    
    async def _parse_javascript(self,
                              content: str,
                              file_node: Node) -> List[Node]:
        """Parse JavaScript/TypeScript code"""
        # Simplified parsing using regex
        # In production, would use proper parser like esprima
        
        entities = []
        
        # Find functions
        func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|\([^)]*\)\s*{))'
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1) or match.group(2)
            if func_name:
                entity_node = await self.graph.add_node(
                    node_type=NodeType.ENTITY,
                    name=func_name,
                    properties={
                        "file_id": file_node.node_id,
                        "entity_type": "function",
                        "line_start": content[:match.start()].count('\n') + 1
                    }
                )
                entities.append(entity_node)
        
        # Find classes
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            entity_node = await self.graph.add_node(
                node_type=NodeType.ENTITY,
                name=class_name,
                properties={
                    "file_id": file_node.node_id,
                    "entity_type": "class",
                    "line_start": content[:match.start()].count('\n') + 1
                }
            )
            entities.append(entity_node)
        
        return entities
    
    def _determine_file_type(self, file_path: Path) -> str:
        """Determine the type of file"""
        
        if file_path.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.c']:
            return 'source'
        elif file_path.suffix in ['.json', '.yaml', '.yml', '.toml']:
            return 'config'
        elif file_path.suffix in ['.md', '.rst', '.txt']:
            return 'documentation'
        elif 'test' in file_path.name.lower():
            return 'test'
        else:
            return 'other'
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php'
        }
        
        return ext_to_lang.get(file_path.suffix, 'unknown')
    
    def _detect_imports(self, content: str, extension: str) -> List[str]:
        """Detect import statements"""
        
        imports = []
        
        if extension == '.py':
            # Python imports
            import_pattern = r'(?:from\s+([\w.]+)|import\s+([\w.,\s]+))'
            for match in re.finditer(import_pattern, content):
                if match.group(1):
                    imports.append(match.group(1))
                elif match.group(2):
                    imports.extend(m.strip() for m in match.group(2).split(','))
                    
        elif extension in ['.js', '.ts']:
            # JavaScript/TypeScript imports
            import_pattern = r'(?:import\s+.*?\s+from\s+[\'"]([^\'"]]+)|require\([\'"]([^\'"]]+))'
            for match in re.finditer(import_pattern, content):
                imports.append(match.group(1) or match.group(2))
        
        return imports

# =====================================================
# MCP Server Implementation
# =====================================================

class LLMMemoryMCPServer:
    """
    MCP Server for LLM memory management with DuckDB backend
    """
    
    def __init__(self, 
                 db_path: str = "data/llm_memory.duckdb",
                 max_context_tokens: int = 100000):
        
        self.db_path = db_path
        self.max_context_tokens = max_context_tokens
        
        # Initialize engines
        self.graph_engine = KnowledgeGraphEngine(db_path)
        self.context_manager = ContextManager(self.graph_engine)
        self.code_analyzer = CodeAnalyzer(self.graph_engine)
        
        # Session management
        self.current_project_id = None
        self.current_session_id = None
        
        logger.info("MCP Server initialized with DuckDB backend")
    
    async def initialize_project(self, project_path: str) -> Dict[str, Any]:
        """Initialize or load a project"""
        
        # Use hash of project path as project ID
        project_id = hashlib.sha256(project_path.encode()).hexdigest()
        
        # Check if project exists
        existing = self.graph_engine.db.fetch_one("""
            SELECT project_id, understanding_level 
            FROM node_project 
            WHERE project_id = ?
        """, [project_id])
        
        if existing:
            self.current_project_id = existing[0]
            understanding_level = existing[1]
            
            # Update last accessed
            self.graph_engine.db.execute("""
                UPDATE node_project 
                SET last_accessed = CURRENT_TIMESTAMP
                WHERE project_id = ?
            """, [self.current_project_id])
            
        else:
            # Create new project
            self.current_project_id = project_id
            self.graph_engine.db.execute("""
                INSERT INTO node_project
                (project_id, project_name, project_path)
                VALUES (?, ?, ?)
            """, [project_id, Path(project_path).name, project_path])
            understanding_level = 0.0
        
        # Create new session with hash ID
        session_id = hashlib.sha256(
            f"{self.current_project_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()
        self.current_session_id = session_id
        
        self.graph_engine.db.execute("""
            INSERT INTO dim_session
            (session_id, project_id, llm_model)
            VALUES (?, ?, ?)
        """, [session_id, self.current_project_id, "claude-3"])
        
        # Get initial statistics
        stats = self.graph_engine.db.fetch_one("""
            SELECT 
                COUNT(DISTINCT f.file_id) as file_count,
                COUNT(DISTINCT e.entity_id) as entity_count,
                COUNT(DISTINCT er.edge_id) as edge_count
            FROM node_project p
            LEFT JOIN node_file f ON f.project_id = p.project_id
            LEFT JOIN node_entity e ON e.file_id = f.file_id
            LEFT JOIN edge_relationship er 
                ON (er.source_id = f.file_id AND er.source_type = 'file')
                OR (er.source_id = e.entity_id AND er.source_type = 'entity')
            WHERE p.project_id = ?
        """, [self.current_project_id])
        
        return {
            "status": "initialized",
            "project_id": self.current_project_id,
            "session_id": self.current_session_id,
            "understanding_level": float(understanding_level),
            "statistics": {
                "files": stats[0] if stats else 0,
                "entities": stats[1] if stats else 0,
                "relationships": stats[2] if stats else 0
            }
        }
    
    async def get_context(self, 
                        max_tokens: Optional[int] = None,
                        focus_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get optimized context for current session"""
        
        if not self.current_project_id:
            return {"error": "No project initialized"}
        
        max_tokens = max_tokens or self.max_context_tokens
        
        # Build graph context
        context = await self.context_manager.build_context(
            project_id=self.current_project_id,
            max_tokens=max_tokens,
            focus_nodes=None  # Would convert focus_files to node IDs
        )
        
        # Compress if needed
        if context.token_count > max_tokens:
            context = await self.context_manager.compress_context(
                context, max_tokens
            )
        
        # Format for LLM
        formatted_context = self._format_context_for_llm(context)
        
        return {
            "context": formatted_context,
            "metrics": {
                "nodes_included": len(context.nodes),
                "edges_included": len(context.edges),
                "token_count": context.token_count,
                "avg_relevance": context.subgraph_metrics.get("avg_relevance", 0)
            }
        }
    
    async def analyze_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze files and update the knowledge graph"""
        
        if not self.current_project_id:
            return {"error": "No project initialized"}
        
        results = []
        
        for file_path in file_paths:
            result = await self.code_analyzer.analyze_file(
                file_path, self.current_project_id
            )
            results.append(result)
        
        # Graph metrics would be updated here if we had the function
        # For now, just skip this step
        
        return {
            "files_analyzed": len(results),
            "results": results
        }
    
    async def find_dependencies(self,
                              file_path: str,
                              max_depth: int = 3) -> Dict[str, Any]:
        """Find all dependencies of a file"""
        
        # Get file node
        file_data = self.graph_engine.conn.execute("""
            SELECT file_id FROM node_file
            WHERE project_id = ? AND file_path = ?
        """, [self.current_project_id, file_path]).fetchone()
        
        if not file_data:
            return {"error": "File not found"}
        
        file_id = file_data[0]
        
        # Find dependencies using SQL function
        # Simplified dependency finding - would need proper implementation
        deps = []
        
        return {
            "file": file_path,
            "dependencies": [
                {
                    "node_id": d[0],
                    "depth": d[1],
                    "path_count": d[2]
                }
                for d in deps
            ]
        }
    
    async def analyze_impact(self, file_path: str) -> Dict[str, Any]:
        """Analyze the impact of changing a file"""
        
        # Get file node
        file_data = self.graph_engine.conn.execute("""
            SELECT file_id FROM node_file
            WHERE project_id = ? AND file_path = ?
        """, [self.current_project_id, file_path]).fetchone()
        
        if not file_data:
            return {"error": "File not found"}
        
        file_id = file_data[0]
        
        # Analyze impact
        # Simplified impact analysis - would need proper implementation
        impacts = []
        
        return {
            "file": file_path,
            "impacted_nodes": [
                {
                    "node_id": i[0],
                    "node_type": i[1],
                    "impact_score": float(i[2]),
                    "reason": i[3]
                }
                for i in impacts
            ]
        }
    
    async def detect_issues(self) -> Dict[str, Any]:
        """Detect issues in the codebase"""
        
        if not self.current_project_id:
            return {"error": "No project initialized"}
        
        # Detect various patterns
        patterns = await self.graph_engine.detect_patterns(
            self.current_project_id,
            ["circular_dependency", "god_class"]
        )
        
        return {
            "issues_found": sum(len(v) for v in patterns.values()),
            "patterns": patterns
        }
    
    async def save_interaction(self,
                             interaction_type: str,
                             user_prompt: str,
                             response: str,
                             files_involved: List[str],
                             success: bool = True) -> Dict[str, Any]:
        """Save an interaction to the graph"""
        
        # Get file IDs
        file_ids = []
        for file_path in files_involved:
            file_data = self.graph_engine.conn.execute("""
                SELECT file_id FROM node_file
                WHERE project_id = ? AND file_path = ?
            """, [self.current_project_id, file_path]).fetchone()
            
            if file_data:
                file_ids.append(file_data[0])
        
        # Save interaction
        result = self.graph_engine.conn.execute("""
            INSERT INTO fact_interaction
            (session_id, interaction_type, user_prompt, llm_response_summary,
             nodes_modified, prompt_tokens, response_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING interaction_id
        """, [
            self.current_session_id,
            interaction_type,
            user_prompt,
            response[:1000],  # Summary only
            file_ids,
            len(user_prompt) // 4,
            len(response) // 4
        ]).fetchone()
        
        # Update relevance for involved files
        await self.context_manager.update_relevance(file_ids, boost=0.1)
        
        return {
            "interaction_id": result[0],
            "files_updated": len(file_ids)
        }
    
    def _format_context_for_llm(self, context: GraphContext) -> str:
        """Format graph context as text for LLM"""
        
        sections = []
        
        # Add node information
        if context.nodes:
            node_section = "=== Key Code Elements ===\n"
            for node in context.nodes[:20]:  # Limit to top 20
                relevance = context.relevance_scores.get(node.node_id, 0)
                node_section += f"- {node.name} ({node.node_type.value}) [relevance: {relevance:.2f}]\n"
                if 'content' in node.properties:
                    node_section += f"  {node.properties['content'][:100]}...\n"
            sections.append(node_section)
        
        # Add relationship information
        if context.edges:
            edge_section = "=== Key Relationships ===\n"
            for edge in context.edges[:15]:  # Limit to top 15
                edge_section += f"- {edge.source_node.name} --{edge.edge_type.value}--> {edge.target_node.name} "
                edge_section += f"[strength: {edge.strength:.2f}]\n"
            sections.append(edge_section)
        
        # Add metrics
        metrics_section = f"""=== Context Metrics ===
Nodes: {context.subgraph_metrics['node_count']}
Edges: {context.subgraph_metrics['edge_count']}
Density: {context.subgraph_metrics['density']:.3f}
Average Relevance: {context.subgraph_metrics['avg_relevance']:.2f}
Token Count: {context.token_count}
"""
        sections.append(metrics_section)
        
        return "\n\n".join(sections)

# =====================================================
# Main Entry Point
# =====================================================

async def main():
    """Demonstration of the MCP server"""
    
    # Initialize server
    server = LLMMemoryMCPServer(
        db_path="data/llm_memory.duckdb",
        max_context_tokens=100000
    )
    
    # Initialize a project
    project_info = await server.initialize_project("/path/to/project")
    print(f"Project initialized: {project_info}")
    
    # Analyze some files
    # analysis = await server.analyze_files([
    #     "/path/to/project/main.py",
    #     "/path/to/project/utils.py"
    # ])
    # print(f"Analysis complete: {analysis}")
    
    # Get context
    context = await server.get_context(max_tokens=4000)
    print(f"Context loaded: {context['metrics']}")
    
    # Find dependencies
    # deps = await server.find_dependencies("/path/to/project/main.py")
    # print(f"Dependencies: {deps}")
    
    # Detect issues
    issues = await server.detect_issues()
    print(f"Issues detected: {issues}")
    
    print("\nMCP Memory Server with DuckDB is running...")

if __name__ == "__main__":
    asyncio.run(main())