#!/usr/bin/env python3
"""
Advanced Graph Algorithms for Knowledge Graph Analysis
Scalable algorithms for million-node graphs
"""

import duckdb
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq
import logging

logger = logging.getLogger(__name__)

@dataclass
class GraphMetrics:
    """Container for graph-wide metrics"""
    node_count: int
    edge_count: int
    density: float
    avg_degree: float
    max_degree: int
    diameter: int
    clustering_coefficient: float
    connected_components: int
    strongly_connected_components: int
    
@dataclass 
class NodeMetrics:
    """Container for node-specific metrics"""
    node_id: int
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    pagerank: float
    clustering_coefficient: float
    eccentricity: int

class GraphAlgorithms:
    """
    Advanced graph algorithms optimized for code knowledge graphs
    Uses DuckDB for efficient large-scale computations
    """
    
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self.cache = {}
        
    # =====================================================
    # Centrality Algorithms
    # =====================================================
    
    def compute_pagerank(self,
                        project_id: int,
                        damping: float = 0.85,
                        iterations: int = 20) -> Dict[int, float]:
        """
        Compute PageRank scores for all nodes
        Identifies most important code elements
        """
        
        # Get graph structure
        edges = self.conn.execute("""
            SELECT 
                e.source_id,
                e.target_id,
                e.strength
            FROM llm_memory.edge_relationship e
            JOIN llm_memory.node_file f ON f.file_id = e.source_id
            WHERE f.project_id = ? 
              AND e.source_type = 'file'
              AND e.target_type = 'file'
              AND e.is_active = true
        """, [project_id]).fetchall()
        
        # Build adjacency lists
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        nodes = set()
        
        for src, dst, weight in edges:
            graph[src].append((dst, weight))
            reverse_graph[dst].append((src, weight))
            nodes.add(src)
            nodes.add(dst)
        
        # Initialize PageRank
        n = len(nodes)
        if n == 0:
            return {}
            
        pagerank = {node: 1.0 / n for node in nodes}
        
        # Power iteration
        for _ in range(iterations):
            new_pagerank = {}
            
            for node in nodes:
                rank = (1 - damping) / n
                
                for predecessor, weight in reverse_graph[node]:
                    out_degree = len(graph[predecessor])
                    if out_degree > 0:
                        rank += damping * weight * pagerank[predecessor] / out_degree
                
                new_pagerank[node] = rank
            
            pagerank = new_pagerank
        
        # Normalize
        total = sum(pagerank.values())
        if total > 0:
            pagerank = {k: v/total for k, v in pagerank.items()}
        
        # Store in database
        for node_id, score in pagerank.items():
            self.conn.execute("""
                INSERT INTO llm_memory.fact_understanding 
                (node_type, node_id, session_id, confidence_score)
                VALUES ('file', ?, 0, ?)
                ON CONFLICT (node_type, node_id) 
                DO UPDATE SET confidence_score = ?
            """, [node_id, score, score])
        
        return pagerank
    
    def compute_betweenness_centrality(self,
                                      project_id: int,
                                      sample_size: Optional[int] = None) -> Dict[int, float]:
        """
        Compute betweenness centrality
        Identifies nodes that act as bridges
        Uses sampling for large graphs
        """
        
        # Get all nodes
        nodes = self.conn.execute("""
            SELECT file_id FROM llm_memory.node_file
            WHERE project_id = ?
        """, [project_id]).fetchall()
        
        node_ids = [n[0] for n in nodes]
        
        # Sample nodes if graph is large
        if sample_size and len(node_ids) > sample_size:
            import random
            node_ids = random.sample(node_ids, sample_size)
        
        betweenness = defaultdict(float)
        
        # For each source node
        for source in node_ids:
            # BFS to find shortest paths
            distances, paths = self._bfs_shortest_paths(source, project_id)
            
            # Accumulate betweenness
            for target in distances:
                if target != source:
                    path = self._reconstruct_path(source, target, paths)
                    for node in path[1:-1]:  # Exclude source and target
                        betweenness[node] += 1
        
        # Normalize
        n = len(node_ids)
        if n > 2:
            norm = 2.0 / ((n - 1) * (n - 2))
            betweenness = {k: v * norm for k, v in betweenness.items()}
        
        return dict(betweenness)
    
    def compute_closeness_centrality(self,
                                    project_id: int) -> Dict[int, float]:
        """
        Compute closeness centrality
        Identifies nodes with shortest average path to all others
        """
        
        closeness = {}
        
        # Get all nodes
        nodes = self.conn.execute("""
            SELECT file_id FROM llm_memory.node_file
            WHERE project_id = ?
        """, [project_id]).fetchall()
        
        for node in nodes:
            node_id = node[0]
            
            # Get distances to all other nodes
            distances = self.conn.execute("""
                WITH RECURSIVE paths AS (
                    SELECT ? as node_id, 0 as distance
                    
                    UNION ALL
                    
                    SELECT 
                        e.target_id as node_id,
                        p.distance + 1 as distance
                    FROM paths p
                    JOIN llm_memory.edge_relationship e
                        ON e.source_id = p.node_id
                    WHERE p.distance < 10
                      AND e.is_active = true
                )
                SELECT node_id, MIN(distance) as min_distance
                FROM paths
                WHERE node_id != ?
                GROUP BY node_id
            """, [node_id, node_id]).fetchall()
            
            if distances:
                total_distance = sum(d[1] for d in distances)
                n_reachable = len(distances)
                
                if total_distance > 0:
                    closeness[node_id] = n_reachable / total_distance
                else:
                    closeness[node_id] = 0
            else:
                closeness[node_id] = 0
        
        return closeness
    
    # =====================================================
    # Community Detection
    # =====================================================
    
    def detect_communities(self,
                          project_id: int,
                          resolution: float = 1.0) -> Dict[int, int]:
        """
        Detect communities using Louvain algorithm
        Groups related code elements
        """
        
        # Get weighted edges
        edges = self.conn.execute("""
            SELECT source_id, target_id, strength
            FROM llm_memory.edge_relationship e
            JOIN llm_memory.node_file f ON f.file_id = e.source_id
            WHERE f.project_id = ?
              AND e.is_active = true
        """, [project_id]).fetchall()
        
        # Build graph
        graph = defaultdict(dict)
        nodes = set()
        
        for src, dst, weight in edges:
            graph[src][dst] = weight
            graph[dst][src] = weight  # Undirected for community detection
            nodes.add(src)
            nodes.add(dst)
        
        # Initialize: each node in its own community
        communities = {node: node for node in nodes}
        
        # Louvain algorithm (simplified)
        improved = True
        while improved:
            improved = False
            
            for node in nodes:
                current_community = communities[node]
                
                # Calculate modularity gain for moving to each neighbor's community
                best_community = current_community
                best_gain = 0
                
                for neighbor in graph[node]:
                    neighbor_community = communities[neighbor]
                    
                    if neighbor_community != current_community:
                        gain = self._modularity_gain(
                            node, neighbor_community, 
                            communities, graph, resolution
                        )
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_community = neighbor_community
                
                if best_community != current_community:
                    communities[node] = best_community
                    improved = True
        
        # Renumber communities
        unique_communities = set(communities.values())
        community_map = {c: i for i, c in enumerate(unique_communities)}
        communities = {node: community_map[c] for node, c in communities.items()}
        
        return communities
    
    def _modularity_gain(self,
                        node: int,
                        target_community: int,
                        communities: Dict[int, int],
                        graph: Dict[int, Dict[int, float]],
                        resolution: float) -> float:
        """Calculate modularity gain from moving node to target community"""
        
        # Sum of weights of edges from node to target community
        ki_in = sum(
            graph[node].get(neighbor, 0)
            for neighbor in graph[node]
            if communities[neighbor] == target_community
        )
        
        # Sum of all weights in target community
        sigma_tot = sum(
            sum(graph[n].values())
            for n in communities
            if communities[n] == target_community
        )
        
        # Degree of node
        ki = sum(graph[node].values())
        
        # Total weight
        m = sum(sum(edges.values()) for edges in graph.values()) / 2
        
        if m == 0:
            return 0
        
        # Modularity gain
        gain = (ki_in - resolution * sigma_tot * ki / (2 * m)) / m
        
        return gain
    
    # =====================================================
    # Pattern Detection
    # =====================================================
    
    def find_code_clones(self,
                        project_id: int,
                        min_similarity: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        Find similar code patterns (potential clones)
        Uses entity similarity
        """
        
        # Get entity signatures
        entities = self.conn.execute("""
            SELECT 
                e.entity_id,
                e.entity_name,
                e.signature,
                e.entity_type
            FROM llm_memory.node_entity e
            JOIN llm_memory.node_file f ON f.file_id = e.file_id
            WHERE f.project_id = ?
        """, [project_id]).fetchall()
        
        clones = []
        
        # Compare entities pairwise
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # Same type and similar name
                if e1[3] == e2[3]:  # Same entity_type
                    similarity = self._string_similarity(e1[1], e2[1])
                    
                    if similarity >= min_similarity:
                        clones.append((e1[0], e2[0], similarity))
                        
                        # Create similarity edge
                        self.conn.execute("""
                            INSERT INTO llm_memory.edge_relationship
                            (source_type, source_id, target_type, target_id,
                             relationship_type, strength, confidence)
                            VALUES ('entity', ?, 'entity', ?, 'similar_to', ?, ?)
                            ON CONFLICT DO NOTHING
                        """, [e1[0], e2[0], similarity, similarity])
        
        return clones
    
    def detect_anti_patterns(self, project_id: int) -> Dict[str, List[Dict]]:
        """
        Detect common anti-patterns in code structure
        """
        
        anti_patterns = {}
        
        # God Class: Too many methods and dependencies
        god_classes = self.conn.execute("""
            SELECT 
                e.entity_id,
                e.entity_name,
                COUNT(DISTINCT e2.entity_id) as method_count,
                COUNT(DISTINCT er.target_id) as dependency_count
            FROM llm_memory.node_entity e
            LEFT JOIN llm_memory.node_entity e2 
                ON e2.parent_entity_id = e.entity_id
            LEFT JOIN llm_memory.edge_relationship er
                ON er.source_id = e.entity_id 
                AND er.source_type = 'entity'
            WHERE e.entity_type = 'class'
            GROUP BY e.entity_id, e.entity_name
            HAVING method_count > 20 OR dependency_count > 15
        """).fetchall()
        
        anti_patterns['god_class'] = [
            {
                'entity_id': gc[0],
                'name': gc[1],
                'methods': gc[2],
                'dependencies': gc[3]
            }
            for gc in god_classes
        ]
        
        # Circular Dependencies
        cycles = self.conn.execute("""
            SELECT * FROM llm_memory.detect_cycles(?)
        """, [project_id]).fetchall()
        
        anti_patterns['circular_dependencies'] = [
            {
                'cycle': cycle[0],
                'type': cycle[1]
            }
            for cycle in cycles
        ]
        
        # Dead Code: Unreferenced entities
        dead_code = self.conn.execute("""
            SELECT 
                e.entity_id,
                e.entity_name,
                e.entity_type
            FROM llm_memory.node_entity e
            LEFT JOIN llm_memory.edge_relationship er
                ON er.target_id = e.entity_id 
                AND er.target_type = 'entity'
            WHERE er.edge_id IS NULL
              AND e.is_exported = false
        """).fetchall()
        
        anti_patterns['dead_code'] = [
            {
                'entity_id': dc[0],
                'name': dc[1],
                'type': dc[2]
            }
            for dc in dead_code
        ]
        
        # Long Method: Methods with too many lines
        long_methods = self.conn.execute("""
            SELECT 
                entity_id,
                entity_name,
                line_end - line_start as lines
            FROM llm_memory.node_entity
            WHERE entity_type IN ('function', 'method')
              AND line_end - line_start > 50
            ORDER BY lines DESC
        """).fetchall()
        
        anti_patterns['long_methods'] = [
            {
                'entity_id': lm[0],
                'name': lm[1],
                'lines': lm[2]
            }
            for lm in long_methods
        ]
        
        return anti_patterns
    
    # =====================================================
    # Path Analysis
    # =====================================================
    
    def find_critical_paths(self,
                           project_id: int,
                           start_files: List[str],
                           end_files: List[str]) -> List[List[int]]:
        """
        Find critical paths between sets of files
        Useful for understanding data/control flow
        """
        
        # Get file IDs
        start_ids = self.conn.execute("""
            SELECT file_id FROM llm_memory.node_file
            WHERE project_id = ? AND file_path = ANY(?)
        """, [project_id, start_files]).fetchall()
        
        end_ids = self.conn.execute("""
            SELECT file_id FROM llm_memory.node_file
            WHERE project_id = ? AND file_path = ANY(?)
        """, [project_id, end_files]).fetchall()
        
        start_ids = [s[0] for s in start_ids]
        end_ids = [e[0] for e in end_ids]
        
        critical_paths = []
        
        for start in start_ids:
            for end in end_ids:
                # Find shortest path
                path = self.conn.execute("""
                    SELECT * FROM llm_memory.find_shortest_path(?, ?)
                """, [start, end]).fetchone()
                
                if path and path[0]:
                    critical_paths.append(path[0])
        
        return critical_paths
    
    def compute_reachability(self,
                           node_id: int,
                           max_hops: int = 5) -> Set[int]:
        """
        Compute all nodes reachable from a given node
        """
        
        reachable = self.conn.execute("""
            WITH RECURSIVE reach AS (
                SELECT ? as node_id, 0 as distance
                
                UNION
                
                SELECT 
                    e.target_id as node_id,
                    r.distance + 1 as distance
                FROM reach r
                JOIN llm_memory.edge_relationship e
                    ON e.source_id = r.node_id
                WHERE r.distance < ?
                  AND e.is_active = true
            )
            SELECT DISTINCT node_id
            FROM reach
            WHERE node_id != ?
        """, [node_id, max_hops, node_id]).fetchall()
        
        return {r[0] for r in reachable}
    
    # =====================================================
    # Graph Metrics
    # =====================================================
    
    def compute_graph_metrics(self, project_id: int) -> GraphMetrics:
        """
        Compute comprehensive graph metrics
        """
        
        # Basic counts
        counts = self.conn.execute("""
            SELECT 
                COUNT(DISTINCT f.file_id) as node_count,
                COUNT(DISTINCT e.edge_id) as edge_count,
                AVG(degree.total) as avg_degree,
                MAX(degree.total) as max_degree
            FROM llm_memory.node_file f
            LEFT JOIN llm_memory.edge_relationship e
                ON (e.source_id = f.file_id AND e.source_type = 'file')
                OR (e.target_id = f.file_id AND e.target_type = 'file')
            LEFT JOIN (
                SELECT 
                    file_id,
                    COUNT(DISTINCT e1.edge_id) + COUNT(DISTINCT e2.edge_id) as total
                FROM llm_memory.node_file f
                LEFT JOIN llm_memory.edge_relationship e1
                    ON e1.source_id = f.file_id
                LEFT JOIN llm_memory.edge_relationship e2
                    ON e2.target_id = f.file_id
                WHERE f.project_id = ?
                GROUP BY file_id
            ) degree ON degree.file_id = f.file_id
            WHERE f.project_id = ?
        """, [project_id, project_id]).fetchone()
        
        node_count = counts[0] or 0
        edge_count = counts[1] or 0
        avg_degree = counts[2] or 0
        max_degree = counts[3] or 0
        
        # Density
        density = 0
        if node_count > 1:
            max_edges = node_count * (node_count - 1)
            density = edge_count / max_edges if max_edges > 0 else 0
        
        # Connected components (simplified - counts weakly connected)
        components = self._count_connected_components(project_id)
        
        # Clustering coefficient
        clustering = self._compute_clustering_coefficient(project_id)
        
        # Diameter (simplified - uses sampling)
        diameter = self._estimate_diameter(project_id)
        
        return GraphMetrics(
            node_count=node_count,
            edge_count=edge_count,
            density=density,
            avg_degree=avg_degree,
            max_degree=max_degree,
            diameter=diameter,
            clustering_coefficient=clustering,
            connected_components=components,
            strongly_connected_components=components  # Simplified
        )
    
    def _count_connected_components(self, project_id: int) -> int:
        """Count connected components using DFS"""
        
        # Get all nodes
        nodes = self.conn.execute("""
            SELECT file_id FROM llm_memory.node_file
            WHERE project_id = ?
        """, [project_id]).fetchall()
        
        node_ids = {n[0] for n in nodes}
        visited = set()
        components = 0
        
        for node in node_ids:
            if node not in visited:
                # DFS from this node
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        
                        # Get neighbors
                        neighbors = self.conn.execute("""
                            SELECT target_id FROM llm_memory.edge_relationship
                            WHERE source_id = ? AND is_active = true
                            UNION
                            SELECT source_id FROM llm_memory.edge_relationship
                            WHERE target_id = ? AND is_active = true
                        """, [current, current]).fetchall()
                        
                        for n in neighbors:
                            if n[0] in node_ids and n[0] not in visited:
                                stack.append(n[0])
                
                components += 1
        
        return components
    
    def _compute_clustering_coefficient(self, project_id: int) -> float:
        """Compute average clustering coefficient"""
        
        # For each node, compute local clustering
        nodes = self.conn.execute("""
            SELECT file_id FROM llm_memory.node_file
            WHERE project_id = ?
            LIMIT 100  -- Sample for performance
        """, [project_id]).fetchall()
        
        coefficients = []
        
        for node in nodes:
            node_id = node[0]
            
            # Get neighbors
            neighbors = self.conn.execute("""
                SELECT DISTINCT 
                    CASE 
                        WHEN source_id = ? THEN target_id
                        ELSE source_id
                    END as neighbor
                FROM llm_memory.edge_relationship
                WHERE (source_id = ? OR target_id = ?)
                  AND is_active = true
            """, [node_id, node_id, node_id]).fetchall()
            
            neighbor_ids = [n[0] for n in neighbors]
            k = len(neighbor_ids)
            
            if k < 2:
                coefficients.append(0)
                continue
            
            # Count edges between neighbors
            edges_between = self.conn.execute("""
                SELECT COUNT(*)
                FROM llm_memory.edge_relationship
                WHERE source_id = ANY(?) 
                  AND target_id = ANY(?)
                  AND is_active = true
            """, [neighbor_ids, neighbor_ids]).fetchone()[0]
            
            # Local clustering coefficient
            max_edges = k * (k - 1) / 2
            coef = edges_between / max_edges if max_edges > 0 else 0
            coefficients.append(coef)
        
        return np.mean(coefficients) if coefficients else 0
    
    def _estimate_diameter(self, project_id: int, sample_size: int = 10) -> int:
        """Estimate graph diameter using sampling"""
        
        # Sample nodes
        nodes = self.conn.execute("""
            SELECT file_id FROM llm_memory.node_file
            WHERE project_id = ?
            ORDER BY RANDOM()
            LIMIT ?
        """, [project_id, sample_size]).fetchall()
        
        max_distance = 0
        
        for node in nodes:
            # Find eccentricity (max distance from this node)
            distances = self.conn.execute("""
                WITH RECURSIVE paths AS (
                    SELECT ? as node_id, 0 as distance
                    
                    UNION ALL
                    
                    SELECT 
                        e.target_id as node_id,
                        p.distance + 1 as distance
                    FROM paths p
                    JOIN llm_memory.edge_relationship e
                        ON e.source_id = p.node_id
                    WHERE p.distance < 20  -- Limit depth
                      AND e.is_active = true
                )
                SELECT MAX(distance)
                FROM paths
            """, [node[0]]).fetchone()[0]
            
            if distances and distances > max_distance:
                max_distance = distances
        
        return max_distance
    
    # =====================================================
    # Helper Methods
    # =====================================================
    
    def _bfs_shortest_paths(self,
                          source: int,
                          project_id: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        """BFS to find shortest paths from source"""
        
        distances = {source: 0}
        paths = {source: None}
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            
            # Get neighbors
            neighbors = self.conn.execute("""
                SELECT target_id 
                FROM llm_memory.edge_relationship
                WHERE source_id = ? AND is_active = true
            """, [current]).fetchall()
            
            for neighbor in neighbors:
                n_id = neighbor[0]
                if n_id not in distances:
                    distances[n_id] = distances[current] + 1
                    paths[n_id] = current
                    queue.append(n_id)
        
        return distances, paths
    
    def _reconstruct_path(self,
                        source: int,
                        target: int,
                        paths: Dict[int, int]) -> List[int]:
        """Reconstruct path from source to target"""
        
        if target not in paths:
            return []
        
        path = []
        current = target
        
        while current is not None:
            path.append(current)
            current = paths.get(current)
        
        return list(reversed(path))
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity using Levenshtein distance"""
        
        if not s1 or not s2:
            return 0
        
        # Simplified - in production use proper edit distance
        if s1 == s2:
            return 1.0
        
        # Check prefix similarity
        common_prefix = 0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                common_prefix += 1
            else:
                break
        
        return common_prefix / max(len(s1), len(s2))


# =====================================================
# Example Usage
# =====================================================

async def analyze_codebase(db_path: str, project_path: str):
    """Example analysis workflow"""
    
    conn = duckdb.connect(db_path)
    algorithms = GraphAlgorithms(conn)
    
    # Assume project_id = 1
    project_id = 1
    
    print("Computing graph metrics...")
    metrics = algorithms.compute_graph_metrics(project_id)
    print(f"Graph: {metrics.node_count} nodes, {metrics.edge_count} edges")
    print(f"Density: {metrics.density:.3f}, Avg degree: {metrics.avg_degree:.2f}")
    
    print("\nComputing PageRank...")
    pagerank = algorithms.compute_pagerank(project_id)
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Most important files:")
    for node_id, score in top_nodes:
        print(f"  Node {node_id}: {score:.4f}")
    
    print("\nDetecting communities...")
    communities = algorithms.detect_communities(project_id)
    community_sizes = defaultdict(int)
    for node, comm in communities.items():
        community_sizes[comm] += 1
    print(f"Found {len(community_sizes)} communities")
    print(f"Largest community: {max(community_sizes.values())} nodes")
    
    print("\nDetecting anti-patterns...")
    anti_patterns = algorithms.detect_anti_patterns(project_id)
    for pattern, instances in anti_patterns.items():
        print(f"  {pattern}: {len(instances)} instances")
    
    print("\nFinding code clones...")
    clones = algorithms.find_code_clones(project_id)
    print(f"Found {len(clones)} potential code clones")
    
    conn.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(analyze_codebase("llm_memory.duckdb", "/path/to/project"))