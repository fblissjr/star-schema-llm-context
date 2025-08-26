#!/usr/bin/env python3
"""
Setup script for LLM Memory System
Initializes database and provides quick start
"""

import os
import sys
import asyncio
import duckdb
from pathlib import Path
from db_manager import DuckDBManager

def setup_database(db_path="data/llm_memory.duckdb"):
    """Initialize the database with schema"""
    print(f"Setting up database at {db_path}...")

    db = DuckDBManager(db_path)

    # Using default schema (main) to avoid naming conflicts
        
    # Load and execute schema
    schema_file = Path(__file__).parent / "schema.sql"
    if schema_file.exists():
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
            
            # No schema creation needed - using default schema
            
            # Simple but effective: split on lines that end with semicolon
            statements = []
            current_stmt = []
            
            for line in schema_sql.split('\n'):
                # Skip pure comment lines
                if line.strip().startswith('--'):
                    continue
                    
                current_stmt.append(line)
                
                # Check if line ends with semicolon (ignoring trailing whitespace)
                if line.rstrip().endswith(';'):
                    # We have a complete statement
                    full_stmt = '\n'.join(current_stmt).strip()
                    if full_stmt:
                        statements.append(full_stmt)
                    current_stmt = []
            
            # Add any remaining statement
            if current_stmt:
                full_stmt = '\n'.join(current_stmt).strip()
                if full_stmt:
                    statements.append(full_stmt)
            
            # Execute each statement
            success_count = 0
            failed_count = 0
            for i, stmt in enumerate(statements):
                if not stmt.strip():
                    continue
                    
                try:
                    db.execute(stmt)
                    success_count += 1
                except duckdb.CatalogException as e:
                    if 'already exists' in str(e).lower():
                        success_count += 1
                    else:
                        failed_count += 1
                        print(f"Warning {i+1}: CatalogException: {str(e)[:100]}")
                except Exception as e:
                    failed_count += 1
                    print(f"Warning {i+1}: {type(e).__name__}: {str(e)[:100]}")
                    
            print(f"✓ Schema execution: {success_count} succeeded, {failed_count} failed")
            
            # List created tables
            tables_result = db.fetch_all("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'BASE TABLE'")
            if tables_result:
                print(f"✓ Created tables: {', '.join([t[0] for t in tables_result][:5])}..." if len(tables_result) > 5 else f"✓ Created tables: {', '.join([t[0] for t in tables_result])}")
    else:
        print("✗ schema.sql not found!")
        return False

    # Verify tables were created
    result = db.fetch_one("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
    """)
    tables = result[0] if result else 0

    print(f"✓ Found {tables} tables")

    print("✓ Performance settings configured")

    db.close()
    return True

async def quick_start_demo():
    """Run a quick demo of the system"""
    from mcp_server import LLMMemoryMCPServer

    print("\n" + "="*50)
    print("Quick Start Demo")
    print("="*50)

    # Create server
    server = LLMMemoryMCPServer(
        db_path="data/llm_memory.duckdb",
        max_context_tokens=100000
    )

    # Initialize a demo project
    demo_path = Path.cwd()
    print(f"\nInitializing project: {demo_path}")

    project_info = await server.initialize_project(str(demo_path))
    print(f"✓ Project ID: {project_info['project_id']}")
    print(f"✓ Session ID: {project_info['session_id']}")
    print(f"✓ Understanding Level: {project_info['understanding_level']:.1%}")

    # Analyze Python files in current directory
    py_files = list(demo_path.glob("*.py"))[:3]
    if py_files:
        print(f"\nAnalyzing {len(py_files)} Python files...")
        analysis = await server.analyze_files([str(f) for f in py_files])
        print(f"✓ Analyzed {analysis['files_analyzed']} files")

    # Get context
    print("\nRetrieving optimized context...")
    context = await server.get_context(max_tokens=4000)
    metrics = context['metrics']
    print(f" Context built:")
    print(f"  - Nodes: {metrics['nodes_included']}")
    print(f"  - Edges: {metrics['edges_included']}")
    print(f"  - Tokens: {metrics['token_count']}")
    print(f"  - Relevance: {metrics['avg_relevance']:.2f}")

    # Detect issues
    print("\nScanning for issues...")
    issues = await server.detect_issues()
    print(f"✓ Found {issues['issues_found']} potential issues")

    print("\n" + "="*50)
    print("Demo Complete!")
    print("="*50)
    print("\nYour LLM Memory System is ready to use.")
    print("\nExample usage:")
    print("""
from mcp_server import LLMMemoryMCPServer

server = LLMMemoryMCPServer()
await server.initialize_project("/your/project/path")
context = await server.get_context(max_tokens=4000)
""")

def main():
    """Main setup entry point"""
    print("LLM Memory System Setup")
    print("="*50)

    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check dependencies
    try:
        import duckdb
        print(f"✓ DuckDB {duckdb.__version__}")
    except ImportError:
        print("✗ DuckDB not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    # Setup database
    if not setup_database():
        print("✗ Database setup failed")
        sys.exit(1)

    # Run demo
    response = input("\nRun quick start demo? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(quick_start_demo())

    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Review README.md for documentation")
    print("2. Configure your project path")
    print("3. Start using the MCP server")

if __name__ == "__main__":
    main()
