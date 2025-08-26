# Makefile for LLM Memory System

.PHONY: help install setup test clean run analyze demo

# Default target
help:
	@echo "LLM Memory System - Available Commands:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make setup      - Initialize database and run setup"
	@echo "  make demo       - Run quick start demo"
	@echo "  make analyze    - Analyze current directory"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean generated files"
	@echo "  make run        - Start MCP server"

# Install dependencies
install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Setup database
setup:
	python setup.py
	@echo "✓ Setup complete"

# Run demo
demo:
	@echo "Running demo..."
	python -c "import asyncio; from mcp_server import main; asyncio.run(main())"

# Analyze current project
analyze:
	@echo "Analyzing current directory..."
	python -c """
import asyncio
import os
from mcp_server import LLMMemoryMCPServer

async def analyze():
    server = LLMMemoryMCPServer()
    await server.initialize_project(os.getcwd())
    files = [f for f in os.listdir('.') if f.endswith('.py')][:5]
    result = await server.analyze_files(files)
    print(f'Analyzed {result["files_analyzed"]} files')
    
asyncio.run(analyze())
"""

# Run tests
test:
	@echo "Running tests..."
	python -c """
import duckdb
print('Testing DuckDB connection...')
conn = duckdb.connect(':memory:')
result = conn.execute('SELECT 1').fetchone()
assert result[0] == 1
print('✓ DuckDB working')

print('Testing schema...')
conn = duckdb.connect('test.duckdb')
with open('schema.sql', 'r') as f:
    conn.execute(f.read())
tables = conn.execute('''
    SELECT COUNT(*) FROM information_schema.tables
    WHERE table_schema = 'llm_memory'
''').fetchone()[0]
print(f'✓ Created {tables} tables')
conn.close()
import os
os.remove('test.duckdb')
"""

# Clean generated files
clean:
	rm -f llm_memory.duckdb
	rm -f llm_memory.log
	rm -f test.duckdb
	rm -rf __pycache__
	rm -rf *.pyc
	@echo "✓ Cleaned generated files"

# Start MCP server
run:
	@echo "Starting MCP server..."
	python mcp_server.py

# Database operations
db-stats:
	@echo "Database statistics:"
	@python -c """
import duckdb
conn = duckdb.connect('llm_memory.duckdb')
stats = conn.execute('''
    SELECT 
        'Projects' as entity,
        COUNT(*) as count
    FROM llm_memory.node_project
    UNION ALL
    SELECT 
        'Files' as entity,
        COUNT(*) as count
    FROM llm_memory.node_file
    UNION ALL
    SELECT 
        'Entities' as entity,
        COUNT(*) as count
    FROM llm_memory.node_entity
    UNION ALL
    SELECT 
        'Edges' as entity,
        COUNT(*) as count
    FROM llm_memory.edge_relationship
''').fetchall()
for entity, count in stats:
    print(f'{entity}: {count}')
"""

db-optimize:
	@echo "Optimizing database..."
	@python -c """
import duckdb
conn = duckdb.connect('llm_memory.duckdb')
conn.execute('VACUUM')
conn.execute('ANALYZE')
conn.execute('SELECT llm_memory.refresh_views()')
conn.execute('SELECT llm_memory.decay_relevance()')
print('✓ Database optimized')
"""

# Development helpers
format:
	black *.py
	@echo "✓ Code formatted"

lint:
	pylint *.py
	@echo "✓ Code linted"