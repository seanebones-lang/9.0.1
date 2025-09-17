#!/usr/bin/env python3
"""
🌊 Cursor DeepBlue Integration
Fast-loading integration for Cursor AI with DeepBlue capabilities
"""

import os
import sys
import json
import time
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path

class CursorDeepBlueIntegration:
    """Fast Cursor integration with DeepBlue capabilities."""
    
    def __init__(self):
        self.deepblue_path = "/Users/seanmcdonnell/Desktop/DeepBlue"
        self.cursor_workspace = os.getcwd()
        self.start_time = time.time()
        self.initialized = False
        
    def quick_init(self) -> Dict[str, Any]:
        """Quick initialization for Cursor integration."""
        try:
            # Check if DeepBlue API is running
            if self._check_api():
                return self._init_api_mode()
            
            # Check for local databases
            if self._check_databases():
                return self._init_database_mode()
            
            # Fallback to file mode
            return self._init_file_mode()
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "load_time": time.time() - self.start_time
            }
    
    def _check_api(self) -> bool:
        """Check if DeepBlue API is running."""
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def _check_databases(self) -> bool:
        """Check if DeepBlue databases exist."""
        db_files = [
            "constrained_rag.db",
            "data_catalog.db",
            "build_diagnosis.db"
        ]
        
        for db_file in db_files:
            if os.path.exists(os.path.join(self.deepblue_path, db_file)):
                return True
        return False
    
    def _init_api_mode(self) -> Dict[str, Any]:
        """Initialize in API mode."""
        self.initialized = True
        return {
            "success": True,
            "mode": "api",
            "url": "http://localhost:8000",
            "load_time": time.time() - self.start_time,
            "capabilities": [
                "real_time_queries",
                "verified_knowledge",
                "build_diagnosis",
                "system_building"
            ]
        }
    
    def _init_database_mode(self) -> Dict[str, Any]:
        """Initialize in database mode."""
        self.initialized = True
        return {
            "success": True,
            "mode": "database",
            "load_time": time.time() - self.start_time,
            "capabilities": [
                "verified_knowledge",
                "build_diagnosis",
                "file_access"
            ]
        }
    
    def _init_file_mode(self) -> Dict[str, Any]:
        """Initialize in file mode."""
        self.initialized = True
        return {
            "success": True,
            "mode": "file",
            "load_time": time.time() - self.start_time,
            "capabilities": [
                "file_access",
                "basic_queries"
            ]
        }
    
    def query_verified_knowledge(self, query: str) -> Dict[str, Any]:
        """Query verified knowledge from DeepBlue."""
        if not self.initialized:
            return {"error": "Not initialized"}
        
        try:
            if self._check_api():
                return self._query_api(query)
            elif self._check_databases():
                return self._query_database(query)
            else:
                return self._query_files(query)
        except Exception as e:
            return {"error": str(e)}
    
    def _query_api(self, query: str) -> Dict[str, Any]:
        """Query via API."""
        try:
            import requests
            response = requests.post(
                "http://localhost:8000/query",
                json={"question": query},
                timeout=5
            )
            return response.json()
        except Exception as e:
            return {"error": f"API query failed: {e}"}
    
    def _query_database(self, query: str) -> Dict[str, Any]:
        """Query local databases."""
        try:
            # Query the constrained RAG database
            db_path = os.path.join(self.deepblue_path, "constrained_rag.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Simple query for relevant data
                cursor.execute("""
                    SELECT content, source, trust_score 
                    FROM rag_data 
                    WHERE content LIKE ? 
                    ORDER BY trust_score DESC 
                    LIMIT 5
                """, (f"%{query}%",))
                
                results = cursor.fetchall()
                conn.close()
                
                return {
                    "success": True,
                    "results": [
                        {
                            "content": row[0],
                            "source": row[1],
                            "trust_score": row[2]
                        }
                        for row in results
                    ],
                    "mode": "database"
                }
            
            return {"error": "No database found"}
        except Exception as e:
            return {"error": f"Database query failed: {e}"}
    
    def _query_files(self, query: str) -> Dict[str, Any]:
        """Query local files."""
        try:
            # Search through README and config files
            files_to_search = [
                "DEEPBLUE_README.md",
                "ULTIMATE_DEEPBLUE_README.md",
                "ENHANCED_DEEPBLUE_README.md"
            ]
            
            results = []
            for filename in files_to_search:
                file_path = os.path.join(self.deepblue_path, filename)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if query.lower() in content.lower():
                            results.append({
                                "content": content[:500],
                                "source": filename,
                                "trust_score": 0.8
                            })
            
            return {
                "success": True,
                "results": results,
                "mode": "file"
            }
        except Exception as e:
            return {"error": f"File query failed: {e}"}
    
    def get_rag_recommendations(self) -> str:
        """Get RAG system recommendations."""
        return """
🌊 DEEPBLUE RAG RECOMMENDATIONS
===============================

VERIFIED TECHNOLOGIES (Current as of Dec 2024):
- LangChain 0.2.x (latest stable)
- FastAPI 0.104+ (high performance API)
- Pinecone (managed vector database)
- Weaviate (open-source vector search)
- OpenAI text-embedding-3-small (embeddings)
- GPT-4/Claude-3 (generation models)

ARCHITECTURE PATTERNS:
1. Modular Design - Independent components
2. Async Processing - Non-blocking operations
3. Caching Layer - Redis for performance
4. Verification System - Hallucination prevention
5. Monitoring - Prometheus metrics

PERFORMANCE OPTIMIZATIONS:
- Lazy loading for components
- Connection pooling
- Batch processing
- Response caching
- CDN for static assets

CURSOR INTEGRATION:
- Fast file access mode
- Real-time API queries
- Verified knowledge base
- Build diagnosis capabilities
- System building tools

QUICK START:
1. Use existing DeepBlue data
2. Query via API if running
3. Access verified knowledge
4. Build with real dependencies
        """
    
    def diagnose_build_issue(self, project_path: str) -> Dict[str, Any]:
        """Diagnose build issues using DeepBlue."""
        if not self.initialized:
            return {"error": "Not initialized"}
        
        try:
            # Check for common build issues
            issues = []
            solutions = []
            
            # Check for package.json
            package_json = os.path.join(project_path, "package.json")
            if os.path.exists(package_json):
                with open(package_json, 'r') as f:
                    data = json.load(f)
                    
                # Check for outdated dependencies
                if "dependencies" in data:
                    for dep, version in data["dependencies"].items():
                        if version.startswith("^") or version.startswith("~"):
                            issues.append(f"Outdated dependency: {dep}@{version}")
                            solutions.append(f"Update {dep} to latest version")
            
            # Check for common errors
            error_files = ["error.log", "build.log", "npm-debug.log"]
            for error_file in error_files:
                error_path = os.path.join(project_path, error_file)
                if os.path.exists(error_path):
                    with open(error_path, 'r') as f:
                        error_content = f.read()
                        if "ENOTFOUND" in error_content:
                            issues.append("Network connectivity issue")
                            solutions.append("Check internet connection and npm registry")
                        elif "EACCES" in error_content:
                            issues.append("Permission denied")
                            solutions.append("Run with sudo or fix permissions")
            
            return {
                "success": True,
                "issues": issues,
                "solutions": solutions,
                "project_path": project_path
            }
            
        except Exception as e:
            return {"error": f"Build diagnosis failed: {e}"}

def main():
    """Main function for Cursor integration."""
    print("🌊 Cursor DeepBlue Integration Starting...")
    
    integration = CursorDeepBlueIntegration()
    result = integration.quick_init()
    
    print(f"✅ Initialization: {result['success']}")
    print(f"⏱️ Load Time: {result['load_time']:.2f}s")
    print(f"🔧 Mode: {result.get('mode', 'unknown')}")
    
    if result['success']:
        print(f"🚀 Capabilities: {', '.join(result.get('capabilities', []))}")
        
        # Test query
        test_query = "What are the best practices for RAG systems?"
        query_result = integration.query_verified_knowledge(test_query)
        
        if query_result.get('success'):
            print(f"📚 Query Results: {len(query_result.get('results', []))} items found")
        else:
            print(f"❌ Query Error: {query_result.get('error')}")
    
    # Show recommendations
    print("\n" + integration.get_rag_recommendations())
    
    return result

if __name__ == "__main__":
    main()
