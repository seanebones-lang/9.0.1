#!/usr/bin/env python3
"""
🌊 Fast DeepBlue Loader
Optimized for Cursor integration with minimal startup time
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class FastDeepBlueLoader:
    """Fast-loading DeepBlue for Cursor integration."""
    
    def __init__(self):
        self.deepblue_path = "/Users/seanmcdonnell/Desktop/DeepBlue"
        self.initialized = False
        self.start_time = time.time()
        
    def quick_init(self) -> Dict[str, Any]:
        """Quick initialization without full system startup."""
        try:
            # Check if DeepBlue is already running
            if self._is_deepblue_running():
                return {
                    "success": True,
                    "mode": "api",
                    "url": "http://localhost:8000",
                    "status": "running",
                    "load_time": time.time() - self.start_time
                }
            
            # Quick file-based access
            return self._quick_file_access()
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "load_time": time.time() - self.start_time
            }
    
    def _is_deepblue_running(self) -> bool:
        """Check if DeepBlue API is running."""
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def _quick_file_access(self) -> Dict[str, Any]:
        """Access DeepBlue files directly for fast loading."""
        try:
            # Check for existing data
            data_files = [
                "constrained_rag.db",
                "data_catalog.db", 
                "build_diagnosis.db"
            ]
            
            available_data = {}
            for file in data_files:
                file_path = os.path.join(self.deepblue_path, file)
                if os.path.exists(file_path):
                    available_data[file] = os.path.getsize(file_path)
            
            # Load key configuration files
            config_files = [
                "config.py",
                "requirements.txt",
                "DEEPBLUE_README.md"
            ]
            
            config_data = {}
            for file in config_files:
                file_path = os.path.join(self.deepblue_path, file)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        config_data[file] = f.read()[:1000]  # First 1000 chars
            
            return {
                "success": True,
                "mode": "file_access",
                "available_data": available_data,
                "config_files": list(config_data.keys()),
                "load_time": time.time() - self.start_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "load_time": time.time() - self.start_time
            }
    
    def get_quick_rag_plan(self) -> str:
        """Get a quick RAG plan without full system initialization."""
        return """
🌊 FAST DEEPBLUE RAG PLAN
========================

QUICK START (30 seconds):
1. Use existing DeepBlue data at /Users/seanmcdonnell/Desktop/DeepBlue/
2. Access verified knowledge from constrained_rag.db
3. Use build diagnosis from build_diagnosis.db
4. Query via API if running: http://localhost:8000/query

VERIFIED TECHNOLOGIES:
- LangChain 0.2.x (latest stable)
- FastAPI 0.104+ (high performance)
- Pinecone/Weaviate (vector databases)
- OpenAI text-embedding-3-small (embeddings)
- GPT-4/Claude-3 (generation)

ARCHITECTURE:
- Modular design with independent components
- Async processing for speed
- Caching for performance
- Verification system for accuracy

CURSOR INTEGRATION:
- Direct file access for fast loading
- API integration when available
- Real-time verification
- Hallucination prevention

PERFORMANCE TARGETS:
- Load time: < 5 seconds
- Query time: < 100ms
- Accuracy: > 90%
- Uptime: 99.9%
        """
    
    def start_lightweight_service(self) -> Dict[str, Any]:
        """Start a lightweight DeepBlue service."""
        try:
            # Check if we can start a minimal service
            import subprocess
            import signal
            
            # Start minimal API server
            cmd = [
                "python3", 
                os.path.join(self.deepblue_path, "api_deepblue_backend.py")
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.deepblue_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check if it's running
            if process.poll() is None:
                return {
                    "success": True,
                    "pid": process.pid,
                    "status": "started",
                    "load_time": time.time() - self.start_time
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to start service",
                    "load_time": time.time() - self.start_time
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "load_time": time.time() - self.start_time
            }

def main():
    """Main function for fast DeepBlue loading."""
    print("🌊 Fast DeepBlue Loader Starting...")
    
    loader = FastDeepBlueLoader()
    
    # Quick initialization
    result = loader.quick_init()
    
    print(f"✅ Initialization: {result['success']}")
    print(f"⏱️ Load Time: {result['load_time']:.2f}s")
    
    if result['success']:
        print(f"🔧 Mode: {result.get('mode', 'unknown')}")
        
        if result.get('mode') == 'api':
            print("🚀 DeepBlue API is running - ready for queries!")
        elif result.get('mode') == 'file_access':
            print("📁 Using file access mode - fast but limited")
            print(f"📊 Available data: {len(result.get('available_data', {}))} files")
    
    # Get quick RAG plan
    print("\n" + loader.get_quick_rag_plan())
    
    return result

if __name__ == "__main__":
    main()
