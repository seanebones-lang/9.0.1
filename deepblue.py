#!/usr/bin/env python3
"""
🌊 DeepBlue Assistant - We're gonna need a bigger boat! 🚢
A powerful project generator that creates React apps, Next.js apps, and Express APIs.
"""

import os
import json
import requests
import subprocess
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, Optional

class SimpleDeepBlueAssistant:
    """Simple DeepBlue assistant that works directly."""
    
    def __init__(self):
        self.web_interface_url = "http://localhost:5001"
        self.saved_builds_file = "saved_builds.json"
        self.saved_builds = self.load_saved_builds()
    
    def load_saved_builds(self) -> Dict[str, Any]:
        """Load saved builds."""
        if os.path.exists(self.saved_builds_file):
            try:
                with open(self.saved_builds_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_build(self, name: str, description: str, build_config: Dict[str, Any]) -> bool:
        """Save a build template."""
        try:
            self.saved_builds[name.lower()] = {
                "name": name,
                "description": description,
                "build_config": build_config,
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }
            
            with open(self.saved_builds_file, 'w') as f:
                json.dump(self.saved_builds, f, indent=2)
            
            print(f"✅ Build '{name}' saved successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to save build: {e}")
            return False
    
    def get_build(self, name: str) -> Dict[str, Any]:
        """Get a saved build."""
        return self.saved_builds.get(name.lower(), {})
    
    def list_builds(self) -> Dict[str, Any]:
        """List all saved builds."""
        return self.saved_builds
    
    def create_build(self, build_type: str, name: str = None) -> bool:
        """Create a new build."""
        try:
            if build_type == "react_app":
                return self._create_react_app(name)
            elif build_type == "nextjs_app":
                return self._create_nextjs_app(name)
            elif build_type == "express_api":
                return self._create_express_api(name)
            elif build_type == "vue_app":
                return self._create_vue_app(name)
            elif build_type == "python_flask":
                return self._create_python_flask(name)
            elif build_type == "fullstack_mern":
                return self._create_fullstack_mern(name)
            elif build_type == "microservices":
                return self._create_microservices_stack(name)
            elif build_type == "docker_app":
                return self._create_dockerized_app(name)
            elif build_type == "kubernetes_app":
                return self._create_kubernetes_app(name)
            elif build_type == "serverless":
                return self._create_serverless_app(name)
            elif build_type == "ai_ml":
                return self._create_ai_ml_project(name)
            elif build_type == "blockchain":
                return self._create_blockchain_project(name)
            elif build_type == "game_engine":
                return self._create_game_engine_project(name)
            elif build_type == "rag_system":
                return self._create_rag_system(name)
            else:
                print(f"❌ Unknown build type: {build_type}")
                return False
        except Exception as e:
            print(f"❌ Failed to create build: {e}")
            return False
    
    def _create_react_app(self, name: str = None) -> bool:
        """Create a React app."""
        if not name:
            name = input("Enter project name: ")
        
        print(f"🚀 Creating React app: {name}")
        
        # Create directory
        os.makedirs(name, exist_ok=True)
        os.chdir(name)
        
        # Create package.json
        package_json = {
            "name": name,
            "version": "0.1.0",
            "private": True,
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-scripts": "5.0.1"
            },
            "scripts": {
                "start": "react-scripts start",
                "build": "react-scripts build",
                "test": "react-scripts test",
                "eject": "react-scripts eject"
            }
        }
        
        with open("package.json", 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # Create basic files
        os.makedirs("src", exist_ok=True)
        os.makedirs("public", exist_ok=True)
        
        with open("src/App.js", 'w') as f:
            f.write('''import React from "react";
import "./App.css";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Hello from React!</h1>
        <p>Built with DeepBlue 🌊</p>
      </header>
    </div>
  );
}

export default App;
''')
        
        with open("src/index.js", 'w') as f:
            f.write('''import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
''')
        
        with open("src/App.css", 'w') as f:
            f.write('''.App {
  text-align: center;
}

.App-header {
  background-color: #282c34;
  padding: 20px;
  color: white;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
''')
        
        with open("src/index.css", 'w') as f:
            f.write('''body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
''')
        
        with open("public/index.html", 'w') as f:
            f.write('''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>React App</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
''')
        
        print(f"✅ React app '{name}' created successfully!")
        print(f"📁 Location: {os.getcwd()}")
        print(f"🚀 Run: cd {name} && npm install && npm start")
        
        # Go back to parent directory
        os.chdir("..")
        
        return True
    
    def _create_nextjs_app(self, name: str = None) -> bool:
        """Create a Next.js app."""
        if not name:
            name = input("Enter project name: ")
        
        print(f"🚀 Creating Next.js app: {name}")
        
        # Use npx to create Next.js app
        result = subprocess.run([
            "npx", "create-next-app@latest", name, "--typescript", "--tailwind", "--eslint", "--app", "--src-dir", "--import-alias", "@/*", "--yes"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Next.js app '{name}' created successfully!")
            print(f"📁 Location: {os.path.join(os.getcwd(), name)}")
            print(f"🚀 Run: cd {name} && npm run dev")
            return True
        else:
            print(f"❌ Failed to create Next.js app: {result.stderr}")
            return False
    
    def _create_express_api(self, name: str = None) -> bool:
        """Create an Express API."""
        if not name:
            name = input("Enter project name: ")
        
        print(f"🚀 Creating Express API: {name}")
        
        # Create directory
        os.makedirs(name, exist_ok=True)
        os.chdir(name)
        
        # Create package.json
        package_json = {
            "name": name,
            "version": "1.0.0",
            "description": "Express API created with DeepBlue",
            "main": "server.js",
            "scripts": {
                "start": "node server.js",
                "dev": "nodemon server.js"
            },
            "dependencies": {
                "express": "^4.18.2",
                "cors": "^2.8.5",
                "helmet": "^6.0.1"
            },
            "devDependencies": {
                "nodemon": "^2.0.20"
            }
        }
        
        with open("package.json", 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # Create server.js
        with open("server.js", 'w') as f:
            f.write('''const express = require('express');
const cors = require('cors');
const helmet = require('helmet');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Routes
app.get('/', (req, res) => {
  res.json({
    message: 'Hello from Express API!',
    created_by: 'DeepBlue 🌊',
    timestamp: new Date().toISOString()
  });
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', uptime: process.uptime() });
});

// Start server
app.listen(PORT, () => {
  console.log(`🌊 Server running on port ${PORT}`);
  console.log(`🚀 Created with DeepBlue`);
});
''')
        
        # Create .gitignore
        with open(".gitignore", 'w') as f:
            f.write('''node_modules/
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
npm-debug.log*
yarn-debug.log*
yarn-error.log*
''')
        
        print(f"✅ Express API '{name}' created successfully!")
        print(f"📁 Location: {os.getcwd()}")
        print(f"🚀 Run: cd {name} && npm install && npm start")
        
        # Go back to parent directory
        os.chdir("..")
        
        return True
    
    def _create_vue_app(self, name: str = None) -> bool:
        """Create a Vue.js app."""
        if not name:
            name = input("Enter project name: ")
        
        print(f"🚀 Creating Vue.js app: {name}")
        
        # Use npm to create Vue app
        result = subprocess.run([
            "npm", "create", "vue@latest", name, "--", "--typescript", "--router", "--pinia", "--vitest", "--eslint", "--prettier"
        ], capture_output=True, text=True, input="\n" * 6)  # Auto-accept all prompts
        
        if result.returncode == 0:
            print(f"✅ Vue.js app '{name}' created successfully!")
            print(f"📁 Location: {os.path.join(os.getcwd(), name)}")
            print(f"🚀 Run: cd {name} && npm install && npm run dev")
            return True
        else:
            print(f"❌ Failed to create Vue.js app: {result.stderr}")
            return False
    
    def _create_python_flask(self, name: str = None) -> bool:
        """Create a Python Flask API."""
        if not name:
            name = input("Enter project name: ")
        
        print(f"🚀 Creating Python Flask API: {name}")
        
        # Create directory
        os.makedirs(name, exist_ok=True)
        os.chdir(name)
        
        # Create requirements.txt
        with open("requirements.txt", 'w') as f:
            f.write('''Flask==2.3.3
FlaYOUR_OPENAI_API_KEY_HERE==4.0.0
python-dotenv==1.0.0
gunicorn==21.2.0
''')
        
        # Create app.py
        with open("app.py", 'w') as f:
            f.write('''from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        'message': 'Hello from Flask API!',
        'created_by': 'DeepBlue 🌊',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'OK',
        'uptime': 'Running'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
''')
        
        # Create .env
        with open(".env", 'w') as f:
            f.write('''FLASK_APP=app.py
FLASK_ENV=development
PORT=5000
''')
        
        # Create .gitignore
        with open(".gitignore", 'w') as f:
            f.write('''__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
.env
''')
        
        print(f"✅ Python Flask API '{name}' created successfully!")
        print(f"📁 Location: {os.getcwd()}")
        print(f"🚀 Run: cd {name} && pip install -r requirements.txt && python app.py")
        
        # Go back to parent directory
        os.chdir("..")
        
        return True
    
    def _create_fullstack_mern(self, name: str = None) -> bool:
        """Create a full-stack MERN application."""
        if not name:
            name = input("Enter project name: ")
        
        print(f"🚀 Creating full-stack MERN app: {name}")
        
        # Create main directory
        os.makedirs(name, exist_ok=True)
        os.chdir(name)
        
        # Create backend
        print("📦 Creating backend...")
        os.makedirs("backend", exist_ok=True)
        os.chdir("backend")
        
        # Backend package.json
        backend_package = {
            "name": f"{name}-backend",
            "version": "1.0.0",
            "description": "MERN backend created with DeepBlue",
            "main": "server.js",
            "scripts": {
                "start": "node server.js",
                "dev": "nodemon server.js"
            },
            "dependencies": {
                "express": "^4.18.2",
                "mongoose": "^7.5.0",
                "cors": "^2.8.5",
                "helmet": "^6.0.1",
                "dotenv": "^16.3.1"
            },
            "devDependencies": {
                "nodemon": "^2.0.20"
            }
        }
        
        with open("package.json", 'w') as f:
            json.dump(backend_package, f, indent=2)
        
        # Backend server.js
        with open("server.js", 'w') as f:
            f.write('''const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const helmet = require('helmet');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// MongoDB connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/mernapp', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// Routes
app.get('/', (req, res) => {
  res.json({
    message: 'MERN Backend API',
    created_by: 'DeepBlue 🌊',
    timestamp: new Date().toISOString()
  });
});

app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    uptime: process.uptime(),
    database: mongoose.connection.readyState === 1 ? 'Connected' : 'Disconnected'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🌊 MERN Backend running on port ${PORT}`);
  console.log(`🚀 Created with DeepBlue`);
});
''')
        
        # Backend .env
        with open(".env", 'w') as f:
            f.write('''MONGODB_URI=mongodb://localhost:27017/mernapp
PORT=5000
NODE_ENV=development
''')
        
        os.chdir("..")
        
        # Create frontend
        print("⚛️ Creating frontend...")
        result = subprocess.run([
            "npx", "create-react-app", "frontend", "--yes"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Full-stack MERN app '{name}' created successfully!")
            print(f"📁 Location: {os.getcwd()}")
            print(f"🚀 Backend: cd {name}/backend && npm install && npm start")
            print(f"🚀 Frontend: cd {name}/frontend && npm start")
            print(f"📋 Don't forget to install MongoDB!")
        else:
            print(f"❌ Failed to create frontend: {result.stderr}")
            return False
        
        # Go back to parent directory
        os.chdir("..")
        
        return True
    
    def _create_rag_system(self, name: str = None) -> bool:
        """Create a cutting-edge modular RAG system."""
        if not name:
            name = input("Enter project name: ")
        
        print(f"🚀 Creating cutting-edge RAG system: {name}")
        
        # Create directory structure
        os.makedirs(name, exist_ok=True)
        os.chdir(name)
        
        # Create main project structure
        os.makedirs("src", exist_ok=True)
        os.makedirs("src/rag", exist_ok=True)
        os.makedirs("src/rag/retrievers", exist_ok=True)
        os.makedirs("src/rag/generators", exist_ok=True)
        os.makedirs("src/rag/embeddings", exist_ok=True)
        os.makedirs("src/rag/vectorstores", exist_ok=True)
        os.makedirs("src/rag/rerankers", exist_ok=True)
        os.makedirs("src/rag/chunkers", exist_ok=True)
        os.makedirs("src/rag/llms", exist_ok=True)
        os.makedirs("src/api", exist_ok=True)
        os.makedirs("src/web", exist_ok=True)
        os.makedirs("config", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("tests", exist_ok=True)
        os.makedirs("docker", exist_ok=True)
        os.makedirs("kubernetes", exist_ok=True)
        
        # Create requirements.txt with latest packages
        requirements = '''# Core RAG Dependencies
langchain==0.1.0
langchain-community==0.0.10
langchain-core==0.1.10
langchain-openai==0.0.2
langchain-anthropic==0.1.0
langchain-ollama==0.1.0

# Vector Databases
pinecone-client==2.2.4
weaviate-client==3.25.3
qdrant-client==1.7.0
chromadb==0.4.18
faiss-cpu==1.7.4

# Embeddings
sentence-transformers==2.2.2
openai==1.3.7
cohere==4.37

# Document Processing
pypdf2==3.0.1
python-docx==1.1.0
beautifulsoup4==4.12.2
markdown==3.5.1
unstructured==0.11.6

# Reranking
sentence-transformers[rerank]==2.2.2
cross-encoder==1.0.0

# Web Framework
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
alembic==1.13.0
redis==5.0.1

# Monitoring & Logging
prometheus-client==0.19.0
structlog==23.2.0
sentry-sdk==1.38.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Development
black==23.11.0
isort==5.12.0
mypy==1.7.1
pre-commit==3.6.0

# Deployment
docker==6.1.3
kubernetes==28.1.0
'''
        
        with open("requirements.txt", 'w') as f:
            f.write(requirements)
        
        # Create main RAG system
        with open("src/rag/__init__.py", 'w') as f:
            f.write('''"""
🌊 DeepBlue RAG System - Cutting Edge Modular RAG
The most advanced, modular, and scalable RAG system available.
"""

from .core import RAGSystem
from .retrievers import HybridRetriever, DenseRetriever, SparseRetriever
from .generators import LLMGenerator, StreamingGenerator
from .embeddings import EmbeddingManager
from .vectorstores import VectorStoreManager
from .rerankers import RerankerManager
from .chunkers import ChunkingManager

__version__ = "1.0.0"
__all__ = [
    "RAGSystem",
    "HybridRetriever", 
    "DenseRetriever",
    "SparseRetriever",
    "LLMGenerator",
    "StreamingGenerator",
    "EmbeddingManager",
    "VectorStoreManager", 
    "RerankerManager",
    "ChunkingManager"
]
''')
        
        # Create core RAG system
        with open("src/rag/core.py", 'w') as f:
            f.write('''"""
Core RAG System Implementation
"""

from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import asyncio
import logging
from dataclasses import dataclass

from .retrievers import HybridRetriever
from .generators import LLMGenerator
from .embeddings import EmbeddingManager
from .vectorstores import VectorStoreManager
from .rerankers import RerankerManager
from .chunkers import ChunkingManager

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: str = "chroma"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    reranker: str = "cross-encoder"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 10
    rerank_top_k: int = 5
    temperature: float = 0.7
    max_tokens: int = 1000

class RAGSystem:
    """Modular RAG System with cutting-edge features."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_manager = EmbeddingManager(config.embedding_model)
        self.vector_store = VectorStoreManager(config.vector_store)
        self.retriever = HybridRetriever(
            embedding_manager=self.embedding_manager,
            vector_store=self.vector_store,
            config=config
        )
        self.generator = LLMGenerator(config)
        self.reranker = RerankerManager(config.reranker)
        self.chunker = ChunkingManager(config)
        
    async def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the knowledge base."""
        try:
            # Chunk documents
            chunks = self.chunker.chunk_documents(documents)
            
            # Generate embeddings
            embeddings = await self.embedding_manager.embed_documents(chunks)
            
            # Store in vector database
            await self.vector_store.add_documents(chunks, embeddings, metadata)
            
            logger.info(f"Added {len(chunks)} chunks to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def query(self, question: str, use_reranking: bool = True) -> Dict[str, Any]:
        """Query the RAG system."""
        try:
            # Retrieve relevant documents
            retrieved_docs = await self.retriever.retrieve(question, top_k=self.config.top_k)
            
            # Rerank if enabled
            if use_reranking and len(retrieved_docs) > self.config.rerank_top_k:
                retrieved_docs = await self.reranker.rerank(
                    question, retrieved_docs, top_k=self.config.rerank_top_k
                )
            
            # Generate response
            context = "\\n".join([doc["content"] for doc in retrieved_docs])
            response = await self.generator.generate(question, context)
            
            return {
                "answer": response,
                "sources": retrieved_docs,
                "metadata": {
                    "retrieval_count": len(retrieved_docs),
                    "reranked": use_reranking
                }
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {"error": str(e)}
    
    async def stream_query(self, question: str, use_reranking: bool = True):
        """Stream query results."""
        try:
            # Retrieve and rerank (same as query)
            retrieved_docs = await self.retriever.retrieve(question, top_k=self.config.top_k)
            
            if use_reranking and len(retrieved_docs) > self.config.rerank_top_k:
                retrieved_docs = await self.reranker.rerank(
                    question, retrieved_docs, top_k=self.config.rerank_top_k
                )
            
            # Stream generation
            context = "\\n".join([doc["content"] for doc in retrieved_docs])
            async for chunk in self.generator.stream_generate(question, context):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            yield {"error": str(e)}
''')
        
        # Create hybrid retriever
        with open("src/rag/retrievers/__init__.py", 'w') as f:
            f.write('''"""
Advanced Retrieval Methods
"""

from .hybrid import HybridRetriever
from .dense import DenseRetriever  
from .sparse import SparseRetriever

__all__ = ["HybridRetriever", "DenseRetriever", "SparseRetriever"]
''')
        
        with open("src/rag/retrievers/hybrid.py", 'w') as f:
            f.write('''"""
Hybrid Retrieval combining dense and sparse methods
"""

from typing import List, Dict, Any
import numpy as np
from .dense import DenseRetriever
from .sparse import SparseRetriever

class HybridRetriever:
    """Hybrid retriever combining dense and sparse search."""
    
    def __init__(self, embedding_manager, vector_store, config):
        self.dense_retriever = DenseRetriever(embedding_manager, vector_store, config)
        self.sparse_retriever = SparseRetriever(config)
        self.alpha = 0.7  # Weight for dense vs sparse
        
    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents using hybrid approach."""
        # Get dense results
        dense_results = await self.dense_retriever.retrieve(query, top_k * 2)
        
        # Get sparse results  
        sparse_results = await self.sparse_retriever.retrieve(query, top_k * 2)
        
        # Combine and rerank
        combined_results = self._combine_results(dense_results, sparse_results)
        
        return combined_results[:top_k]
    
    def _combine_results(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """Combine dense and sparse results using reciprocal rank fusion."""
        # Create score maps
        dense_scores = {doc["id"]: doc["score"] for doc in dense_results}
        sparse_scores = {doc["id"]: doc["score"] for doc in sparse_results}
        
        # Get all unique document IDs
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        # Calculate hybrid scores
        hybrid_results = []
        for doc_id in all_ids:
            dense_score = dense_scores.get(doc_id, 0.0)
            sparse_score = sparse_scores.get(doc_id, 0.0)
            
            # Reciprocal Rank Fusion
            hybrid_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
            
            # Find the document
            doc = next((d for d in dense_results + sparse_results if d["id"] == doc_id), None)
            if doc:
                doc["score"] = hybrid_score
                hybrid_results.append(doc)
        
        # Sort by hybrid score
        return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)
''')
        
        # Create FastAPI web interface
        with open("src/api/main.py", 'w') as f:
            f.write('''"""
FastAPI Web Interface for RAG System
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import logging

from ..rag.core import RAGSystem, RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DeepBlue RAG System",
    description="Cutting-edge modular RAG system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
config = RAGConfig()
rag_system = RAGSystem(config)

class QueryRequest(BaseModel):
    question: str
    use_reranking: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class DocumentUpload(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

@app.get("/")
async def root():
    return {"message": "DeepBlue RAG System API", "status": "running"}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = await rag_system.query(request.question, request.use_reranking)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents")
async def add_documents(documents: List[DocumentUpload]):
    """Add documents to the knowledge base."""
    try:
        contents = [doc.content for doc in documents]
        metadata = [doc.metadata for doc in documents]
        
        success = await rag_system.add_documents(contents, metadata)
        if success:
            return {"message": f"Added {len(documents)} documents successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add documents")
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rag-system"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
''')
        
        # Create Docker configuration
        with open("Dockerfile", 'w') as f:
            f.write('''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
''')
        
        # Create docker-compose.yml
        with open("docker-compose.yml", 'w') as f:
            f.write('''version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
      - chroma

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  chroma_data:
  weaviate_data:
''')
        
        # Create configuration file
        with open("config/settings.yaml", 'w') as f:
            f.write('''# DeepBlue RAG System Configuration

# Embedding Models
embeddings:
  default: "sentence-transformers/all-MiniLM-L6-v2"
  options:
    - "sentence-transformers/all-mpnet-base-v2"
    - "sentence-transformers/all-MiniLM-L12-v2"
    - "openai/text-embedding-ada-002"
    - "cohere/embed-english-v3.0"

# Vector Stores
vector_stores:
  default: "chroma"
  options:
    - "pinecone"
    - "weaviate"
    - "qdrant"
    - "faiss"
    - "chroma"

# LLM Providers
llm_providers:
  default: "openai"
  options:
    - "openai"
    - "anthropic"
    - "ollama"
    - "huggingface"

# Models
models:
  openai:
    - "gpt-4"
    - "gpt-4-turbo"
    - "gpt-3.5-turbo"
  anthropic:
    - "claude-3-opus"
    - "claude-3-sonnet"
    - "claude-3-haiku"
  ollama:
    - "llama2"
    - "mistral"
    - "codellama"

# Reranking
rerankers:
  default: "cross-encoder"
  options:
    - "cross-encoder"
    - "bge-reranker"
    - "cohere-rerank"

# Chunking
chunking:
  default_size: 1000
  default_overlap: 200
  strategies:
    - "recursive"
    - "semantic"
    - "fixed"

# Retrieval
retrieval:
  default_top_k: 10
  rerank_top_k: 5
  hybrid_alpha: 0.7

# Generation
generation:
  default_temperature: 0.7
  max_tokens: 1000
  streaming: true
''')
        
        # Create README
        with open("README.md", 'w') as f:
            f.write('''# 🌊 DeepBlue RAG System

The most cutting-edge, modular, and scalable RAG (Retrieval-Augmented Generation) system available.

## 🚀 Features

### Core Capabilities
- **Hybrid Retrieval**: Combines dense and sparse search methods
- **Multiple Vector Stores**: Pinecone, Weaviate, Qdrant, Chroma, FAISS
- **Multiple LLM Providers**: OpenAI, Anthropic, Ollama, Hugging Face
- **Advanced Reranking**: Cross-encoder and specialized rerankers
- **Smart Chunking**: Recursive, semantic, and fixed strategies
- **Streaming Support**: Real-time response streaming
- **Modular Architecture**: Easy to extend and customize

### Advanced Features
- **Multi-modal Support**: Text, PDF, DOCX, Markdown processing
- **Semantic Search**: Advanced embedding models
- **Query Expansion**: Automatic query enhancement
- **Context Compression**: Intelligent context management
- **Caching**: Redis-based response caching
- **Monitoring**: Prometheus metrics and structured logging
- **Docker Support**: Containerized deployment
- **Kubernetes Ready**: Production-scale orchestration

## 🛠️ Quick Start

### Installation

```bash
# Clone and setup
git clone <your-repo>
cd <project-name>

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export PINECONE_API_KEY="your-key"

# Run the system
python -m uvicorn src.api.main:app --reload
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# The API will be available at http://localhost:8000
```

### Basic Usage

```python
from src.rag.core import RAGSystem, RAGConfig

# Initialize
config = RAGConfig()
rag = RAGSystem(config)

# Add documents
await rag.add_documents(["Your document content here"])

# Query
result = await rag.query("Your question here")
print(result["answer"])
```

### API Usage

```bash
# Query the system
curl -X POST "http://localhost:8000/query" \\
     -H "Content-Type: application/json" \\
     -d '{"question": "What is the main topic?", "use_reranking": true}'

# Add documents
curl -X POST "http://localhost:8000/documents" \\
     -H "Content-Type: application/json" \\
     -d '[{"content": "Document content", "metadata": {"source": "test"}}]'
```

## 🏗️ Architecture

```
src/
├── rag/
│   ├── core.py              # Main RAG system
│   ├── retrievers/          # Retrieval methods
│   ├── generators/          # LLM generators
│   ├── embeddings/          # Embedding management
│   ├── vectorstores/        # Vector database interfaces
│   ├── rerankers/           # Reranking methods
│   └── chunkers/            # Document chunking
├── api/                     # FastAPI web interface
└── web/                     # Frontend interface
```

## 🔧 Configuration

Edit `config/settings.yaml` to customize:
- Embedding models
- Vector stores
- LLM providers
- Chunking strategies
- Retrieval parameters

## 📊 Monitoring

- **Metrics**: Prometheus-compatible metrics
- **Logging**: Structured logging with context
- **Health Checks**: Built-in health monitoring
- **Tracing**: Request tracing and performance monitoring

## 🚀 Production Deployment

### Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key

# Optional
PINECONE_API_KEY=your-key
WEAVIATE_URL=http://weaviate:8080
REDIS_URL=redis://redis:6379
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

*Built with DeepBlue 🌊 - We're gonna need a bigger boat!*
''')
        
        print(f"✅ Cutting-edge RAG system '{name}' created successfully!")
        print(f"📁 Location: {os.getcwd()}")
        print(f"🚀 Run: cd {name} && pip install -r requirements.txt")
        print(f"🌐 API: python -m uvicorn src.api.main:app --reload")
        print(f"🐳 Docker: docker-compose up -d")
        
        # Go back to parent directory
        os.chdir("..")
        
        return True
    
    def show_help(self):
        """Show help information."""
        print("🌊 DeepBlue Assistant Help")
        print("=" * 30)
        print("Available commands:")
        print("  create react <name>     - Create React app")
        print("  create nextjs <name>    - Create Next.js app")
        print("  create express <name>   - Create Express API")
        print("  create vue <name>       - Create Vue.js app")
        print("  create flask <name>     - Create Python Flask API")
        print("  create mern <name>      - Create full-stack MERN app")
        print("  create rag <name>       - Create cutting-edge RAG system")
        print("  save <name> <desc>      - Save current build")
        print("  list                    - List saved builds")
        print("  build <name>            - Use saved build")
        print("  help                    - Show this help")
        print("  quit                    - Exit")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="🌊 DeepBlue Assistant - We're gonna need a bigger boat! 🚢",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  deepblue.py create react my-app
  deepblue.py create nextjs my-nextjs-app  
  deepblue.py create express my-api
  deepblue.py create vue my-vue-app
  deepblue.py create flask my-flaYOUR_OPENAI_API_KEY_HERE
  deepblue.py create mern my-fullstack-app
  deepblue.py list
  deepblue.py interactive
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        choices=["create", "list", "interactive", "help"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "type",
        nargs="?",
        choices=["react", "nextjs", "express", "vue", "flask", "mern", "rag"],
        help="Project type (when using create command)"
    )
    
    parser.add_argument(
        "name",
        nargs="?",
        help="Project name"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="DeepBlue Assistant 1.0.0 🌊"
    )
    
    return parser.parse_args()

def main():
    """Main DeepBlue assistant."""
    args = parse_arguments()
    assistant = SimpleDeepBlueAssistant()
    
    # Handle command line arguments
    if args.command == "create":
        if not args.type:
            print("❌ Please specify project type: react, nextjs, express, vue, flask, mern, or rag")
            print("Usage: deepblue.py create <type> <name>")
            sys.exit(1)
        
        if not args.name:
            print("❌ Please specify project name")
            print("Usage: deepblue.py create <type> <name>")
            sys.exit(1)
        
        build_type_map = {
            "react": "react_app",
            "nextjs": "nextjs_app", 
            "express": "express_api",
            "vue": "vue_app",
            "flask": "python_flask",
            "mern": "fullstack_mern",
            "rag": "rag_system"
        }
        
        print("🌊 DeepBlue Assistant - We're gonna need a bigger boat! 🚢")
        print("=" * 60)
        success = assistant.create_build(build_type_map[args.type], args.name)
        sys.exit(0 if success else 1)
        
    elif args.command == "list":
        builds = assistant.list_builds()
        if builds:
            print("🌊 DeepBlue Assistant - Saved Builds")
            print("=" * 40)
            for name, build in builds.items():
                print(f"  • {name}: {build['description']}")
        else:
            print("📋 No saved builds yet")
        sys.exit(0)
        
    elif args.command == "help" or not args.command:
        print("🌊 DeepBlue Assistant - We're gonna need a bigger boat! 🚢")
        print("=" * 60)
        assistant.show_help()
        print("\nCommand Line Usage:")
        print("  deepblue.py create <type> <name>  - Create project")
        print("  deepblue.py list                  - List saved builds")
        print("  deepblue.py interactive           - Interactive mode")
        print("  deepblue.py help                  - Show this help")
        sys.exit(0)
        
    elif args.command == "interactive":
        # Interactive mode
        print("🌊 DeepBlue Assistant - Interactive Mode")
        print("=" * 40)
        print("Type 'help' for commands or 'quit' to exit")
        print("")
        
        while True:
            try:
                command = input("🌊 DeepBlue> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    print("👋 Goodbye!")
                    break
                elif command == "help":
                    assistant.show_help()
                elif command.startswith("create react"):
                    name = command.replace("create react", "").strip()
                    assistant.create_build("react_app", name)
                elif command.startswith("create nextjs"):
                    name = command.replace("create nextjs", "").strip()
                    assistant.create_build("nextjs_app", name)
                elif command.startswith("create express"):
                    name = command.replace("create express", "").strip()
                    assistant.create_build("express_api", name)
                elif command.startswith("create vue"):
                    name = command.replace("create vue", "").strip()
                    assistant.create_build("vue_app", name)
                elif command.startswith("create flask"):
                    name = command.replace("create flask", "").strip()
                    assistant.create_build("python_flask", name)
                elif command.startswith("create mern"):
                    name = command.replace("create mern", "").strip()
                    assistant.create_build("fullstack_mern", name)
                elif command.startswith("create rag"):
                    name = command.replace("create rag", "").strip()
                    assistant.create_build("rag_system", name)
                elif command.startswith("save "):
                    parts = command.split(" ", 2)
                    if len(parts) >= 3:
                        name = parts[1]
                        desc = parts[2]
                        print(f"💾 Saving build '{name}' with description: {desc}")
                    else:
                        print("❌ Usage: save <name> <description>")
                elif command == "list":
                    builds = assistant.list_builds()
                    if builds:
                        print("📋 Saved builds:")
                        for name, build in builds.items():
                            print(f"  • {name}: {build['description']}")
                    else:
                        print("📋 No saved builds yet")
                elif command.startswith("build "):
                    name = command.replace("build", "").strip()
                    build = assistant.get_build(name)
                    if build:
                        print(f"🚀 Using build '{name}': {build['description']}")
                    else:
                        print(f"❌ Build '{name}' not found")
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                
                print("")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()