"""Start the RAG pipeline FastAPI server."""

import uvicorn

from rag_pipeline.config import load_config

if __name__ == "__main__":
    config = load_config()
    uvicorn.run(
        "rag_pipeline.api.app:app",
        host=config.api_host,
        port=config.api_port,
        reload=True,
    )
