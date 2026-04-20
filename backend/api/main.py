from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from api.routes.transcriber_endpoints import router as transcriber_router
from api.routes.process_endpoints import router as process_router
from api.routes.tools_endpoints import router as tools_router
from api.routes.workflow_endpoints import router as workflow_router
from api.routes.retrieval_endpoints import router as retrieval_router

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Audio Preprocessor API is starting up...")
    logger.info("Transcriber loaded with model: base")
    logger.info("Preprocessor initialized")
    yield
    logger.info("Shutting down Audio Preprocessor API...")


app = FastAPI(
    title="Audio Preprocessor API",
    description="API for transcribing and preprocessing audio files",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcriber_router)
app.include_router(process_router)
app.include_router(tools_router)
app.include_router(workflow_router)
app.include_router(retrieval_router)


@app.get("/")
async def root():
    return {
        "status": "SUCCESS",
        "service": "Audio Preprocessor API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "SUCCESS",
        "transcriber": "ready",
        "preprocessor": "ready",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
