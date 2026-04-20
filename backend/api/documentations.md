# Audio Preprocessor API — System Documentation

## 1. Overview

The system is a FastAPI-based backend that implements a multi-stage pipeline for processing audio and text data. It supports transcription, preprocessing, optional automation workflows, external tool integrations, and persistent storage via JSONL files.

The architecture is divided into distinct layers:

* Transcription layer (audio/text → raw text)
* Preprocessing layer (text → structured output)
* Tool layer (post-processing utilities)
* Persistence layer (history storage)
* Retrieval layer (accessing stored records)
* Workflow layer (combined operations)

---

## 2. Core Data Models

### 2.1 Transcription Object

Represents raw transcribed or user-provided text.

```json
{
  "id": "uuid",
  "name": "string",
  "transcription": "string",
  "timestamp": "string"
}
```

### 2.2 Preprocessed Result

Represents structured output generated from transcription.

```json
{
  "id": "uuid",
  "name": "string",
  "processed_data": "string"
}
```

---

## 3. Endpoints

## 3.1 Transcription Endpoints

### POST /transcribe/audio

**Responsibility**

* Accepts an audio file
* Performs speech-to-text transcription
* Saves result to persistent storage (transcriptions.jsonl)
* Returns transcription object

**Input**

* multipart/form-data
* file: audio file (mp3, wav, m4a, etc.)

**Output**

```json
{
  "status": "success",
  "message": "string",
  "data": {
    "id": "uuid",
    "name": "string",
    "transcription": "string",
    "timestamp": "string"
  }
}
```

**Significance**

* Primary entry point for audio-based processing
* Automatically persists data for historical access

---

### POST /transcribe/text

**Responsibility**

* Accepts raw text input
* Converts it into a transcription object
* Saves it to persistent storage (transcriptions.jsonl)
* Returns transcription object

**Input**

```json
{
  "name": "string",
  "transcription": "string"
}
```

**Output**

```json
{
  "status": "success",
  "message": "string",
  "data": Transcription
}
```

**Significance**

* Enables text-based workflow without audio processing
* Provides alternative entry point for transcription system

---

## 3.2 Preprocessing Endpoint

### POST /process

**Responsibility**

* Accepts transcription object
* Applies LLM-based preprocessing
* Returns structured processed output

**Input**

```json
{
  "id": "uuid",
  "name": "string",
  "transcription": "string"
}
```

**Output**

```json
{
  "status": "success",
  "message": "string",
  "data": {
    "id": "uuid",
    "name": "string",
    "processed_data": "string"
  }
}
```

**Significance**

* Core intelligence layer of the system
* Stateless (does not read from database)

---

## 3.3 Combined Workflow Endpoint

### POST /transcribe-and-process/audio

**Responsibility**

* Executes full pipeline in one request:

  1. Audio transcription
  2. Preprocessing of result
* Returns both outputs

**Input**

* multipart/form-data
* file: audio file

**Output**

```json
{
  "status": "success",
  "transcription": Transcription,
  "preprocessed": PreprocessedResult
}
```

**Significance**

* Convenience endpoint for automation
* Bypasses manual step-by-step workflow

---

## 3.4 Tool Endpoints

### POST /send-email

**Responsibility**

* Sends processed data via external email service (n8n integration)

**Input**

```json
{
  "to": "string",
  "subject": "string",
  "processed_data": "string",
  "user_message": "string",
  "sender": "string"
}
```

**Output**

```json
{
  "status": "success",
  "message": "string",
  "email": "string"
}
```

---

### POST /extract-text

**Responsibility**

* Extracts keywords and keypoints from processed data

**Input**

```json
{
  "processed_data": "string"
}
```

**Output**

```json
{
  "status": "success",
  "data": {
    "keywords": [],
    "keypoints": []
  }
}
```

---

### POST /translate

**Responsibility**

* Translates processed data into a target language

**Input**

```json
{
  "language": "string",
  "processed_data": "string"
}
```

**Output**

```json
{
  "status": "success",
  "translated_data": "string"
}
```

---

## 3.5 Retrieval Endpoints

### GET /transcriptions/{id}

**Responsibility**

* Retrieves a stored transcription by ID from JSONL storage

**Input**

* Path parameter: transcription_id

**Output**

```json
{
  "status": "success",
  "data": Transcription
}
```

---

### GET /preprocessings/{id}

**Responsibility**

* Retrieves a stored preprocessing result by ID

**Input**

* Path parameter: preprocessing_id

**Output**

```json
{
  "status": "success",
  "data": PreprocessedResult
}
```

---

## 3.6 System Endpoints

### GET /

**Responsibility**

* Basic health check endpoint for API availability

### GET /health

**Responsibility**

* Returns system readiness status of core modules (transcriber, preprocessor)

---

## 4. Data Flow Summary

### Standard Workflow

```
Audio/Text
   ↓
Transcription Layer
   ↓
Preprocessing Layer
   ↓
Tool Layer (optional)
```

### Persistence Flow

```
Transcriber → writes to transcriptions.jsonl
Preprocessor → (optionally extendable to storage)
Retrieval endpoints → read from JSONL files
```

---

## 5. Architecture Decision: Modularization Approach

The system will be refactored using a **router-based modular architecture**.

### 5.1 Target Structure

```
api/
  main.py

  routes/
    transcribe.py
    process.py
    tools.py
    retrieval.py
    workflow.py

```

---

### 5.2 Design Principles

* `main.py` acts only as an application entry point and router registry
* Each domain (transcription, preprocessing, tools, retrieval) has its own router module
* Core logic (Transcriber, Preprocessor) is isolated from FastAPI

---

### 5.3 Integration Pattern

```
main.py
  → include_router()

routes/
  → define API endpoints only

core/tools/databases/
  → contain all business logic and persistence
```

---

## 6. Key Architectural Insight

The system is designed as a **pipeline with optional persistence**, not a tightly coupled application.

* Pipeline can operate without database (stateless mode)
* Database provides historical access and recovery (stateful mode)
* Tool layer extends processed outputs into external actions

---

## 7. Conclusion

The system is functionally complete as a backend pipeline but requires modular refactoring to improve:

* Maintainability
* Scalability
* Separation of concerns
* Testability
* Future extension (AI agents, batch processing, external integrations)

The modular router-based architecture has been selected as the target refactor strategy.
