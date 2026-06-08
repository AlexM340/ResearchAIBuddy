# ResearchAIBuddy Architecture Design

This document captures the current architecture as an editable set of Mermaid
diagrams. It reflects the codebase shape as a modular Streamlit monolith with
external storage and AI integrations.

## 1. System Context

```mermaid
flowchart LR
    User[User] --> UI[Streamlit Web UI]

    UI --> App[CerebrumAISystem\nApplication Orchestrator]

    App --> Gemini[Google Gemini\nLLM + HyDE + query synthesis]
    App --> Tavily[Tavily\nWeb search fallback + suggestions]
    App --> Postgres[(Postgres + pgvector\nDocuments, chunks, chats, memory, tasks)]
    App --> Neo4j[(Neo4j\nKnowledge graph)]
    App --> FS[(Local filesystem\nDocument library, uploads, caches)]

    Postgres --> PgVector[pgvector HNSW\nVector similarity search]
```

## 2. Container View

```mermaid
flowchart TB
    subgraph Presentation["Presentation Layer"]
        Main[src/main_flash.py\nBootstrap, config, navigation, sidebar]
        Views[src/views/*\nSecond Brain, Notebooks, Tasks, Graph, Logs]
        ViewHelpers[src/views/_shared.py\nChat lifecycle, rendering, question handling]
        ViewCache[src/views/_cache.py\nShort TTL Streamlit read cache]
    end

    subgraph Application["Application Layer"]
        System[src/rag_module_flash.py\nCerebrumAISystem]
        LLM[OptimizedFlashLLM\nGemini wrapper]
        Retriever[SimpleRetriever\nIn-memory vector retrieval]
        Processor[SimpleDocumentProcessor\nLoad and chunk documents]
        Reranker[MultilingualReranker\nCrossEncoder lazy rerank]
        RespCache[ResponseCache\nDiskcache or memory fallback]
        EmbedCache[PersistentEmbeddingsCache\nFile hash + model signature]
    end

    subgraph QueryIntel["Query Intelligence Layer"]
        Router[src/query/intent_router.py\nIntent classification]
        Fusion[src/query/hybrid_retriever.py\nVector + graph evidence fusion]
        Context[src/query/context_builder.py\nContext budget + citations]
    end

    subgraph Storage["Storage Layer"]
        PgClient[src/storage/postgres_client.py\nConnection and migrations]
        Repo[src/storage/repositories.py\nRepository gateway]
        Schema[src/storage/schema.sql + migrations\nDB schema]
    end

    subgraph Graph["Graph Layer"]
        NeoClient[src/graph/neo4j_client.py\nNeo4j driver wrapper]
        GraphIngest[src/graph/graph_ingestion.py\nConcept/relation ingestion]
        GraphQuery[src/graph/graph_query.py\nGraph retrieval]
    end

    subgraph Eval["Evaluation Layer"]
        EvalRunner[src/eval/runner.py\nA/B evaluation]
        Metrics[src/eval/metrics.py\nQuality metrics]
    end

    Main --> Views
    Views --> ViewHelpers
    Views --> ViewCache
    ViewHelpers --> System

    System --> LLM
    System --> Retriever
    System --> Processor
    System --> Reranker
    System --> RespCache
    Retriever --> EmbedCache
    System --> Router
    System --> Fusion
    System --> Context
    System --> Repo
    Repo --> PgClient
    PgClient --> Schema
    System --> NeoClient
    System --> GraphIngest
    System --> GraphQuery
    EvalRunner --> System
    EvalRunner --> Metrics
```

## 3. Query Data Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit View
    participant Shared as views/_shared.py
    participant System as CerebrumAISystem
    participant Router as Intent Router
    participant Pg as Postgres + pgvector
    participant Graph as Neo4j
    participant Memory as Memory/Notes Search
    participant Ctx as Context Builder
    participant LLM as Gemini
    participant Tavily as Tavily
    participant Repo as Repository

    User->>UI: Ask question
    UI->>Shared: process_question()
    Shared->>System: query(question, retrieval_mode, scope)

    System->>System: Compute shared query embedding
    System->>Pg: Vector search chunks
    Pg-->>System: Vector candidates

    System->>Router: Classify intent and build plan
    Router-->>System: vector / hybrid / external route

    alt Hybrid route
        System->>Graph: Retrieve graph paths
        Graph-->>System: Graph evidence
    end

    System->>Memory: Search preferences, decisions, notes, tasks
    Memory-->>System: Memory evidence

    alt No sufficient internal context or force_web
        System->>Tavily: Web search
        Tavily-->>System: External sources
    end

    System->>Ctx: Build bounded context + provenance
    Ctx-->>System: Prompt + citation map

    System->>System: Check response cache
    alt Cache miss
        System->>LLM: Generate grounded answer
        LLM-->>System: Answer
        System->>System: Store response cache
    end

    System->>System: Validate citations
    System->>System: Extract memory/task/note candidates
    System->>Repo: Persist retrieval log and memory side effects
    System-->>Shared: Query result
    Shared->>Repo: Append chat exchange
    Shared-->>UI: Rerender chat
```

## 4. Document Ingestion Flow

```mermaid
flowchart TD
    Upload[User uploads file\nor adds web URL] --> Library[DocumentManager\nLocal document library]
    Library --> Metadata[Build metadata\ncollection, hash, source path]

    Metadata --> Filter[CerebrumAISystem.filter_paths_for_ingestion]
    Filter --> RepoHash[Repository.list_missing_hashes]
    RepoHash --> Decision{Already indexed?}

    Decision -- Yes --> MarkIndexed[Mark local document indexed]
    Decision -- No --> Load[SimpleDocumentProcessor.load_documents]

    Load --> Chunk[Chunk documents]
    Chunk --> Embed[SentenceTransformer embeddings\nbatched for storage]
    Embed --> Cache[PersistentEmbeddingsCache\nfile hash + embeddings]
    Embed --> Retriever[SimpleRetriever\nin-memory index]
    Embed --> Pg[(Postgres documents/chunks/chunk_embeddings)]
    Chunk --> Graph{Graph enabled?}
    Graph -- Yes --> Neo4j[GraphIngestionService\nConcepts + relations]
    Graph -- No --> Done[Done]
    Neo4j --> Done
    Pg --> Done
    Retriever --> Done
    Cache --> Done
```

## 5. Memory And Side Effects Flow

```mermaid
flowchart LR
    Answer[Generated or cached answer] --> Extract[Candidate extraction]

    Extract --> Pref[Preference candidates]
    Extract --> Dec[Decision candidates]
    Extract --> Task[Task candidates]
    Extract --> Note[Note candidates]
    Extract --> Episode[Episodic capture]

    Pref --> Gate{Confidence gate}
    Dec --> Gate
    Task --> Gate
    Note --> Gate

    Gate -- High confidence --> Persist[Persist canonical memory\npreferences, decisions, tasks, notes]
    Gate -- Medium confidence --> Proposal[Create memory proposal\nfor user review]
    Gate -- Low confidence --> Drop[Ignore]

    Episode --> Consolidate{Consolidation enabled?}
    Consolidate -- Yes --> Semantic[Promote useful episodes\nto semantic decisions]
    Consolidate -- No --> End[End]

    Persist --> Artifacts[Update memory_artifacts\nvector searchable memory]
    Proposal --> Inbox[Memory inbox UI]
    Semantic --> Artifacts
```

## 6. Recommended Target Architecture

```mermaid
flowchart TB
    subgraph UI["UI Layer"]
        Pages[Streamlit pages]
        Components[Reusable UI components]
    end

    subgraph AppServices["Application Services"]
        QueryService[QueryService]
        IngestionService[IngestionService]
        MemoryService[MemoryService]
        SuggestionService[SourceSuggestionService]
        AnalysisService[AnalysisService]
    end

    subgraph Domain["Domain/Core"]
        QueryPipeline[Retrieval pipeline]
        PromptBuilder[Prompt and citation builder]
        MemoryPolicy[Memory confidence policies]
        Models[Typed result models]
    end

    subgraph Infra["Infrastructure"]
        LLMGateway[LLM gateway]
        WebSearchGateway[Web search gateway]
        VectorStore[Vector store]
        GraphStore[Graph store]
        FileStore[File store]
        CacheStore[Cache store]
    end

    subgraph Repositories["Repository Interfaces"]
        ChatRepo[ChatRepository]
        DocumentRepo[DocumentRepository]
        MemoryRepo[MemoryRepository]
        TaskRepo[TaskRepository]
        NoteRepo[NoteRepository]
        RetrievalLogRepo[RetrievalLogRepository]
    end

    Pages --> Components
    Components --> AppServices
    AppServices --> Domain
    Domain --> Repositories
    Domain --> Infra
    Repositories --> VectorStore
    Repositories --> FileStore
    Repositories --> GraphStore
```

## 7. Key Design Notes

- Keep Streamlit views thin: render UI, collect input, call application services.
- Split `CerebrumAISystem` into focused services before adding more features.
- Keep retrieval, memory capture, answer generation, and persistence as separate
  pipeline stages.
- Use typed DTOs for query results and evidence instead of large unstructured
  dictionaries.
- Lazy-load heavy ML libraries so tests and UI boot do not import the whole ML
  stack unless retrieval actually needs it.
- Treat Postgres as the source of truth; keep local JSON/document library as an
  ingestion cache and fallback only.
