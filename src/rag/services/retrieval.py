"""RAG retrieval and answer generation service."""
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from rag.core.embedder import Embedder
from rag.core.llm import OllamaClient
from rag.db.chroma import ChromaClient
from rag.db.repositories import DocumentRepository
from rag.schemas import QueryResponse, SourceChunk


class RetrievalService:
    def __init__(
        self,
        embedder: Embedder,
        chroma: ChromaClient,
        llm: OllamaClient,
    ):
        self.embedder = embedder
        self.chroma = chroma
        self.llm = llm

    async def query(
        self,
        question: str,
        top_k: int,
        db: AsyncSession,
    ) -> QueryResponse:
        """
        Full query pipeline:
        1. Embed the question
        2. Query ChromaDB for top_k similar chunks
        3. Fetch document filenames from PostgreSQL using document_ids from chroma results
        4. Build context and prompt
        5. Generate answer via LLM
        6. Return QueryResponse with answer and sources (with similarity scores)
        """
        # Step 1: Embed the question
        query_embedding = await self.embedder.embed_query(question)

        # Step 2: Query ChromaDB for top_k similar chunks
        collection_size = await self.chroma.count()
        if collection_size == 0:
            return QueryResponse(answer="No documents have been ingested yet.", sources=[])
        effective_top_k = min(top_k, collection_size)
        results = await self.chroma.query(
            query_embedding=query_embedding,
            n_results=effective_top_k,
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # If no chunks found, return early
        if not ids:
            return QueryResponse(answer="No relevant documents found.", sources=[])

        # Step 3: Fetch document filenames from PostgreSQL using document_ids from chroma results
        doc_repo = DocumentRepository(db)
        doc_id_to_filename: dict[str, str] = {}
        for meta in metadatas:
            doc_id_str = meta.get("document_id", "")
            if doc_id_str and doc_id_str not in doc_id_to_filename:
                try:
                    doc_uuid = uuid.UUID(doc_id_str)
                except ValueError:
                    continue  # skip invalid metadata
                doc = await doc_repo.get_by_id(doc_uuid)
                doc_id_to_filename[doc_id_str] = doc.filename if doc else meta.get("filename", "")

        # Step 4: Build context and prompt
        context_chunks = list(documents)
        prompt = self.llm.build_rag_prompt(question=question, context_chunks=context_chunks)

        # Step 5: Generate answer via LLM
        answer = await self.llm.generate(prompt)

        # Step 6: Build sources with similarity scores (score = 1 - distance)
        sources = []
        for text, meta, distance in zip(documents, metadatas, distances):
            score = 1.0 - distance
            doc_id_str = meta.get("document_id", "")
            filename = doc_id_to_filename.get(doc_id_str, meta.get("filename", ""))
            sources.append(
                SourceChunk(
                    document_id=uuid.UUID(doc_id_str),
                    filename=filename,
                    chunk_index=meta.get("chunk_index", 0),
                    text=text,
                    score=score,
                )
            )

        return QueryResponse(answer=answer, sources=sources)
