# Retrieval Pipeline

## Ingestion and chunking

Documents in the data directory with a `.md` or `.txt` suffix are read at
startup. Each document is split on paragraph boundaries and then packed into
word-bounded chunks of 180 words with a 40-word sliding overlap. The overlap
exists so that a sentence straddling a chunk boundary is still retrievable from
either side; without it, answers that depend on a boundary sentence are lost.

Ingestion is fingerprinted. The fingerprint hashes every document's contents
together with the chunk size, the chunk overlap, and the name of the embedding
model. If the fingerprint is unchanged and the collection already holds the
expected number of chunks, ingestion is skipped entirely, so restarting a
container does not re-embed an unchanged corpus. Changing the embedding model
changes the fingerprint and therefore forces a rebuild, which is correct:
vectors from two different models are not comparable.

A rebuild deletes and recreates the collection rather than upserting in place.
That is the only way a deleted source document or a shrunken document actually
disappears from the index.

## Embeddings

The default embedding backend is BAAI/bge-small-en-v1.5 executed by ONNX
Runtime through fastembed. It produces 384-dimensional vectors, needs no API
key and no PyTorch, and runs on CPU or CoreML. A hosted OpenAI backend using
text-embedding-3-small is available for parity with a managed stack, and a
deterministic hashing backend exists purely so tests and CI can run without
downloading weights.

## Vector database

Vectors are stored in ChromaDB in persistent mode, in a collection configured
for cosine distance with an HNSW index. Queries return cosine distance, which
the retriever converts to a similarity score of one minus the distance so that
higher is better. Matches below a configurable minimum score are dropped before
they reach the prompt, because padding a prompt with irrelevant chunks makes
answers worse, not better.
