# Engineering AI Systems

Retrieval-Augmented Generation combines a retriever with a generator. Documents are split into chunks, embedded as vectors, and searched using similarity before the language model writes an answer.

Production assistants need boundaries around the model. Structured output makes responses predictable, tool calling gives the model controlled capabilities, and retries, rate limiting, caching, and provider fallback improve reliability.

vLLM exposes an OpenAI-compatible API for serving open-source models such as Llama or Mistral locally. Docker Compose can run the application services together, while model weights are mounted from a host cache.
