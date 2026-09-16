# Serving and Reliability

## Providers

The assistant talks to any OpenAI-compatible chat completions endpoint. Three
provider modes exist. The `local` mode runs without an API key and without a
model; it answers from retrieved context and selects tools heuristically, which
makes the project demonstrable with no credentials. The `openai` mode calls a
hosted model. The `vllm` mode calls a self-hosted vLLM server.

vLLM exposes an OpenAI-compatible API for serving open-source models such as
Llama 3 or Mistral locally. It raises throughput using paged attention, which
stores the key-value cache in non-contiguous pages to cut memory fragmentation,
and continuous batching, which admits new requests into a running batch instead
of waiting for the whole batch to finish. Tool calling on vLLM requires the
server to be started with automatic tool choice and a tool-call parser matching
the model family.

## Tool calling

Tools are declared as OpenAI function schemas and the model chooses which to
call. The registry is an allow-list: a name the model invents is rejected,
arguments are parsed and validated before execution, and a failing tool returns
a JSON error back to the model so it can correct itself rather than taking the
request down. The calculator evaluates arithmetic by walking a restricted
abstract syntax tree, so expressions cannot reach imports, attributes, or
function calls. The loop is bounded by a maximum iteration count, and the final
turn is issued with tools disabled so the model must produce an answer.

## Reliability

Transient upstream failures are retried with exponential backoff and jitter.
Only retryable conditions are retried: timeouts, connection errors, and the
status codes 408, 409, 425, 429, 500, 502, 503 and 504. An authentication or
validation error fails immediately instead of burning the retry budget. If the
primary provider is still failing, a configured fallback provider serves the
request and the response reports which provider answered.

Requests are rate limited per client address using a sliding window, and a
throttled client receives a Retry-After header. Responses are cached in a
bounded TTL cache keyed on the question and the sampling parameters. Retrieval
failures degrade to a context-free answer rather than failing the request;
only a total provider failure returns 503.
