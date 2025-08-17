# Clones Data Pipeline

Process recordings from the Clones factory demos and (optionally) grade task completion with a modern, reliable LLM-based evaluator.

## Why teams use this

* **End-to-end pipeline**: From raw desktop/web captures to structured messages and a single `scores.json` you can trust.
* **Production-grade evaluator**: Strict JSON Schema outputs, deterministic scoring, and robust error handling reduce flaky runs.
* **Multimodal without the bill shock**: Vision supported with sensible defaults (low-detail images, text truncation, image caps).
* **Developer-friendly**: Clean TypeScript API, Bun-first CLI, and focused logs you can ship to prod.
* **Extensible scoring**: Tweak weights at runtime to emphasize outcomes, process, or efficiency—without changing your consumers.
* **Strong tests**: Fast unit tests with a mocked client and optional real-API integration tests.

## How it works (in 30 seconds)

1. **Extract**: Parse web/desktop sessions (events, guac, video).
2. **Augment**: Enrich with captions/state transitions/structured hints (web).
3. **Format**: Convert to SFT-style messages.
4. **Grade** (optional): Chunk the messages, run the LLM evaluator, and produce `scores.json`.
5. **Report**: Persist `results.json/html`, `sft.json/html`, and `scores.json`.

---

## Prerequisites

* [Bun](https://bun.sh) 1.2+
* `ffmpeg` and `ffprobe` on your PATH (for video extraction)
* For grading mode: `OPENAI_API_KEY`

## Input Data Formats

The pipeline accepts one of the following session layouts.

1. **Gym Desktop recordings**

* `session_id.mp4` – session video
* `session_id.events.jsonl` – event stream in JSONL
* `session_id.meta.json` – optional metadata

2. **Gym Web recordings**

* `session_id.events.json` – event data
* `session_id.guac` – Guacamole recording
* `session_id.guac.m4v` – session video

---

## Usage

You can run the pipeline in two ways.

### A) Original format (separate data/sessions/output)

```bash
bun run src/index.ts -o output_dir -f web -s session_id1,session_id2 -d data_directory
bun run src/index.ts -o output_dir -f desktop -s 20250211_215443 -d data
```

### B) Simplified format (from the session directory)

```bash
cd path/to/session_directory
bun run src/index.ts -f desktop -i .
```

### Arguments

* `-o, --out`           Output directory
* `-f, --format`        `web` or `desktop`
* `-s, --sessions`      Comma-separated list of session IDs
* `-d, --data`          Directory containing the input data
* `-i, --input`         Session directory (parent used as data & output)
* `--ffmpeg`            Path to `ffmpeg` (default: `ffmpeg`)
* `--ffprobe`           Path to `ffprobe` (default: `ffprobe`)
* `--grade`             Enable grading mode (requires `OPENAI_API_KEY`)
* `--chunk-size`        Messages per chunk when grading (default: 4)

---

## Grading mode

The grader evaluates a session using structured JSON outputs and a deterministic scoring model.

### Key properties

* **Structured outputs**: JSON Schema strict mode for guaranteed fields and types.
* **Multi-layer validation**: Uses SDK `message.parsed` when available, plus local Zod validation with business rules (2-6 observation lines).
* **Deterministic scoring**: Final `score` is computed in code from component scores and normalized weights, with single rounding.
* **Reproducible results**: Fixed seed (default 42), temperature=0, top_p=1, and zero penalties ensure identical outputs.
* **Smart error handling**: Typed errors (Timeout, Permanent, Transient) with intelligent retry logic that respects Retry-After headers.
* **Concurrent processing**: Sessions processed in parallel with built-in rate limiter (10 tokens, 2/sec refill by default).
* **Sequential chunks**: Within each session, chunks are processed sequentially to maintain summary dependencies.
* **Timeouts & retries**: Per-request timeout (default 60s) and exponential backoff with jitter.
* **Robust parsing**: Safe JSON extraction (fenced blocks or balanced braces) and clear error messages.
* **Observability**: Comprehensive metrics collection (tokens, timing, retries) with configurable hooks for monitoring.
* **No chain-of-thought**: Concise observations and a short final rationale only.
* **Cost-aware multimodal**: Low-detail images, text truncation, and image-per-chunk limits.

### Setup

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Run

```bash
# Grade existing sft.json in the current session directory
bun run src/index.ts -i . --grade

# Or process multiple sessions and then grade each
bun run src/index.ts -d data -s session1,session2 -o output --grade
```

### What grading produces

For each session, `scores.json`:

```json
{
  "summary": "One-paragraph outcome summary.",
  "observations": "• High-level bullets (2–6 lines)\n• No chain-of-thought",
  "reasoning": "Short final rationale.",
  "score": 73,
  "confidence": 0.90,
  "outcomeAchievement": 80,
  "processQuality": 70,
  "efficiency": 60
}
```

### CLI flow in grading mode

* If `sft.json` exists, it is read, chunked, and graded.
* If `sft.json` does not exist, the pipeline runs first to produce it, then grading proceeds.

---

## Programmatic use

```ts
import { 
  Grader, 
  type Chunk, 
  type MetaData,
  TimeoutError,
  PermanentError, 
  TransientError
} from "./src/stages/grading/grader";

const grader = new Grader({
  apiKey: process.env.OPENAI_API_KEY!,
  chunkSize: 4,
  model: "gpt-4o",
  timeout: 60_000,
  maxRetries: 3,
  seed: 42,  // Fixed seed for deterministic output
  rateLimiter: {
    maxTokens: 10,    // Token bucket size
    refillRate: 2     // Tokens per second refill rate
  },
  onMetrics: (metrics) => {
    // Custom metrics handler for observability
    console.log(`Request ${metrics.responseId}: ${metrics.usage?.totalTokens} tokens, ${metrics.timing.durationMs}ms`);
    if (metrics.outcome !== 'success') {
      console.warn(`Request failed: ${metrics.error?.message}`);
    }
  }
});

const meta: MetaData = {
  sessionId: "session_123",
  platform: "web",
  taskDescription: "User signs in and configures settings"
};

// Convert your SFT-style messages to chunks the grader expects
function sftToChunks(messages: any[], chunkSize = 4): Chunk[] {
  const items = (messages ?? []).map((m) => {
    if (typeof m === "string") return { type: "text", text: m };
    if (m && typeof m.content === "string") return { type: "text", text: m.content };
    if (m && m.type === "image" && typeof m.data === "string") {
      return { type: "image", data: m.data, mime: m.mime ?? "image/jpeg", cropInfo: m.cropInfo };
    }
    return { type: "text", text: JSON.stringify(m).slice(0, 1000) };
  });
  const chunks: Chunk[] = [];
  for (let i = 0; i < items.length; i += chunkSize) chunks.push(items.slice(i, i + chunkSize));
  return chunks;
}

try {
  const chunks = sftToChunks(messages, 4);
  const result = await grader.evaluateSession(chunks, meta);
  console.log(result.score, result.summary);
} catch (error) {
  if (error instanceof PermanentError) {
    console.error("Permanent error (will not retry):", error.message);
  } else if (error instanceof TimeoutError) {
    console.error("Request timed out:", error.message);
  } else if (error instanceof TransientError) {
    console.error("Transient error (retries exhausted):", error.message);
  }
}

// Process multiple sessions concurrently
const sessions = [
  { id: "session_1", messages: messages1 },
  { id: "session_2", messages: messages2 },
  { id: "session_3", messages: messages3 }
];

const promises = sessions.map(session => 
  grader.evaluateSession(
    sftToChunks(session.messages, 4),
    { sessionId: session.id, platform: "web" }
  )
);

const results = await Promise.allSettled(promises);
console.log(`Processed ${results.length} sessions concurrently`);

// Check rate limiter status
const stats = grader.getRateLimiterStats();
console.log(`Rate limiter: ${stats.tokens} tokens, ${stats.queueLength} queued`);
```

### Observability and metrics

The grader provides comprehensive metrics for monitoring and debugging:

```ts
import { type RequestMetrics } from "./src/stages/grading/grader";

const grader = new Grader({
  apiKey: process.env.OPENAI_API_KEY!,
  onMetrics: (metrics: RequestMetrics) => {
    // OpenAI response tracking
    console.log(`Response ID: ${metrics.responseId}`);
    console.log(`Model fingerprint: ${metrics.systemFingerprint}`);
    
    // Token usage
    if (metrics.usage) {
      console.log(`Tokens: ${metrics.usage.promptTokens} prompt + ${metrics.usage.completionTokens} completion = ${metrics.usage.totalTokens} total`);
    }
    
    // Timing and retry information
    console.log(`Duration: ${metrics.timing.durationMs}ms with ${metrics.timing.retryCount} retries`);
    if (metrics.timing.retryDelays.length > 0) {
      console.log(`Retry delays: ${metrics.timing.retryDelays.join(', ')}ms`);
    }
    
    // Request context
    console.log(`Session: ${metrics.context.sessionId}, Model: ${metrics.context.model}`);
    console.log(`Type: ${metrics.context.isFinal ? 'Final' : `Chunk ${metrics.context.chunkIndex}/${metrics.context.totalChunks}`}`);
    
    // Outcome and error handling
    console.log(`Outcome: ${metrics.outcome}`);
    if (metrics.error) {
      console.error(`Error: ${metrics.error.type} - ${metrics.error.message} (Status: ${metrics.error.statusCode})`);
    }
    
    // Send to your monitoring system
    // sendToDatadog(metrics);
    // sendToPrometheus(metrics);
  }
});
```

The metrics hook is called for every API request with detailed information about:
- **OpenAI metadata**: `response.id`, `system_fingerprint` for tracing
- **Token usage**: Prompt, completion, and total token counts
- **Timing**: Start/end times, duration, retry count and delays
- **Context**: Session ID, chunk information, model used
- **Outcome**: Success, permanent error, transient error, or timeout
- **Error details**: Type, message, and HTTP status code when applicable

### Adjusting evaluation weights

```ts
grader.updateEvaluationCriteria({
  outcomeAchievement: { weight: 0.6 },
  efficiency: { weight: 0.1 }
});

const criteria = grader.getEvaluationCriteria();
```

---

## Outputs from the pipeline (non-grading)

When not in `--grade` mode, the pipeline writes:

* `results.html` – session visualization
* `results.json` – processed session data
* `sft.html` – formatted messages preview
* `sft.json` – formatted messages used for grading

---

## Tests

### Unit tests (mocked)

```bash
bun test
# or
bun run test:grading
```

### Integration tests (real API)

Requires `OPENAI_API_KEY`. These validate structured outputs and end-to-end flow.

```bash
bun run test:grading:integration
```

---

## Security and reliability notes

* **Input sanitization**: All user text and `cropInfo` are sanitized to remove control characters and potential injection patterns.
* **Comprehensive logging security**: Enhanced redaction removes sensitive data from logs including:
  - **Email addresses**: `user@example.com` → `[EMAIL]`
  - **API keys**: OpenAI (`sk-...`), AWS (`AKIA...`), Google (`AIza...`), GitHub (`ghp_...`), Stripe (`sk_live_...`)
  - **OAuth tokens**: Bearer tokens, JWT tokens, Google OAuth (`ya29...`)
  - **Private keys**: PEM, SSH, OpenSSH private keys
  - **Database URLs**: Connection strings with credentials
  - **Base64 data**: Images and potential sensitive encoded content
  - **IP addresses**: Network information redaction
* **Input guards**: Text length is truncated; images per chunk are capped; numeric options are normalized.
* **Failure handling**: Timeouts, retries with backoff, and explicit JSON parse errors make failure modes predictable.

---

## Troubleshooting

* **Missing `OPENAI_API_KEY`**: Set the environment variable for grading.
* **`ffmpeg` / `ffprobe` not found**: Install and ensure both are on PATH.
* **Timeouts**: Increase `timeout` or reduce chunk size.
* **Invalid model output**: The grader throws on bad JSON; check prompts and input size, then rerun.

---

