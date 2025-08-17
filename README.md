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
* **Deterministic scoring**: Final `score` is computed in code from component scores and configured weights.
* **Timeouts & retries**: Per-request timeout (default 60s) and exponential backoff with jitter.
* **Robust parsing**: Safe JSON extraction (fenced blocks or balanced braces) and clear error messages.
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
import { Grader, type Chunk, type MetaData } from "./src/stages/grading/grader";

const grader = new Grader({
  apiKey: process.env.OPENAI_API_KEY!,
  chunkSize: 4,
  model: "gpt-4o",
  timeout: 60_000,
  maxRetries: 3
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

const chunks = sftToChunks(messages, 4);
const result = await grader.evaluateSession(chunks, meta);
console.log(result.score, result.summary);
```

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

* **Logging hygiene**: Default logs avoid printing sensitive data and truncate large payloads.
* **Input guards**: Text length is truncated; images per chunk are capped; numeric options are normalized.
* **Failure handling**: Timeouts, retries with backoff, and explicit JSON parse errors make failure modes predictable.

---

## Troubleshooting

* **Missing `OPENAI_API_KEY`**: Set the environment variable for grading.
* **`ffmpeg` / `ffprobe` not found**: Install and ensure both are on PATH.
* **Timeouts**: Increase `timeout` or reduce chunk size.
* **Invalid model output**: The grader throws on bad JSON; check prompts and input size, then rerun.

---

If you want a short badge-style summary for the repo description, use:
“End-to-end Clones pipeline with structured LLM grading, deterministic scoring, and cost-aware multimodal support.”
