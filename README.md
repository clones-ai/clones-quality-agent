# Viralmind Data Pipeline

Use the Viralmind data pipeline to gather information about data collected with the Viralmind training gym.

## Input Data Format

The pipeline expects input data in one of these formats:

1. Gym Desktop recordings:
   - `session_id.mp4` - The video recording
   - `session_id.events.jsonl` - Event data in JSONL format
   - `session_id.meta.json` - Optional metadata about the recording

2. Gym Web recordings:
   - `session_id.events.json` - Event data
   - `session_id.guac` - Guacamole recording file
   - `session_id.guac.m4v` - Video recording

## Usage

You can run the pipeline in two ways:

1. Original format with separate data, sessions, and output directories:

```bash
bun run src/index.ts -o output_dir -f web -s session_id1,session_id2 -d data_directory
bun run src/index.ts -o output_dir -f desktop -i 20250211_215443 -d data
```

2. Simplified format using the current directory:

```bash
cd path/to/session_directory
bun run src/index.ts -f desktop -i .
```

### Arguments

- `-o, --output`: Directory to save processed output
- `-f, --format`: Format of the input data. Either `web` or `desktop`.
- `-s, --sessions`: Comma-separated list of session IDs to process
- `-d, --data`: Directory containing the input data files
- `-i, --input`: Session directory to process. The parent directory will be used as both data and output directory.
- `--ffmpeg`: Path to the ffmpeg binary. Defaults to `ffmpeg`.
- `--ffprobe`: Path to the ffprobe binary. Defaults to `ffprobe`.

Additional options:

- `--grade`: Enable grading mode to evaluate task completion (requires OPENAI_API_KEY)
- `--chunk-size`: Number of messages per chunk when grading (default: 4)

### Grading Mode

The pipeline includes a grading mode that evaluates task completion using GPT-4V. To use grading mode:

1. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

2. Run with the `--grade` flag:

```bash
# Grade existing sft.json files
bun run src/index.ts -i . --grade

# Or process multiple sessions
bun run src/index.ts -d data -s session1,session2 -o output --grade
```

The grader will:

1. Look for an existing sft.json file in each session directory
2. If found, grade it directly
3. If not found, run the normal pipeline first to generate sft.json, then grade it
4. Output a scores.json file containing:
   - A summary of completed tasks
   - A score from 0-100
   - Detailed reasoning for the score

You can adjust the chunk size (default 4) to control how many messages are processed at once:

```bash
bun run src/index.ts -i . --grade --chunk-size 8
```
