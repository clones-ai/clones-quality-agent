# Clones Data Pipeline

Use the Clones data pipeline to gather information about data collected with the Clones factory's demos.

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

## 🚀 Advanced Grading Mode

The pipeline includes a state-of-the-art grading mode that evaluates task completion using modern LLM capabilities.

### **🔧 Robust Infrastructure**
- **60-second timeout** on OpenAI API calls (prevents infinite hanging)
- **Exponential backoff with jitter**: Smart retry delays (500ms → 1s → 2s → 4s → 8s max)
- **Automatic retry logic**: Up to 3 attempts with intelligent delay patterns
- **Structured logging** for better observability and debugging
- **Graceful error handling** for network issues and API failures

### **🧠 Intelligent Evaluation Framework**
- **Holistic Scoring**: Modern 3-component evaluation system
  - **Outcome Achievement (50%)**: Goal completion and objective fulfillment
  - **Process Quality (30%)**: Problem-solving approach, error recovery, adaptability
  - **Efficiency (20%)**: Time management, direct paths, resource utilization
- **Structured Outputs**: OpenAI JSON Schema strict mode for guaranteed response format
- **Deterministic Scoring**: Mathematically calculated scores from components (build-reproducible)
- **Confidence scoring**: 0.0-1.0 confidence levels for each evaluation
- **Chain-of-thought reasoning**: 6-step systematic evaluation process with brief observations

### **🎯 Advanced Features**
- **Dynamic evaluation criteria**: Configurable scoring weights for different contexts
- **Score transparency**: Raw model scores preserved alongside calculated deterministic scores
- **Enhanced context awareness**: Adaptive evaluation for different interaction types
- **Anti-hallucination safeguards**: Explicit instructions against inferring non-existent actions
- **Modern prompt engineering**: Optimized for GPT-4o/Claude-3.5 with 2025 best practices

### Setup

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

### **Grading Process**

The modernized grader will:

1. Look for an existing `sft.json` file in each session directory
2. If found, grade it directly using the advanced evaluation framework
3. If not found, run the normal pipeline first to generate `sft.json`, then grade it
4. Output a comprehensive `scores.json` file containing:
   - **Summary**: Bullet-point overview of all accomplished tasks
   - **Observations**: Brief, high-level bullet points (2-4 max) of key insights
   - **Score**: Deterministic score calculated from weighted components (0-100)
   - **Model Score Raw**: Original LLM score for diagnostic comparison
   - **Reasoning**: Clear justification based on the holistic evaluation framework
   - **Confidence**: AI confidence level (0.0-1.0) in the assessment
   - **Component Scores**: Individual scores for outcome, process quality, and efficiency

### **Enhanced Console Output**
```
Score: 78/100 (Confidence: 92.0%)
Model Raw Score: 75/100 (difference: 3)

Score Breakdown:
- Outcome Achievement: 75/100
- Process Quality: 85/100  
- Efficiency: 80/100

Summary:
• Successfully navigated to target application
• Completed 3 out of 4 primary objectives
• Demonstrated good error recovery when encountering obstacles

Observations:
• User demonstrated strong navigation skills and adaptability.
• Process was methodical with effective error recovery.
• Final objective incomplete but progress was substantial.

Reasoning:
High outcome achievement (75% of objectives completed) combined with excellent 
process quality and good efficiency. Minor deduction for incomplete final objective.
```

### **Configuration Options**

#### **Basic Configuration**
```bash
# Adjust chunk size (default 4) to control message processing
bun run src/index.ts -i . --grade --chunk-size 8

# Use specific model (default: gpt-4o)
GRADER_MODEL=gpt-4o bun run src/index.ts -i . --grade
```

#### **Advanced Configuration (Programmatic)**
```typescript
import { Grader } from './src/stages/grading/grader';

// Modern configuration API
const grader = new Grader({
  apiKey: process.env.OPENAI_API_KEY!,
  chunkSize: 4,
  model: 'gpt-4o',
  timeout: 60000,        // 60-second timeout
  maxRetries: 3          // Maximum retry attempts
}, logger);

// Customize evaluation criteria for different contexts
grader.updateEvaluationCriteria({
  outcomeAchievement: { weight: 0.6 }, // More focus on results
  processQuality: { weight: 0.25 },    // Standard process evaluation  
  efficiency: { weight: 0.15 }         // Less emphasis on speed
});

// Grade with custom criteria
const result = await grader.grade(metaData, messages);
```

## Running Tests

To run the test suite, use the following command:

```bash
bun test
```

### **🧪 Comprehensive Grading Tests**

The project includes extensive tests for the modernized grading system:

#### **Test Categories**
- **Unit Tests**: 27 tests with mocked API calls (fast execution)
- **Integration Tests**: Real OpenAI API validation (requires OPENAI_API_KEY)
- **Coverage**: 100% of new APIs and features tested
- **Feature Tests**: Structured Outputs, deterministic scoring, exponential backoff

#### **Running Tests**
```bash
# Unit tests (mocked, fast) - 27 tests, 107 assertions
bun run test:grading:unit

# Integration tests (real API, requires OPENAI_API_KEY)
bun run test:grading:integration

# All grading tests (unit + integration)
bun run test:grading:all

# Default: run unit tests only
bun run test:grading
```

## **🔧 Technical Details**

### **Structured Outputs Implementation**
- **JSON Schema Strict Mode**: Guaranteed response format with OpenAI's latest API
- **Type Safety**: Automatic validation of required fields and data types
- **Performance**: Reduced token usage by eliminating verbose JSON examples

### **Deterministic Scoring Algorithm**
```typescript
// Mathematically calculated from weighted components
const calculatedScore = Math.round(
  outcomeScore * 0.5 +      // 50% weight
  processScore * 0.3 +      // 30% weight  
  efficiencyScore * 0.2     // 20% weight
);

// Raw model score preserved for diagnostics
result.modelScoreRaw = originalLLMScore;
result.score = calculatedScore;  // Used for final scoring
```

### **Exponential Backoff Formula**
```typescript
const delay = Math.min(8000, 500 * 2 ** (retries - 1)) + Math.random() * 250;
// Retry 1: 500-750ms
// Retry 2: 1000-1250ms  
// Retry 3: 2000-2250ms
// Maximum: 8000ms + jitter
```

## **🚀 Future-Proof Design**

This modernized grading system is designed for **2025 LLM capabilities** and beyond:

- **GPT-5 Ready**: Optimized prompts and Structured Outputs for next-generation models
- **Scalable Architecture**: Configurable evaluation criteria for different contexts
- **Extensible Framework**: Easy to add new evaluation dimensions
- **Production Grade**: Battle-tested error handling and comprehensive logging
- **Build Reproducible**: Deterministic scoring ensures consistent results across deployments
