import { Pipeline } from './pipeline/pipeline';
import { visualizeEvents, visualizeMessages } from './shared/utils/visualization';
import { DenseCaptionAugmenter } from './stages/augmentation/dense-caption-augmenter';
import { StateTransitionAugmenter } from './stages/augmentation/state-transition-augmenter';
import { StructuredDataAugmenter } from './stages/augmentation/structured-data-augmenter';
import { EventExtractor } from './stages/extraction/event-extractor';
import { GuacExtractor } from './stages/extraction/guac-extractor';
import { VideoExtractor } from './stages/extraction/video-extractor';
import { GymDesktopExtractor } from './stages/extraction/simple-extractor';
import { MessageFormatter } from './stages/formatting/message-formatter';
import path from 'path';

import { Grader, type GraderLogger, type Chunk, type MetaData } from './stages/grading/grader';

// Optional: Create a custom logger for production use
class ProductionLogger implements GraderLogger {
  info(message: string, err?: Error, meta?: Record<string, unknown>): void {
    console.log(`[GRADER-INFO] ${message}`, meta ? JSON.stringify(meta) : '');
    if (err) console.log(`[GRADER-INFO-ERR] ${err.message}`);
  }
  warn(message: string, err?: Error, meta?: Record<string, unknown>): void {
    console.warn(`[GRADER-WARN] ${message}`, meta ? JSON.stringify(meta) : '');
    if (err) console.warn(`[GRADER-WARN-ERR] ${err.message}`);
  }
  error(message: string, err?: Error, meta?: Record<string, unknown>): void {
    console.error(`[GRADER-ERROR] ${message}`, err?.message || '', meta ? JSON.stringify(meta) : '');
  }
  debug(message: string, err?: Error, meta?: Record<string, unknown>): void {
    if (process.env.NODE_ENV === 'development') {
      console.debug(`[GRADER-DEBUG] ${message}`, meta ? JSON.stringify(meta) : '');
      if (err) console.debug(`[GRADER-DEBUG-ERR] ${err.message}`);
    }
  }
}
import { parseArgs } from 'util';

const { values } = parseArgs({
  args: Bun.argv,
  options: {
    data: {
      short: 'd',
      type: 'string'
    },
    out: {
      short: 'o',
      type: 'string'
    },
    sessions: {
      short: 's',
      type: 'string'
    },
    input: {
      short: 'i',
      type: 'string'
    },
    format: {
      short: 'f',
      type: 'string'
    },
    grade: {
      type: 'boolean',
      default: false
    },
    'chunk-size': {
      type: 'string',
      default: '4'
    },
    ffmpeg: {
      type: 'string',
      default: 'ffmpeg'
    },
    ffprobe: {
      type: 'string',
      default: 'ffprobe'
    }
  },
  strict: true,
  allowPositionals: true
});

// Convert an array of SFT messages into Grader chunks of size N
function sftToChunks(messages: any[], chunkSize: number): Chunk[] {
  const items = (messages ?? []).map((m: any) => {
    // Common cases: { role, content }, or strings
    if (typeof m === 'string') return { type: 'text', text: String(m) };
    if (m && typeof m.content === 'string') return { type: 'text', text: m.content };
    // Optional image support if present in SFT (base64 fields)
    if (m && m.type === 'image' && typeof m.data === 'string') {
      return { type: 'image', data: m.data, mime: m.mime ?? 'image/jpeg', cropInfo: m.cropInfo };
    }
    // Fallback: stringify unknown message shape (trim to avoid huge prompts)
    return { type: 'text', text: JSON.stringify(m).slice(0, 1000) };
  });
  const chunks: Chunk[] = [];
  for (let i = 0; i < items.length; i += chunkSize) {
    chunks.push(items.slice(i, i + chunkSize) as Chunk);
  }
  return chunks;
}


// Check for OpenAI API key if grading
if (values.grade && !process.env.OPENAI_API_KEY) {
  console.error('Error: OPENAI_API_KEY environment variable is required for grading mode');
  process.exit(1);
}

// Handle both input formats
let dataDir: string;
let sessions: string[];
let outDir: string;
const format: string = values.format || 'web';

if (values.input) {
  // New format: -i directory
  const inputPath = path.resolve(values.input);
  dataDir = path.dirname(inputPath);
  sessions = [path.basename(inputPath)];
  outDir = dataDir;
} else {
  // Original format: -d data -s sessions -o output
  dataDir = values.data || '.';
  sessions = values.sessions?.split(',') || [];
  outDir = values.out || '.';
}

// Initialize pipeline for both modes
const pipeline = new Pipeline({
  dataDir: dataDir,
  outputDir: outDir,
  sessionIds: sessions,
  extractors: [
    new VideoExtractor(dataDir, values.ffmpeg, values.ffprobe),
    ...(format === 'desktop'
      ? [new GymDesktopExtractor(dataDir)]
      : [new GuacExtractor(dataDir), new EventExtractor(dataDir)])
  ],
  augmenters:
    format === 'desktop'
      ? []
      : [
        new DenseCaptionAugmenter(1),
        new StateTransitionAugmenter(1),
        new StructuredDataAugmenter(1)
      ]
});

console.log(`Starting processing of ${sessions.length} sessions...`);

// Function to process a single session
async function processSession(
  session: string,
  grader: Grader,
  pipeline: Pipeline,
  dataDir: string,
  outDir: string,
  format: string,
  chunkSize: number
): Promise<void> {
  console.log(`\nProcessing session: ${session}`);
  const sftPath = path.join(dataDir, session, 'sft.json');
  const metaPath = path.join(dataDir, session, 'meta.json');

  // Check if sft.json exists
  try {
    await Bun.file(sftPath).json();

    // Grade existing sft.json
    console.log('Found existing sft.json, grading...');
    const sftMessages = await Bun.file(sftPath).json();
    let metaJson: any = {};
    try { metaJson = await Bun.file(metaPath).json(); } catch { }
    const meta: MetaData = {
      sessionId: session,
      platform: (format === 'desktop' ? 'desktop' : 'web'),
      taskDescription: metaJson?.quest?.title ?? metaJson?.title ?? metaJson?.description ?? undefined
    };
    const chunks = sftToChunks(sftMessages, chunkSize);
    const result = await grader.evaluateSession(chunks, meta);
    if (result) {
      console.log('\nGrading complete!');
      console.log(`Score: ${result.score}/100 (Confidence: ${(result.confidence * 100).toFixed(1)}%)`);
      console.log('\nScore Breakdown:');
      console.log(`- Outcome Achievement: ${result.outcomeAchievement}/100`);
      console.log(`- Process Quality: ${result.processQuality}/100`);
      console.log(`- Efficiency: ${result.efficiency}/100`);
      console.log('\nSummary:');
      console.log(result.summary);
      console.log('\nObservations:');
      console.log(result.observations);
      console.log('\nReasoning:');
      console.log(result.reasoning);

      // Write scores to file
      await Bun.write(path.join(outDir, session, 'scores.json'), JSON.stringify(result, null, 2));
    } else {
      console.error('Failed to grade session');
    }
  } catch (error) {
    // Run normal pipeline if sft.json doesn't exist
    console.log('No sft.json found, running pipeline...');
    const results = await pipeline.process(session);
    const html = visualizeEvents(results);
    await Bun.write(path.join(outDir, session, `results.html`), html);
    await Bun.write(path.join(outDir, session, `results.json`), JSON.stringify(results, null, 2));

    // Format messages
    const formatter = new MessageFormatter();
    const messages = await formatter.process(results);

    // Write formatted messages
    const msg_html = visualizeMessages(messages);
    await Bun.write(path.join(outDir, session, `sft.html`), msg_html);
    await Bun.write(path.join(outDir, session, `sft.json`), JSON.stringify(messages, null, 2));

    // Now grade the newly created sft.json
    console.log('\nGrading new sft.json...');
    const sftMessages2 = await Bun.file(sftPath).json();
    let metaJson2: any = {};
    try { metaJson2 = await Bun.file(metaPath).json(); } catch { }
    const meta2: MetaData = {
      sessionId: session,
      platform: (format === 'desktop' ? 'desktop' : 'web'),
      taskDescription: metaJson2?.quest?.title ?? metaJson2?.title ?? metaJson2?.description ?? undefined
    };
    const chunks2 = sftToChunks(sftMessages2, chunkSize);
    const result = await grader.evaluateSession(chunks2, meta2);
    if (result) {
      console.log('\nGrading complete!');
      console.log(`Score: ${result.score}/100 (Confidence: ${(result.confidence * 100).toFixed(1)}%)`);
      console.log('\nScore Breakdown:');
      console.log(`- Outcome Achievement: ${result.outcomeAchievement}/100`);
      console.log(`- Process Quality: ${result.processQuality}/100`);
      console.log(`- Efficiency: ${result.efficiency}/100`);
      console.log('\nSummary:');
      console.log(result.summary);
      console.log('\nObservations:');
      console.log(result.observations);
      console.log('\nReasoning:');
      console.log(result.reasoning);

      // Write scores to file
      await Bun.write(path.join(outDir, session, 'scores.json'), JSON.stringify(result, null, 2));
    } else {
      console.error('Failed to grade session');
    }
  }
}

if (values.grade) {
  // Grading mode with parallelization
  const productionLogger = new ProductionLogger();
  const parsed = Number(values['chunk-size']);
  const safeChunk = Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
  const grader = new Grader({
    apiKey: process.env.OPENAI_API_KEY!,
    chunkSize: safeChunk,
    timeout: 60_000,
    maxRetries: 3,
    seed: 42,
    rateLimiter: {
      maxTokens: 10,
      refillRate: 2
    }
  }, productionLogger);

  console.log(`Starting parallel processing of ${sessions.length} sessions...`);

  // Process sessions in parallel with rate limiting handled by the grader
  const sessionPromises = sessions.map(session =>
    processSession(session, grader, pipeline, dataDir, outDir, format, safeChunk || 4)
      .catch(error => {
        console.error(`Error processing session ${session}:`, error.message);
        return null; // Don't fail the entire batch
      })
  );

  const results = await Promise.allSettled(sessionPromises);

  // Log rate limiter stats
  const stats = grader.getRateLimiterStats();
  console.log(`\nRate limiter stats: ${stats.tokens} tokens remaining, ${stats.queueLength} requests queued`);

  // Report results
  const successful = results.filter(r => r.status === 'fulfilled').length;
  const failed = results.filter(r => r.status === 'rejected').length;
  console.log(`\nCompleted: ${successful} successful, ${failed} failed out of ${sessions.length} sessions`);

} else {
  // Normal pipeline mode (non-grading, sequential)
  for (const session of sessions) {
    const results = await pipeline.process(session);
    const html = visualizeEvents(results);
    await Bun.write(path.join(outDir, session, `results.html`), html);
    await Bun.write(path.join(outDir, session, `results.json`), JSON.stringify(results, null, 2));

    // Then format them into messages
    const formatter = new MessageFormatter();
    const messages = await formatter.process(results);

    // Write formatted messages visualization
    const msg_html = visualizeMessages(messages);
    await Bun.write(path.join(outDir, session, `sft.html`), msg_html);
    await Bun.write(path.join(outDir, session, `sft.json`), JSON.stringify(messages, null, 2));
  }
}

console.log(`Wrote sessions to ${outDir}`);
