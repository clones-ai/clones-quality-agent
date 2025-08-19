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

import { Grader } from './stages/grading/grader';
import { GraderLogger, Chunk, MetaData } from './stages/grading/grader/types';

import packageJson from '../package.json';

console.log(`Pipeline version: ${packageJson.version}`);

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

async function gradeSftFile(
  grader: Grader,
  session: string,
  sftPath: string,
  metaPath: string,
  outDir: string,
  chunkSize: number,
  format: string
): Promise<void> {
  console.log(`Grading sft.json for session: ${session}`);
  const sftMessages = await Bun.file(sftPath).json();
  let metaJson: any = {};
  try {
    metaJson = await Bun.file(metaPath).json();
  } catch { }
  const meta: MetaData = {
    sessionId: session,
    platform: format === 'desktop' ? 'desktop' : 'web',
    taskDescription:
      metaJson?.quest?.title ?? metaJson?.title ?? metaJson?.description ?? undefined
  };
  const chunks = sftToChunks(sftMessages, chunkSize);
  const result = await grader.evaluateSession(chunks, meta);
  if (result) {
    console.log('\nGrading complete!');
    console.log(`Score: ${result.score}/100 (Confidence: ${(result.confidence).toFixed(1)}%)`);
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

    await Bun.write(path.join(outDir, session, 'scores.json'), JSON.stringify(result, null, 2));
  } else {
    console.error('Failed to grade session');
  }
}

async function runPipelineAndFormat(
  pipeline: Pipeline,
  session: string,
  outDir: string
): Promise<void> {
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
}

async function processSession(
  session: string,
  pipeline: Pipeline,
  dataDir: string,
  outDir: string,
  format: string,
  chunkSize: number,
  grader?: Grader
): Promise<void> {
  console.log(`\nProcessing session: ${session}`);
  const sftPath = path.join(dataDir, session, 'sft.json');
  const metaPath = path.join(dataDir, session, 'meta.json');
  const sftExists = await Bun.file(sftPath).exists();

  if (!sftExists) {
    console.log('No sft.json found, running pipeline...');
    await runPipelineAndFormat(pipeline, session, outDir);
  } else {
    console.log('Found existing sft.json.');
  }

  if (grader) {
    await gradeSftFile(grader, session, sftPath, metaPath, outDir, chunkSize, format);
  }
}

async function processAllSessions() {
  const chunkSize = Number.isFinite(Number(values['chunk-size']))
    ? Number(values['chunk-size'])
    : 4;

  if (values.grade) {
    const productionLogger = new ProductionLogger();
    const allMetrics: any[] = [];
    const grader = new Grader(
      {
        apiKey: process.env.OPENAI_API_KEY!,
        chunkSize,
        timeout: 60_000,
        maxRetries: 3,
        seed: 42,
        rateLimiter: { maxTokens: 10, refillRate: 2 },
        onMetrics: metrics => {
          allMetrics.push(metrics);
        }
      },
      productionLogger
    );

    console.log(`Starting parallel processing of ${sessions.length} sessions...`);
    const sessionPromises = sessions.map(session =>
      processSession(session, pipeline, dataDir, outDir, format, chunkSize, grader).catch(
        error => {
          console.error(`Error processing session ${session}:`, error.message);
          return null;
        }
      )
    );
    await Promise.allSettled(sessionPromises);

    const stats = grader.getRateLimiterStats();
    console.log(
      `\nRate limiter stats: ${stats.tokens} tokens remaining, ${stats.queueLength} requests queued`
    );

    const results = await Promise.allSettled(sessionPromises);
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = sessions.length - successful;
    console.log(
      `\nCompleted: ${successful} successful, ${failed} failed out of ${sessions.length} sessions`
    );

    // Generate metrics summary for each session
    const sessionMetrics = sessions.map(sessionId => {
      const sessionData = allMetrics.filter(m => m.context.sessionId === sessionId);

      if (sessionData.length === 0) {
        return {
          sessionId,
          status: 'failed',
          error: 'No metrics collected (processing failed)',
          totalRequests: 0,
          totalTokens: 0,
          totalDuration: 0,
          averageRetries: 0
        };
      }

      const totalTokens = sessionData.reduce((sum, m) => sum + (m.usage?.totalTokens || 0), 0);
      const totalDuration = sessionData.reduce((sum, m) => sum + m.timing.durationMs, 0);
      const totalRetries = sessionData.reduce((sum, m) => sum + m.timing.retryCount, 0);
      const successfulRequests = sessionData.filter(m => m.outcome === 'success').length;

      return {
        sessionId,
        status: successfulRequests === sessionData.length ? 'success' : 'partial_failure',
        totalRequests: sessionData.length,
        successfulRequests,
        failedRequests: sessionData.length - successfulRequests,
        totalTokens,
        totalDuration,
        averageRetries: sessionData.length > 0 ? (totalRetries / sessionData.length) : 0,
        details: sessionData.map(m => ({
          responseId: m.responseId,
          type: m.context.isFinal ? 'final' : 'chunk',
          chunkIndex: m.context.chunkIndex,
          outcome: m.outcome,
          tokens: m.usage?.totalTokens || 0,
          duration: m.timing.durationMs,
          retries: m.timing.retryCount,
          error: m.error?.message
        }))
      };
    });

    // Write metrics.json for each session
    for (const sessionMetric of sessionMetrics) {
      const metricsPath = path.join(outDir, sessionMetric.sessionId, 'metrics.json');
      await Bun.write(metricsPath, JSON.stringify(sessionMetric, null, 2));
      console.log(`📊 Metrics written to: ${metricsPath}`);
    }

    // Write global metrics summary
    const globalMetrics = {
      timestamp: new Date().toISOString(),
      pipeline: {
        version: "2.0.0",
        mode: "grading",
        sessions: sessions.length,
        successful: successful,
        failed: failed
      },
      totals: {
        requests: allMetrics.length,
        successfulRequests: allMetrics.filter(m => m.outcome === 'success').length,
        tokens: allMetrics.reduce((sum, m) => sum + (m.usage?.totalTokens || 0), 0),
        duration: allMetrics.reduce((sum, m) => sum + m.timing.durationMs, 0),
        retries: allMetrics.reduce((sum, m) => sum + m.timing.retryCount, 0)
      },
      rateLimiter: stats,
      sessions: sessionMetrics
    };

    const globalMetricsPath = path.join(outDir, 'metrics.json');
    await Bun.write(globalMetricsPath, JSON.stringify(globalMetrics, null, 2));
    console.log(`📊 Global metrics written to: ${globalMetricsPath}`);
  } else {
    console.log(`Starting sequential processing of ${sessions.length} sessions...`);
    for (const session of sessions) {
      await processSession(session, pipeline, dataDir, outDir, format, chunkSize);
    }
  }
}

processAllSessions().then(() => {
  console.log(`Wrote sessions to ${outDir}`);
});
