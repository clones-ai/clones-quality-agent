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

import { Grader, type GraderLogger } from './stages/grading/grader';

// Optional: Create a custom logger for production use
class ProductionLogger implements GraderLogger {
  info(message: string, meta?: Record<string, unknown>): void {
    console.log(`[GRADER-INFO] ${message}`, meta ? JSON.stringify(meta) : '');
  }

  error(message: string, error?: Error, meta?: Record<string, unknown>): void {
    console.error(`[GRADER-ERROR] ${message}`, error?.message || '', meta ? JSON.stringify(meta) : '');
  }

  debug(message: string, meta?: Record<string, unknown>): void {
    if (process.env.NODE_ENV === 'development') {
      console.debug(`[GRADER-DEBUG] ${message}`, meta ? JSON.stringify(meta) : '');
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
    format === 'desktop'
      ? new GymDesktopExtractor(dataDir)
      : (new GuacExtractor(dataDir), new EventExtractor(dataDir))
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

if (values.grade) {
  // Grading mode
  const productionLogger = new ProductionLogger();
  const grader = new Grader({
    apiKey: process.env.OPENAI_API_KEY!,
    chunkSize: parseInt(values['chunk-size']),
    timeout: 60 * 1000,
    maxRetries: 3
  }, productionLogger);

  for (const session of sessions) {
    console.log(`\nProcessing session: ${session}`);
    const sftPath = path.join(dataDir, session, 'sft.json');
    const metaPath = path.join(dataDir, session, 'meta.json');

    // Check if sft.json exists
    try {
      await Bun.file(sftPath).json();

      // Grade existing sft.json
      console.log('Found existing sft.json, grading...');
      const result = await grader.grade(metaPath, sftPath);
      if (result) {
        console.log('\nGrading complete!');
        console.log(`Score: ${result.score}`);
        console.log('\nSummary:');
        console.log(result.summary);
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
      const result = await grader.grade(metaPath, sftPath);
      if (result) {
        console.log('\nGrading complete!');
        console.log(`Score: ${result.score}`);
        console.log('\nSummary:');
        console.log(result.summary);
        console.log('\nReasoning:');
        console.log(result.reasoning);

        // Write scores to file
        await Bun.write(path.join(outDir, session, 'scores.json'), JSON.stringify(result, null, 2));
      } else {
        console.error('Failed to grade session');
      }
    }
  }
} else {
  // Normal pipeline mode
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
