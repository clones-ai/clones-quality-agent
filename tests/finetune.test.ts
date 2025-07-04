/**
 * @file Test for the fine-tuning dataset generation and model training pipeline.
 *
 * This test file is responsible for creating a complete dataset for fine-tuning
 * an OpenAI model and simulating the fine-tuning process. It uses the `PaintPipeline`
 * to generate synthetic conversation data from doodles, formats it for training,
 * and provides a placeholder for launching the actual fine-tuning job.
 *
 * The test is divided into two main parts:
 * 1. Dataset Generation: Creates the `.jsonl` file required for fine-tuning.
 * 2. Fine-Tuning Simulation: A placeholder that logs instructions for manual
 *    fine-tuning and displays results from a pre-existing stats file.
 *
 * Workflow:
 *
 * Test 1: 'should generate fine-tuning dataset'
 * 1. Scans the `data/doodles` directory for `.ndjson` doodle files.
 * 2. Initializes the `PaintPipeline` to process a configurable number of trajectories.
 * 3. For each trajectory, it generates a sequence of events from a random selection of doodles.
 * 4. The events are formatted into a conversation using `MessageFormatter`.
 * 5. The conversation is converted to the OpenAI message format.
 * 6. The total token count is checked; if it exceeds the maximum, the trajectory is skipped.
 * 7. Valid trajectories are written to a `.jsonl` file (e.g., `finetune_1_dataset.jsonl`).
 * 8. A companion `_stats.json` file is generated with statistics about the dataset.
 *
 * Test 2: 'should finetune the painting model'
 * 1. Checks a `SKIP_FINETUNING` flag to determine whether to simulate the training process.
 * 2. If skipping, it loads pre-existing results from `finetuning_stats.json` and displays them.
 * 3. If not skipping, it provides instructions for the user to manually upload the `.jsonl`
 *    file to the OpenAI platform.
 * 4. It calculates and logs the estimated training cost based on the number of tokens.
 */
import { describe, test, it, expect } from 'bun:test';
import fs from 'node:fs';
import path from 'path';
import { PaintPipeline } from '../src/pipeline/paint-pipeline';
import { MessageFormatter } from '../src/stages/formatting/message-formatter';
import { OpenAIUtils } from '../src/shared/utils/openai';

// this uses the cached fine-tuning job
// turn this on when recording a demo so you don't have to wait 1 hour
const SKIP_FINETUNING = true;

describe('PaintPipeline - Fine-tuning Dataset Generation', () => {
  console.log('--- Starting Fine-tuning Test ---');
  const NUM_TRAJECTORIES = 1; // Number of trajectories to generate
  const DATA_DIR = path.join(process.cwd(), 'data', 'tests', 'finetune');
  const METADATA_PATH = path.join(DATA_DIR, 'jspaint_0.json');
  const OUTPUT_DIR = path.join(DATA_DIR, 'finetune');
  console.log(`Using data from: ${DATA_DIR}`);
  console.log(`Outputting to: ${OUTPUT_DIR}`);

  // Create output directory if it doesn't exist
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    console.log('Created output directory.');
  }

  it(
    'should generate fine-tuning dataset',
    async () => {
      console.log('\n--- Generating Fine-tuning Dataset ---');
      // Get list of available doodle files
      const doodleDir = path.join(DATA_DIR, 'doodles');
      const doodleFiles = fs
        .readdirSync(doodleDir)
        .filter((f) => f.endsWith('.ndjson'))
        .map((f) => f.replace('.ndjson', ''));
      console.log(`Found ${doodleFiles.length} doodle files.`);

      // Initialize pipeline and formatter
      const pipeline = new PaintPipeline(DATA_DIR, METADATA_PATH);
      const formatter = new MessageFormatter();
      console.log('Pipeline and formatter initialized.');

      // Open write stream for JSONL output
      const outputPath = path.join(OUTPUT_DIR, `finetune_${NUM_TRAJECTORIES}_dataset.jsonl`);
      const writeStream = fs.createWriteStream(outputPath);
      console.log(`Writing dataset to: ${outputPath}`);

      // Stats tracking
      let totalMessages = 0;
      let imageMessages = 0;
      let textMessages = 0;
      let validTrajectories = 0;
      let skippedTrajectories = 0;

      console.log(`Generating ${NUM_TRAJECTORIES} trajectories...`);
      // Generate trajectories
      for (let i = 0; i < NUM_TRAJECTORIES; i++) {
        console.log(`\nProcessing trajectory ${i + 1}/${NUM_TRAJECTORIES}...`);
        // Generate events for 5 random doodles
        console.log('Generating synthetic events...');
        const events = await pipeline.process(doodleFiles, 2);
        console.log(`Generated ${events.length} events.`);

        // Format into messages
        console.log('Formatting events into messages...');
        const messages = await formatter.process(events);
        console.log(`Formatted into ${messages.length} messages.`);

        // Validate messages
        for (const msg of messages) {
          expect(msg).toHaveProperty('role');
          expect(msg).toHaveProperty('content');
          expect(msg).toHaveProperty('timestamp');
        }

        // Convert to OpenAI format
        console.log('Converting to OpenAI format...');
        const openaiMessages = OpenAIUtils.convertToOpenAIFormat(messages);

        // Check token count
        const tokenCount = OpenAIUtils.countConversationTokens(openaiMessages);
        if (tokenCount > OpenAIUtils.MAX_TOKENS) {
          console.log(`Skipping trajectory ${i + 1} - exceeds token limit (${tokenCount} tokens)`);
          skippedTrajectories++;
          continue;
        }

        // Update stats
        totalMessages += messages.length;
        imageMessages += messages.filter(
          (m) => typeof m.content === 'object' && m.content.type === 'image'
        ).length;
        textMessages += messages.filter((m) => typeof m.content === 'string').length;
        validTrajectories++;

        // Write to file
        console.log('Writing trajectory to file...');
        writeStream.write(JSON.stringify({ messages: openaiMessages }) + '\n');

        // Log progress
        console.log(
          `Generated trajectory ${i + 1}/${NUM_TRAJECTORIES} (${validTrajectories} valid, ${skippedTrajectories} skipped)`
        );
      }

      // Close write stream
      writeStream.end();

      // Generate stats
      const stats = {
        totalTrajectories: validTrajectories,
        skippedTrajectories,
        totalMessages,
        avgMessagesPerTrajectory: totalMessages / validTrajectories,
        imageMessages,
        textMessages
      };

      // Save stats
      console.log('\nGenerating and saving stats...');
      const statsPath = path.join(OUTPUT_DIR, `dataset_${NUM_TRAJECTORIES}_stats.json`);
      fs.writeFileSync(statsPath, JSON.stringify(stats, null, 2));

      console.log('Dataset generation complete!');
      console.log('Stats:', stats);
      console.log(`Dataset saved to: ${outputPath}`);
      console.log(`Stats saved to: ${statsPath}`);
    },
    { timeout: 20 * 60 * 1000 }
  );

  it(
    'should finetune the painting model',
    async () => {
      console.log('\n--- Simulating Fine-tuning ---');
      if (SKIP_FINETUNING) {
        console.log('fine-tuning skipped: model already exists!');
      } else {
        console.log('fine-tuning via openai lib not yet implemented');
        console.log('please upload the .jsonl to https://platform.openai.com/finetune/');
        return;
      }

      // Load and log finetuning stats
      console.log('Loading fine-tuning stats...');
      const statsPath = path.join(DATA_DIR, 'finetuning_stats.json');
      const stats = JSON.parse(fs.readFileSync(statsPath, 'utf-8'));

      console.log('\nFinetuning Stats:');
      console.log('------------------');
      console.log(`Job ID: ${stats.jobId}`);
      console.log(`Model: ${stats.outputModel}`);
      console.log(`Trained Tokens: ${stats.trainedTokens.toLocaleString()}`);
      console.log(`Epochs: ${stats.epochs}`);
      console.log(`Batch Size: ${stats.batchSize}`);
      console.log(`Learning Rate Multiplier: ${stats.lrMultiplier}`);
      console.log(`Training Method: ${stats.trainingMethod}`);
      console.log(`Final Loss: ${stats.finalLoss}`);
      console.log(`Time Taken: ${stats.timeTaken}`);

      console.log('\nYour model has been trained!\n');

      // Calculate cost
      const costUSD = (stats.trainedTokens / 1_000_000) * 25;
      const costSOL = (costUSD * 0.16) / 25; // $25 = 0.16 SOL conversion
      console.log(`Cost: ${costSOL.toFixed(4)} SOL`);
    },
    { timeout: 120 * 60 * 1000 }
  );
});
