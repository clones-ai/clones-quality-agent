/**
 * @file Test for the synthetic data generation pipeline using JSPaint doodles.
 *
 * This test validates the `PaintPipeline`'s ability to generate a realistic,
 * chronologically-ordered stream of browser events from a set of `.ndjson` doodle files.
 * It also verifies the subsequent formatting of these events into a structured message format.
 *
 * Workflow:
 * 1. Scans a directory for available doodle data files (e.g., `whale.ndjson`).
 * 2. Initializes the `PaintPipeline` with metadata.
 * 3. Processes a random sample of doodles to generate a flat list of synthetic events.
 * 4. Performs validation on the generated events:
 *    - Ensures that events were created.
 *    - Verifies that critical event types (`quest`, `frame`, `mousedrag`, `reasoning`) are present.
 *    - Confirms that the event timestamps are monotonically increasing.
 * 5. Formats the raw events into structured messages using the `MessageFormatter`.
 * 6. Verifies that the final messages have the correct structure (`role`, `content`, `timestamp`).
 * 7. Generates two HTML files for manual review:
 *    - `synthetic_events.html`: A visualization of the raw event stream.
 *    - `synthetic_messages.html`: A visualization of the final formatted messages.
 */

import { test, expect } from 'bun:test';
import fs from 'node:fs';
import path from 'path';
import { PaintPipeline } from '../src/pipeline/paint-pipeline';
import { MessageFormatter } from '../src/stages/formatting/message-formatter';
import { visualizeEvents, visualizeMessages } from '../src/shared/utils/visualization';

const TEST_DATA_DIR = path.join(process.cwd(), 'data', 'tests', 'synthetic');
const METADATA_PATH = path.join(TEST_DATA_DIR, 'jspaint_0.json');

test(
  'Synthetic Paint Pipeline should generate and format events',
  async () => {
    console.log(`--- Starting Synthetic Paint Pipeline Test ---`);
    console.log(`Using test data from: ${TEST_DATA_DIR}`);

    // Get list of available doodle files from the test directory
    const doodleFiles = fs
      .readdirSync(path.join(TEST_DATA_DIR, 'doodles'))
      .filter((f) => f.endsWith('.ndjson') && f.startsWith('whale'))
      .map((f) => f.replace('.ndjson', ''));

    console.log(`Found ${doodleFiles.length} doodle file(s) to process.`);
    expect(doodleFiles.length).toBeGreaterThan(0);

    // Initialize pipeline
    console.log('Initializing PaintPipeline...');
    const pipeline = new PaintPipeline(TEST_DATA_DIR, METADATA_PATH);
    console.log('Pipeline initialized.');

    // Generate events for a single doodle to ensure determinism
    console.log('Generating synthetic events...');
    const events = await pipeline.process(doodleFiles, 1);
    console.log(`Generated ${events.length} events.`);

    // Verify we have events
    expect(events.length).toBeGreaterThan(0);

    // Check we have all expected event types and their data is valid
    console.log('Verifying event types and their content...');
    const questEvent = events.find((e) => e.type === 'quest');
    expect(questEvent).toBeDefined();
    expect(questEvent?.data.message).toBeTypeOf('string');
    expect(questEvent?.data.message?.length).toBeGreaterThan(0);

    const frameEvent = events.find((e) => e.type === 'frame');
    expect(frameEvent).toBeDefined();
    expect(frameEvent?.data.frame).toBeTypeOf('string');
    expect(frameEvent?.data.frame?.length).toBeGreaterThan(0);

    const mousedragEvent = events.find((e) => e.type === 'mousedrag');
    expect(mousedragEvent).toBeDefined();
    expect(mousedragEvent?.data.coordinates).toBeArray();
    expect(mousedragEvent?.data.coordinates?.length).toBeGreaterThan(0);

    const reasoningEvent = events.find((e) => e.type === 'reasoning');
    expect(reasoningEvent).toBeDefined();
    expect(reasoningEvent?.data.text).toBeTypeOf('string');
    expect(reasoningEvent?.data.text?.length).toBeGreaterThan(0);

    // Verify timeline consistency
    const timestamps = events.map((e) => e.timestamp);
    expect(timestamps).toEqual([...timestamps].sort((a, b) => a - b));
    console.log('Event verification successful.');

    // Generate visualization
    const eventsHtmlPath = path.join(TEST_DATA_DIR, 'synthetic_events.html');
    console.log(`Generating events visualization at: ${eventsHtmlPath}`);
    const html = visualizeEvents(events);
    fs.writeFileSync(eventsHtmlPath, html);

    // Format into messages
    console.log('Formatting events into messages...');
    const formatter = new MessageFormatter();
    const messages = await formatter.process(events);
    console.log(`Formatted into ${messages.length} messages.`);

    // Generate message visualization
    const messagesHtmlPath = path.join(TEST_DATA_DIR, 'synthetic_messages.html');
    console.log(`Generating messages visualization at: ${messagesHtmlPath}`);
    const msgHtml = visualizeMessages(messages);
    fs.writeFileSync(messagesHtmlPath, msgHtml);

    // Verify message formatting
    console.log('Verifying message format...');
    for (const msg of messages) {
      expect(msg).toHaveProperty('role');
      expect(['user', 'assistant'].includes(msg.role)).toBe(true);
      expect(msg).toHaveProperty('content');
      if (typeof msg.content === 'string') {
        expect(msg.content.length).toBeGreaterThan(0);
      } else {
        expect(msg.content).toHaveProperty('type', 'image');
        expect(typeof msg.content.data).toBe('string');
        expect(msg.content.data.length).toBeGreaterThan(0);
      }
      expect(msg).toHaveProperty('timestamp');
    }
    console.log('Message format verification successful.');
    console.log('--- Synthetic Paint Pipeline Test Finished Successfully ---');
  },
  { timeout: 60 * 1000 }
);
