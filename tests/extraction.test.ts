/**
 * @file End-to-end tests for the data extraction and processing pipeline.
 *
 * This file contains a series of tests that validate the entire data processing workflow,
 * from raw data extraction to final message formatting. It ensures that different data sources
 * (video, guacamole logs, session events) are correctly processed, merged, augmented, and
 * formatted for use in downstream tasks.
 *
 * The tests are organized into three main suites:
 * 1. Extraction Pipeline: Verifies that each individual extractor (Video, Guac, Event)
 *    can process its corresponding data source and produce a valid stream of events.
 * 2. Full Pipeline Integration: Tests the complete pipeline's ability to merge events
 *    from all extractors, apply augmentations (dense captions, state transitions, etc.),
 *    and produce a single, chronologically consistent timeline.
 * 3. Message Formatting: Ensures that the final, processed event stream can be correctly
 *    formatted into a sequence of messages suitable for model training, including
 *    validating the structure of user and assistant messages.
 *
 * Each test generates HTML visualization files (`*_test.html`) in the `data/` directory
 * to allow for manual inspection of the results.
 *
 * Note: Some tests are marked as `test.todo` and may be disabled.
 */
import { describe, expect, it, test } from 'bun:test';
import { VideoExtractor } from '../src/stages/extraction/video-extractor';
import { GuacExtractor } from '../src/stages/extraction/guac-extractor';
import { EventExtractor } from '../src/stages/extraction/event-extractor';
import { MessageFormatter } from '../src/stages/formatting/message-formatter';
import { visualizeEvents, visualizeMessages } from '../src/shared/utils/visualization';
import { DenseCaptionAugmenter } from '../src/stages/augmentation/dense-caption-augmenter';
import { StateTransitionAugmenter } from '../src/stages/augmentation/state-transition-augmenter';
import { StructuredDataAugmenter } from '../src/stages/augmentation/structured-data-augmenter';
import { Pipeline } from '../src/pipeline/pipeline';
import fs from 'node:fs/promises';
import path from 'path';
import { ProcessedEvent } from '../src/shared/types';
import { DemoDesktopExtractor } from '../src/stages/extraction/simple-extractor';

describe('Keyboard Layout Support', () => {
  test('should correctly handle AZERTY keyboard layout using actual_char', async () => {
    // Create a mock data directory
    const mockDataDir = path.join(process.cwd(), 'data', 'tests', 'extraction');

    // Create synthetic input_log.jsonl content for typing "HEADER" on AZERTY keyboard
    // On AZERTY: H=H, E=E, A=KeyQ, D=D, E=E, R=R
    const azertyEvents = [
      // H
      { event: 'keydown', data: { key: 'KeyH', actual_char: 'UnicodeInfo { name: Some("H"), unicode: [72], is_dead: false }' }, time: 1000 },
      { event: 'keyup', data: { key: 'KeyH', actual_char: '' }, time: 1050 },
      // E
      { event: 'keydown', data: { key: 'KeyE', actual_char: 'UnicodeInfo { name: Some("E"), unicode: [69], is_dead: false }' }, time: 1100 },
      { event: 'keyup', data: { key: 'KeyE', actual_char: '' }, time: 1150 },
      // A (KeyQ on AZERTY)
      { event: 'keydown', data: { key: 'KeyQ', actual_char: 'UnicodeInfo { name: Some("A"), unicode: [65], is_dead: false }' }, time: 1200 },
      { event: 'keyup', data: { key: 'KeyQ', actual_char: '' }, time: 1250 },
      // D
      { event: 'keydown', data: { key: 'KeyD', actual_char: 'UnicodeInfo { name: Some("D"), unicode: [68], is_dead: false }' }, time: 1300 },
      { event: 'keyup', data: { key: 'KeyD', actual_char: '' }, time: 1350 },
      // E
      { event: 'keydown', data: { key: 'KeyE', actual_char: 'UnicodeInfo { name: Some("E"), unicode: [69], is_dead: false }' }, time: 1400 },
      { event: 'keyup', data: { key: 'KeyE', actual_char: '' }, time: 1450 },
      // R
      { event: 'keydown', data: { key: 'KeyR', actual_char: 'UnicodeInfo { name: Some("R"), unicode: [82], is_dead: false }' }, time: 1500 },
      { event: 'keyup', data: { key: 'KeyR', actual_char: '' }, time: 1550 },
    ];

    // Create a temporary test directory and file
    const testDir = path.join(mockDataDir, 'azerty_test_' + Date.now());
    await fs.mkdir(testDir, { recursive: true });

    const jsonlPath = path.join(testDir, 'input_log.jsonl');
    const jsonlContent = azertyEvents.map(e => JSON.stringify(e)).join('\n');
    await fs.writeFile(jsonlPath, jsonlContent);

    // Process with the extractor
    const extractor = new DemoDesktopExtractor(mockDataDir);
    const testId = path.basename(testDir);
    const processedEvents = await extractor.process(testId);

    // Find the type event
    const typeEvents = processedEvents.filter(e => e.type === 'type');

    // Should have exactly one type event with "HEADER"
    expect(typeEvents.length).toBe(1);
    expect(typeEvents[0].data.text).toBe('HEADER');

    // The OLD behavior (using key only) would have produced "heqder"
    // This test verifies we now correctly use actual_char to produce "HEADER"

    console.log('✅ AZERTY keyboard test passed: typed "HEADER" correctly');
    console.log('   Keys pressed: KeyH, KeyE, KeyQ, KeyD, KeyE, KeyR');
    console.log('   Result:', typeEvents[0].data.text);

    // Cleanup
    await fs.rm(testDir, { recursive: true, force: true });
  });

  test('should fallback to key mapping when actual_char is not available', async () => {
    // Create a mock data directory
    const mockDataDir = path.join(process.cwd(), 'data', 'tests', 'extraction');

    // Create synthetic input_log.jsonl without actual_char (legacy format)
    const legacyEvents = [
      { event: 'keydown', data: { key: 'KeyH' }, time: 1000 },
      { event: 'keyup', data: { key: 'KeyH' }, time: 1050 },
      { event: 'keydown', data: { key: 'KeyE' }, time: 1100 },
      { event: 'keyup', data: { key: 'KeyE' }, time: 1150 },
    ];

    // Create a temporary test directory and file
    const testDir = path.join(mockDataDir, 'legacy_test_' + Date.now());
    await fs.mkdir(testDir, { recursive: true });

    const jsonlPath = path.join(testDir, 'input_log.jsonl');
    const jsonlContent = legacyEvents.map(e => JSON.stringify(e)).join('\n');
    await fs.writeFile(jsonlPath, jsonlContent);

    // Process with the extractor
    const extractor = new DemoDesktopExtractor(mockDataDir);
    const testId = path.basename(testDir);
    const processedEvents = await extractor.process(testId);

    // Find the type event
    const typeEvents = processedEvents.filter(e => e.type === 'type');

    // Should have exactly one type event with "he" (legacy fallback behavior)
    expect(typeEvents.length).toBe(1);
    expect(typeEvents[0].data.text).toBe('he');

    console.log('✅ Legacy fallback test passed: correctly fell back to key mapping');
    console.log('   Result:', typeEvents[0].data.text);

    // Cleanup
    await fs.rm(testDir, { recursive: true, force: true });
  });
});

describe('Extraction Pipeline', () => {
  const TEST_SESSION_ID = '6792a2a124f444f0e39ce887';
  const DATA_DIR = path.join(process.cwd(), 'data', 'tests', 'extraction');

  test(
    'should extract and visualize video frames',
    async () => {
      const extractor = new VideoExtractor(DATA_DIR);
      const events = await extractor.process(TEST_SESSION_ID);

      const html = visualizeEvents(events);
      await fs.writeFile(path.join(DATA_DIR, 'video_test.html'), html);

      expect(events.length).toBeGreaterThan(0);
      expect(events[0].type).toBe('frame');
    },
    { timeout: 60 * 1000 }
  );

  test('should extract and visualize guac events', async () => {
    const extractor = new GuacExtractor(DATA_DIR);
    const events = await extractor.process(TEST_SESSION_ID);

    const html = visualizeEvents(events);
    await fs.writeFile(path.join(DATA_DIR, 'guac_test.html'), html);

    // Check for mouse events
    expect(events.some((e) => e.type === 'mouseclick')).toBe(true);
    expect(events.some((e) => e.type === 'mousedrag')).toBe(true);

    // Check for keyboard events
    expect(events.some((e) => e.type === 'hotkey')).toBe(true);
    expect(events.some((e) => e.type === 'type')).toBe(true);
  });

  test('should extract and visualize session events', async () => {
    const extractor = new EventExtractor(DATA_DIR);
    const events = await extractor.process(TEST_SESSION_ID);

    const html = visualizeEvents(events);
    await fs.writeFile(path.join(DATA_DIR, 'events_test.html'), html);

    expect(events.some((e) => e.type === 'quest' || e.type === 'hint')).toBe(true);
  });
});

describe('Full Pipeline Integration with Augmentation', () => {
  const TEST_SESSION_ID = '6792a2a124f444f0e39ce887';
  const DATA_DIR = path.join(process.cwd(), 'data', 'tests', 'extraction');

  it(
    'should process and merge all events',
    async () => {
      const pipeline = new Pipeline({
        dataDir: DATA_DIR,
        outputDir: DATA_DIR,
        sessionIds: [TEST_SESSION_ID],
        extractors: [
          new VideoExtractor(DATA_DIR),
          new GuacExtractor(DATA_DIR),
          new EventExtractor(DATA_DIR)
        ],
        augmenters: [
          new DenseCaptionAugmenter(1),
          new StateTransitionAugmenter(1),
          new StructuredDataAugmenter(1)
        ]
      });

      const results = await pipeline.process(TEST_SESSION_ID);
      expect(results.length).toBeGreaterThan(0);

      const html = visualizeEvents(results);
      await fs.writeFile(path.join(DATA_DIR, 'pipeline_test.html'), html);

      // Verify timeline consistency
      const timestamps = results.map((e) => e.timestamp);
      expect(timestamps).toEqual([...timestamps].sort((a, b) => a - b));

      // Verify we have synthetic events
      expect(results.some((e: ProcessedEvent) => e.type === 'dense_caption')).toBe(true);
      expect(results.some((e: ProcessedEvent) => e.type === 'state_transition')).toBe(true);
      expect(results.some((e: ProcessedEvent) => e.type === 'structured_data')).toBe(true);
    },
    { timeout: 5 * 60000 }
  );
});

describe('Message Formatting', () => {
  const TEST_SESSION_ID = '6792a2a124f444f0e39ce887';
  const DATA_DIR = path.join(process.cwd(), 'data', 'tests', 'extraction');

  it(
    'should format events into messages',
    async () => {
      const pipeline = new Pipeline({
        dataDir: DATA_DIR,
        outputDir: DATA_DIR,
        sessionIds: [TEST_SESSION_ID],
        extractors: [
          new VideoExtractor(DATA_DIR),
          new GuacExtractor(DATA_DIR),
          new EventExtractor(DATA_DIR)
        ],
        augmenters: [
          new DenseCaptionAugmenter(1),
          new StateTransitionAugmenter(1),
          new StructuredDataAugmenter(1)
        ]
      });

      const events = await pipeline.process(TEST_SESSION_ID);
      expect(events.length).toBeGreaterThan(0);

      // Then format them into messages
      const formatter = new MessageFormatter();
      const messages = await formatter.process(events);

      // Write formatted messages visualization
      const html = visualizeMessages(messages);
      await fs.writeFile(path.join(DATA_DIR, 'messages_test.html'), html);

      // Verify message formatting
      for (const msg of messages) {
        expect(msg).toHaveProperty('role');
        expect(msg).toHaveProperty('content');
        expect(msg).toHaveProperty('timestamp');

        if (msg.role === 'user') {
          if (typeof msg.content === 'object') {
            expect(msg.content.type).toBe('image');
            expect(msg.content.data).toBeDefined();
          } else {
            // For hint events
            expect(typeof msg.content).toBe('string');
          }
        }
      }
    },
    { timeout: 5 * 60000 }
  );
});
