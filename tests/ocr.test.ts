/**
 * @file Non-regression test for the OCR (Optical Character Recognition) functionality.
 *
 * This test aims to ensure that the performance and accuracy of text extraction
 * from an image remain consistent across code changes.
 *
 * Workflow:
 * 1. Loads a predefined test image (`ocr_frame_175000.png`).
 * 2. Loads a ground truth JSON file (`ocr_words.json`) containing the expected
 *    results for this image (full text, list of words, and confidence scores).
 * 3. Runs the Tesseract.js OCR engine on the test image.
 * 4. Compares the results obtained from the OCR with the data from the ground truth file.
 *
 * The test passes if the recognized text and individual words match the reference file,
 * thus ensuring that the OCR functionality does not regress.
 */

import { test, expect } from 'bun:test';
import { createWorker } from 'tesseract.js';
import path from 'path';
import fs from 'node:fs';


const TEST_IMAGE = path.join(process.cwd(), 'data', 'tests', 'ocr', 'ocr_frame_175000.png');
const GROUND_TRUTH_PATH = path.join(process.cwd(), 'data', 'tests', 'ocr', 'ocr_words.json');

test(
  'OCR should extract words with bounding boxes',
  async () => {
    // 1. Load ground truth data
    const groundTruth = JSON.parse(fs.readFileSync(GROUND_TRUTH_PATH, 'utf-8'));

    try {
      console.log('Creating Tesseract worker...');
      const worker = await createWorker('eng');
      console.log('Worker created.');

      console.log(`Recognizing text from: ${TEST_IMAGE}`);
      const ret = await worker.recognize(TEST_IMAGE, {}, { blocks: true });
      console.log('Recognition finished.');

      expect(ret.data).not.toBeNull();
      expect(ret.data.blocks).not.toBeNull();
      if (!ret.data || !ret.data.blocks) return; // Type guard

      // 2. Format results from OCR
      const ocrWords = ret.data.blocks
        .map((block) => block.paragraphs.map((paragraph) => paragraph.lines.map((line) => line.words)))
        .flat(3);

      const ocrResult = {
        text: ret.data.text,
        words: ocrWords.map((w) => ({
          text: w.text,
          x: w.bbox.x0,
          y: w.bbox.y0,
          width: w.bbox.x1 - w.bbox.x0,
          height: w.bbox.y1 - w.bbox.y0,
          confidence: Math.floor(w.confidence) // Floor confidence to avoid floating point issues
        }))
      };

      // 3. Compare OCR results with ground truth
      console.log('Comparing OCR results with ground truth...');

      // Compare full text (ignoring whitespace differences)
      expect(ocrResult.text.replace(/\s/g, '')).toEqual(groundTruth.text.replace(/\s/g, ''));

      // Compare number of words
      expect(ocrResult.words.length).toEqual(groundTruth.words.length);

      // Compare each word
      for (let i = 0; i < ocrResult.words.length; i++) {
        const ocrWord = ocrResult.words[i];
        const truthWord = groundTruth.words[i];

        // Compare text and confidence
        expect(ocrWord.text).toEqual(truthWord.text);
        expect(ocrWord.confidence).toBeGreaterThanOrEqual(truthWord.confidence - 2); // Allow a small tolerance

        // Check for valid bounding box values
        expect(ocrWord.x).toBeGreaterThanOrEqual(0);
        expect(ocrWord.y).toBeGreaterThanOrEqual(0);
        expect(ocrWord.width).toBeGreaterThanOrEqual(0);
        expect(ocrWord.height).toBeGreaterThanOrEqual(0);
      }

      console.log('Comparison successful!');

      await worker.terminate();
    } catch (error) {
      console.error('An error occurred during the OCR test:', error);
      throw error;
    }
  },
  { timeout: 30000 }
);
