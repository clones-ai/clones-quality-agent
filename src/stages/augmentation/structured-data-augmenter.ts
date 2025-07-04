import { ProcessedEvent, PipelineStage } from '../../shared/types';
import OpenAI from 'openai';
import { createWorker } from 'tesseract.js';
import path from 'path';
import fs from 'node:fs';

interface TextElement {
  text: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface StructuredQuery {
  query: string;
  response: any;
}

export class StructuredDataAugmenter implements PipelineStage<ProcessedEvent[], ProcessedEvent[]> {
  private openai: OpenAI;
  private worker: Awaited<ReturnType<typeof createWorker>> | null = null;

  constructor(private maxSamples: number = 3) {
    this.openai = new OpenAI();
  }

  async init() {
    if (!this.worker) {
      this.worker = await createWorker('eng');
    }
  }

  async cleanup() {
    if (this.worker) {
      await this.worker.terminate();
      this.worker = null;
    }
  }

  private async extractTextWithCoordinates(
    imageBase64: string,
    timestamp: number
  ): Promise<TextElement[]> {
    if (!this.worker) {
      await this.init();
    }

    // Convert base64 to buffer and save image for debugging
    const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(base64Data, 'base64');

    const debugPath = path.join(process.cwd(), 'data', 'debug');
    if (!fs.existsSync(debugPath)) {
      fs.mkdirSync(debugPath, { recursive: true });
    }
    const imagePath = path.join(debugPath, `ocr_frame_${timestamp}.png`);
    fs.writeFileSync(imagePath, imageBuffer);

    // Get words with bounding boxes
    const { data } = await this.worker!.recognize(imagePath, {}, { blocks: true });

    // Type guard for safety
    if (!data || !data.blocks) {
      console.log(`[StructuredDataAugmenter] No text blocks found in frame ${timestamp}`);
      return [];
    }

    const words = data.blocks
      .map((block) => block.paragraphs.map((paragraph) => paragraph.lines.map((line) => line.words)))
      .flat(3);

    // Save OCR output for debugging
    fs.writeFileSync(
      path.join(debugPath, `ocr_words_${timestamp}.json`),
      JSON.stringify(
        {
          text: data.text,
          words: words.map((w) => ({
            text: w.text,
            x: w.bbox.x0,
            y: w.bbox.y0,
            width: w.bbox.x1 - w.bbox.x0,
            height: w.bbox.y1 - w.bbox.y0,
            confidence: w.confidence
          }))
        },
        null,
        2
      )
    );

    // Convert to TextElement format
    const elements = words.map((word) => ({
      text: word.text,
      x: word.bbox.x0,
      y: word.bbox.y0,
      width: word.bbox.x1 - word.bbox.x0,
      height: word.bbox.y1 - word.bbox.y0
    }));

    // Log stats
    console.log(
      `[StructuredDataAugmenter] Extracted ${elements.length} words from frame ${timestamp}`
    );
    if (elements.length > 0) {
      console.log(`[StructuredDataAugmenter] Sample words:`, elements.slice(0, 3));
    }

    return elements;
  }

  private async generateStructuredQueries(elements: TextElement[]): Promise<StructuredQuery[]> {
    const prompt = `You are a structured data analyzer. Given text elements and their coordinates from a GUI screenshot, generate 3 queries about the interface layout and their responses.

The text elements are:
${elements.map((e) => `"${e.text}" at (${e.x},${e.y})`).join('\n')}

Respond with a JSON array containing exactly 3 objects. Each object should have a "query" field asking about some aspect of the interface (buttons, text fields, navigation, etc) and a "response" field with the structured answer.

Example response format (do not use markdown):
[{"query":"What buttons appear in the bottom-left corner?","response":{"buttons":[{"text":"Cancel","position":{"x":10,"y":450}},{"text":"Back","position":{"x":80,"y":450}}]}}]`;

    const response = await this.openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        {
          role: 'user',
          content: prompt
        }
      ],
      max_tokens: 1000,
      temperature: 0.7
    });

    const content = response.choices[0].message.content || '[]';
    try {
      return JSON.parse(content.replace(/```[a-z]*\n?|\n?```/g, ''));
    } catch (error) {
      console.error('Failed to parse OpenAI response:', content);
      throw error;
    }
  }

  async process(events: ProcessedEvent[]): Promise<ProcessedEvent[]> {
    await this.init();
    console.log('\n[StructuredDataAugmenter] Starting structured data analysis...');

    // Get frame events
    const frameEvents = events.filter((e) => e.type === 'frame');
    console.log(`[StructuredDataAugmenter] Found ${frameEvents.length} total frames`);

    // Randomly sample frames up to maxSamples
    const indices = Array.from({ length: frameEvents.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const sampledFrames = indices
      .slice(0, Math.min(this.maxSamples, frameEvents.length))
      .map((i) => frameEvents[i]);

    console.log(`[StructuredDataAugmenter] Selected ${sampledFrames.length} frames for analysis`);

    let completed = 0;
    for (const frameEvent of sampledFrames) {
      if (!frameEvent.data.frame) continue;

      try {
        console.log(
          `[StructuredDataAugmenter] Processing frame ${++completed}/${sampledFrames.length}`
        );

        // Extract text elements with coordinates
        const elements = await this.extractTextWithCoordinates(
          frameEvent.data.frame,
          frameEvent.timestamp
        );
        console.log(`[StructuredDataAugmenter] Extracted ${elements.length} text elements`);

        // Generate structured queries and responses
        const queries = await this.generateStructuredQueries(elements);
        console.log(`[StructuredDataAugmenter] Generated ${queries.length} structured queries`);

        // Add structured data event
        events.push({
          type: 'structured_data',
          timestamp: frameEvent.timestamp,
          data: {
            frame: frameEvent.data.frame,
            text: JSON.stringify(
              {
                elements,
                queries
              },
              null,
              2
            )
          }
        });
      } catch (error) {
        console.error(`[StructuredDataAugmenter] Error processing frame ${completed}:`, error);
      }
    }

    console.log(
      `[StructuredDataAugmenter] Completed structured data analysis. Processed ${completed} frames.\n`
    );
    await this.cleanup();
    return events;
  }
}
