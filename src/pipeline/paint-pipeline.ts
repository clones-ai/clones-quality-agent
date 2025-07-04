import fs from 'node:fs';
import path from 'path';
import { ProcessedEvent } from '../shared/types';
import { QuickDrawLoader } from '../shared/utils/quickdraw-loader';
import { BSpline } from '../shared/utils/spline';

interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface PaintState {
  frame: string; // Path to frame image
  elements: {
    [key: string]: BBox;
  };
}

interface PaintMetadata {
  [state: string]: PaintState;
}

const REASONING_TEMPLATES = {
  newDrawing: [
    'I need to clear the canvas for a new drawing',
    'Let me start fresh by clearing the current canvas',
    "I'll clear this to make space for the next drawing",
    'Time to clear the canvas for a fresh start'
  ],
  clickFile: [
    "I'll click the File menu to find the clear option",
    'Opening the File menu to access canvas options',
    'Going to the File menu to start over',
    'Let me access the File menu first'
  ],
  clickNew: [
    'Selecting New to reset the canvas',
    'Clicking New to start fresh',
    'Creating a new canvas',
    'Going to create a new drawing space'
  ],
  savePrompt: [
    "I don't need to save the current drawing",
    "No need to save this since we're starting fresh",
    "I'll click No to discard the current drawing",
    'Clicking No to proceed with clearing'
  ],
  drawSegment: [
    'Drawing stroke {n} of {total}, starting {direction}',
    'Adding stroke {n}/{total} going {direction}',
    'Making stroke {n} of {total} {direction}',
    'For stroke {n}/{total}, drawing {direction}'
  ]
};

function getStrokeDirection(points: Array<{ x: number; y: number }>): string {
  if (points.length < 2) return 'forward';

  const dx = points[1].x - points[0].x;
  const dy = points[1].y - points[0].y;
  const angle = (Math.atan2(dy, dx) * 180) / Math.PI;

  // Convert angle to 8 main directions
  const directions = [
    { min: -22.5, max: 22.5, name: 'rightward' },
    { min: 22.5, max: 67.5, name: 'down and right' },
    { min: 67.5, max: 112.5, name: 'downward' },
    { min: 112.5, max: 157.5, name: 'down and left' },
    { min: 157.5, max: 180, name: 'leftward' },
    { min: -180, max: -157.5, name: 'leftward' },
    { min: -157.5, max: -112.5, name: 'up and left' },
    { min: -112.5, max: -67.5, name: 'upward' },
    { min: -67.5, max: -22.5, name: 'up and right' }
  ];

  for (const dir of directions) {
    if (angle >= dir.min && angle < dir.max) {
      return dir.name;
    }
  }

  return 'forward'; // fallback
}

export class PaintPipeline {
  private metadata: PaintMetadata;
  private loader: QuickDrawLoader;
  private drawingCanvas: any;
  private drawingCtx: any;
  private overlayCanvas: any;
  private overlayCtx: any;
  private currentFrame: string = '';
  private spline: BSpline;

  constructor(
    private dataDir: string,
    private metadataPath: string,
    private startTime: number = 0
  ) {
    this.metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
    this.loader = new QuickDrawLoader(startTime);
    this.spline = new BSpline(10); // Cubic B-spline for smooth curves

    // Initialize canvases lazily in initializeCanvases
    this.drawingCanvas = null;
    this.drawingCtx = null;
    this.overlayCanvas = null;
    this.overlayCtx = null;
  }

  private getRandomTemplate(templates: string[]): string {
    return templates[Math.floor(Math.random() * templates.length)];
  }

  private async loadImageAsBase64(imagePath: string): Promise<string> {
    const fullPath = path.join(this.dataDir, imagePath);
    const buffer = await fs.promises.readFile(fullPath);
    return buffer.toString('base64');
  }

  private bboxToCoords(bbox: BBox): { x: number; y: number } {
    return {
      x: bbox.x1 + (bbox.x2 - bbox.x1) / 2,
      y: bbox.y1 + (bbox.y2 - bbox.y1) / 2
    };
  }

  private async initializeCanvases(frame: string) {
    const { createCanvas, loadImage } = require('canvas');
    const fullPath = path.join(this.dataDir, frame);
    const image = await loadImage(fullPath);

    // Only create canvases if they don't exist yet
    if (!this.drawingCanvas) {
      this.drawingCanvas = createCanvas(image.width, image.height);
      this.drawingCtx = this.drawingCanvas.getContext('2d');

      this.overlayCanvas = createCanvas(image.width / 2, image.height / 2);
      this.overlayCtx = this.overlayCanvas.getContext('2d');
      this.overlayCtx.strokeStyle = 'black';
      this.overlayCtx.lineWidth = 1;
      this.overlayCtx.lineCap = 'round';
      this.overlayCtx.lineJoin = 'round';
    }

    // Always update base frame
    this.drawingCtx.drawImage(image, 0, 0);
    this.currentFrame = frame;
  }

  private clearDrawing() {
    if (this.overlayCtx) {
      const width = this.overlayCanvas.width;
      const height = this.overlayCanvas.height;
      this.overlayCtx.clearRect(0, 0, width, height);
    }
  }

  private drawSegment(points: Array<{ x: number; y: number }>) {
    // Scale points to half size
    const scaledPoints = points.map((p) => ({
      x: p.x / 2,
      y: p.y / 2
    }));

    // Generate smooth curve points using B-spline
    // const smoothPoints = this.spline.generateCurve(scaledPoints);
    const smoothPoints = scaledPoints;

    // Draw the smooth path
    this.overlayCtx.beginPath();
    this.overlayCtx.moveTo(smoothPoints[0].x, smoothPoints[0].y);
    for (let i = 1; i < smoothPoints.length; i++) {
      this.overlayCtx.lineTo(smoothPoints[i].x, smoothPoints[i].y);
    }
    this.overlayCtx.stroke();
  }

  private getCurrentFrame(): string {
    // Create a temporary canvas to avoid modifying the base frame
    const { createCanvas } = require('canvas');
    const tempCanvas = createCanvas(this.drawingCanvas.width, this.drawingCanvas.height);
    const tempCtx = tempCanvas.getContext('2d');

    // Draw the base frame
    tempCtx.drawImage(this.drawingCanvas, 0, 0);

    // Scale up overlay with point filtering (nearest neighbor)
    tempCtx.imageSmoothingEnabled = false;
    tempCtx.drawImage(
      this.overlayCanvas,
      0,
      0,
      this.overlayCanvas.width,
      this.overlayCanvas.height,
      0,
      0,
      this.drawingCanvas.width,
      this.drawingCanvas.height
    );

    // Convert to base64
    return tempCanvas.toDataURL().split(',')[1];
  }

  private async drawTrajectoryOnFrame(
    frame: string,
    points: Array<{ x: number; y: number }>
  ): Promise<string> {
    await this.initializeCanvases(frame);
    this.drawSegment(points);
    return this.getCurrentFrame();
  }

  async process(doodleNames: string[], numDoodles: number = 5): Promise<ProcessedEvent[]> {
    const events: ProcessedEvent[] = [];
    let currentTime = this.startTime;
    let processedDoodles = 0;

    // Keep trying until we have enough doodles
    while (processedDoodles < numDoodles) {
      // Pick a random doodle file
      const randomIndex = Math.floor(Math.random() * doodleNames.length);
      const doodleName = doodleNames[randomIndex];

      try {
        // Add placeholder quest that will be populated with first doodle event
        let quest_idx = events.length;
        events.push({
          type: 'quest',
          timestamp: currentTime,
          data: {}
        });
        currentTime += 500;

        // Clear canvas sequence for all except first doodle
        if (processedDoodles > 0) {
          // Add reasoning for clearing
          events.push({
            type: 'reasoning',
            timestamp: currentTime,
            data: { text: this.getRandomTemplate(REASONING_TEMPLATES.newDrawing) }
          });
          currentTime += 1000;

          // Click File
          const fileCoords = this.bboxToCoords(this.metadata.init.elements.FileMenu);
          events.push({
            type: 'reasoning',
            timestamp: currentTime,
            data: { text: this.getRandomTemplate(REASONING_TEMPLATES.clickFile) }
          });
          currentTime += 500;

          events.push({
            type: 'mouseclick',
            timestamp: currentTime,
            data: fileCoords
          });
          currentTime += 500;

          // Show file menu frame
          await this.initializeCanvases(this.metadata.file.frame);
          events.push({
            type: 'frame',
            timestamp: currentTime,
            data: { frame: this.getCurrentFrame() }
          });
          currentTime += 500;

          // Click New
          const newCoords = this.bboxToCoords(this.metadata.file.elements.New);
          events.push({
            type: 'reasoning',
            timestamp: currentTime,
            data: { text: this.getRandomTemplate(REASONING_TEMPLATES.clickNew) }
          });
          currentTime += 500;

          events.push({
            type: 'mouseclick',
            timestamp: currentTime,
            data: newCoords
          });
          currentTime += 500;

          // Show save prompt frame
          await this.initializeCanvases(this.metadata.save.frame);
          events.push({
            type: 'frame',
            timestamp: currentTime,
            data: { frame: this.getCurrentFrame() }
          });
          currentTime += 500;

          // Click No
          const noCoords = this.bboxToCoords(this.metadata.save.elements.DontSave);
          events.push({
            type: 'reasoning',
            timestamp: currentTime,
            data: { text: this.getRandomTemplate(REASONING_TEMPLATES.savePrompt) }
          });
          currentTime += 500;

          events.push({
            type: 'mouseclick',
            timestamp: currentTime,
            data: noCoords
          });
          currentTime += 500;

          // Clear drawing after clicking No
          this.clearDrawing();
        }

        // Show initial canvas frame
        await this.initializeCanvases(this.metadata.init.frame);
        events.push({
          type: 'frame',
          timestamp: currentTime,
          data: { frame: this.getCurrentFrame() }
        });
        currentTime += 1000;

        // Load one random drawing from this doodle file
        const doodleEvents = await this.loader.loadRandomDrawingFromNDJSON(
          path.join(this.dataDir, 'doodles', `${doodleName}.ndjson`),
          {
            x: this.metadata.init.elements.Canvas.x1,
            y: this.metadata.init.elements.Canvas.y1,
            width: this.metadata.init.elements.Canvas.x2 - this.metadata.init.elements.Canvas.x1,
            height: this.metadata.init.elements.Canvas.y2 - this.metadata.init.elements.Canvas.y1
          },
          currentTime
        );

        // Assert first event is a quest and copy its data
        if (!doodleEvents[0] || doodleEvents[0].type !== 'quest') {
          throw new Error('First doodle event must be a quest');
        }
        events[quest_idx].data = { ...doodleEvents[0].data };
        doodleEvents.shift(); // Remove the quest event since we already used it

        // TODO: remove this bandaid fix and find out whats causing the NaNs
        // Check for NaN coordinates in any doodle event
        const hasNaNCoords = doodleEvents.some(
          (event) =>
            event.type === 'mousedrag' &&
            event.data.coordinates?.some((coord) => Number.isNaN(coord.x) || Number.isNaN(coord.y))
        );

        if (hasNaNCoords) {
          // Delete all events since the quest
          events.splice(quest_idx);
          // Rewind time
          currentTime = events[events.length - 1]?.timestamp || this.startTime;
          // Decrement processed doodles
          processedDoodles--;
          continue;
        }

        const num_strokes = events.filter((e) => e.type === 'mousedrag').length;

        // Process each doodle event
        for (const event of doodleEvents) {
          if (event.type === 'mousedrag') {
            // Get stroke direction from coordinates
            const direction = getStrokeDirection(event.data.coordinates!);

            // Add reasoning for this segment
            events.push({
              type: 'reasoning',
              timestamp: event.timestamp - 1,
              data: {
                text: this.getRandomTemplate(REASONING_TEMPLATES.drawSegment)
                  .replace(
                    '{n}',
                    (events.filter((e) => e.type === 'mousedrag').length - num_strokes).toString()
                  )
                  .replace(
                    '{total}',
                    doodleEvents.filter((e) => e.type === 'mousedrag').length.toString()
                  )
                  .replace('{direction}', direction)
              }
            });
          }
          events.push(event);
          if (event.type === 'mousedrag' && event.data.coordinates?.length) {
            // Add frame showing the drawn trajectory
            events.push({
              type: 'frame',
              timestamp: event.timestamp + 1,
              data: {
                frame: await this.drawTrajectoryOnFrame(
                  this.metadata.init.frame,
                  event.data.coordinates!
                )
              }
            });
          }

          // Update current time to last event's timestamp
          currentTime = event.timestamp;
        }

        processedDoodles++;
        currentTime += 2000; // Add extra time between doodles
      } catch (error: unknown) {
        console.warn(`Failed to process doodle ${doodleName}:`, error);
        // Continue to next doodle without incrementing processedDoodles
      }
    }

    if (processedDoodles === 0) {
      throw new Error('Failed to process any doodles');
    }

    return events;
  }
}
