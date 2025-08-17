import { OpenAI } from 'openai';
import path from 'path';
import { sleep } from '../../shared/utils/sleep';

export interface MetaData {
  readonly id: string;
  readonly timestamp: string;
  readonly duration_seconds: number;
  readonly status: string;
  readonly reason: string;
  readonly title: string;
  readonly description: string;
  readonly platform: string;
  readonly arch: string;
  readonly version: string;
  readonly locale: string;
  readonly primary_monitor: {
    readonly width: number;
    readonly height: number;
  };
  readonly quest: {
    readonly title: string;
    readonly app: string;
    readonly icon_url: string;
    readonly objectives: readonly string[];
    readonly content: string;
  };
}

export interface Message {
  readonly role: 'user' | 'assistant';
  readonly content: string | {
    readonly type: 'image';
    readonly data: string;
  };
}

export interface GradeResult {
  readonly summary: string;
  readonly observations: string;
  readonly score: number;
  readonly reasoning: string;
  readonly confidence: number;
  readonly outcomeAchievement: number;
  readonly processQuality: number;
  readonly efficiency: number;
}

export interface GraderConfig {
  readonly apiKey: string;
  readonly chunkSize?: number;
  readonly model?: string;
  readonly timeout?: number;
  readonly maxRetries?: number;
}

export interface GraderLogger {
  info(message: string, meta?: Record<string, unknown>): void;
  error(message: string, error?: Error, meta?: Record<string, unknown>): void;
  debug(message: string, meta?: Record<string, unknown>): void;
}

class DefaultLogger implements GraderLogger {
  info(message: string, meta?: Record<string, unknown>): void {
    console.log(`[INFO] ${message}`, meta ? JSON.stringify(meta) : '');
  }

  error(message: string, error?: Error, meta?: Record<string, unknown>): void {
    console.error(`[ERROR] ${message}`, error?.message || '', meta ? JSON.stringify(meta) : '');
  }

  debug(message: string, meta?: Record<string, unknown>): void {
    console.debug(`[DEBUG] ${message}`, meta ? JSON.stringify(meta) : '');
  }
}

interface EvaluationCriteria {
  outcomeAchievement: {
    weight: number;
    description: string;
  };
  processQuality: {
    weight: number;
    description: string;
  };
  efficiency: {
    weight: number;
    description: string;
  };
}

interface ChunkEvaluation {
  summary: string;
}

interface FinalEvaluation {
  summary: string;
  observations: string;
  score: number;
  reasoning: string;
  confidence: number;
  outcomeAchievement: number;
  processQuality: number;
  efficiency: number;
}

// JSON Schema definitions for Structured Outputs
const FINAL_EVALUATION_SCHEMA = {
  name: "final_evaluation",
  strict: true,
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["summary", "observations", "score", "reasoning", "confidence", "outcomeAchievement", "processQuality", "efficiency"],
    properties: {
      summary: {
        type: "string",
        minLength: 1,
        description: "Comprehensive bullet-point overview of all progress made across all chunks"
      },
      observations: {
        type: "string",
        minLength: 1,
        description: "Brief, high-level bullet points (2-4 max) of key insights from evaluation. No detailed step-by-step reasoning."
      },
      score: {
        type: "integer",
        minimum: 0,
        maximum: 100,
        description: "Overall score based on weighted evaluation criteria"
      },
      reasoning: {
        type: "string",
        minLength: 1,
        description: "Clear justification for the final score based on the evaluation framework. Brief and high-level."
      },
      confidence: {
        type: "number",
        minimum: 0,
        maximum: 1,
        description: "Confidence level in the evaluation (0.0 to 1.0)"
      },
      outcomeAchievement: {
        type: "integer",
        minimum: 0,
        maximum: 100,
        description: "Score for goal completion and objective fulfillment"
      },
      processQuality: {
        type: "integer",
        minimum: 0,
        maximum: 100,
        description: "Score for problem-solving approach, error recovery, and adaptability"
      },
      efficiency: {
        type: "integer",
        minimum: 0,
        maximum: 100,
        description: "Score for time management, direct paths, and resource utilization"
      }
    }
  }
} as const;

const CHUNK_EVALUATION_SCHEMA = {
  name: "chunk_evaluation",
  strict: true,
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["summary"],
    properties: {
      summary: {
        type: "string",
        minLength: 1,
        description: "Comprehensive bullet-point overview of all progress made so far, combining previous summary with this chunk's accomplishments"
      }
    }
  }
} as const;

export class Grader {
  private readonly client: OpenAI;
  private readonly chunkSize: number;
  private readonly model: string;
  private readonly evaluationCriteria: EvaluationCriteria;
  private readonly maxRetries: number;
  private readonly logger: GraderLogger;

  constructor(config: GraderConfig, logger?: GraderLogger) {
    if (!config.apiKey.trim()) {
      throw new Error('OpenAI API key is required and cannot be empty');
    }

    this.chunkSize = Math.max(1, config.chunkSize ?? 4);
    this.maxRetries = Math.max(1, config.maxRetries ?? 3);
    this.logger = logger ?? new DefaultLogger();

    this.client = new OpenAI({
      apiKey: config.apiKey,
      timeout: config.timeout ?? 60 * 1000, // 60-second timeout by default
      maxRetries: this.maxRetries
    });

    // Use environment variable GRADER_MODEL if available, otherwise use provided model or default to gpt-4o
    this.model = config.model || process.env.GRADER_MODEL || 'gpt-4o';

    this.logger.info('Grader initialized', {
      model: this.model,
      chunkSize: this.chunkSize,
      maxRetries: this.maxRetries,
      timeout: config.timeout ?? 60000
    });

    // Evaluation framework: Outcome (50%) + Process (30%) + Efficiency (20%)
    this.evaluationCriteria = {
      outcomeAchievement: {
        weight: 0.5,
        description: 'Goal completion and objective fulfillment'
      },
      processQuality: {
        weight: 0.3,
        description: 'Problem-solving approach, error recovery, and adaptability'
      },
      efficiency: {
        weight: 0.2,
        description: 'Time management, direct paths, and resource utilization'
      }
    };
  }

  // Legacy constructor for backward compatibility
  static create(apiKey: string, chunkSize: number = 4, model?: string): Grader {
    return new Grader({ apiKey, chunkSize, model });
  }

  // Method to update evaluation criteria (useful for different evaluation contexts)
  updateEvaluationCriteria(criteria: Partial<{
    outcomeAchievement: Partial<{ weight: number; description: string }>;
    processQuality: Partial<{ weight: number; description: string }>;
    efficiency: Partial<{ weight: number; description: string }>;
  }>): void {
    if (criteria.outcomeAchievement) {
      this.evaluationCriteria.outcomeAchievement = { ...this.evaluationCriteria.outcomeAchievement, ...criteria.outcomeAchievement };
    }
    if (criteria.processQuality) {
      this.evaluationCriteria.processQuality = { ...this.evaluationCriteria.processQuality, ...criteria.processQuality };
    }
    if (criteria.efficiency) {
      this.evaluationCriteria.efficiency = { ...this.evaluationCriteria.efficiency, ...criteria.efficiency };
    }

    // Validate that weights sum to 1.0 (with small tolerance)
    const totalWeight = this.evaluationCriteria.outcomeAchievement.weight +
      this.evaluationCriteria.processQuality.weight +
      this.evaluationCriteria.efficiency.weight;

    if (Math.abs(totalWeight - 1.0) > 0.01) {
      this.logger.error('Evaluation criteria weights do not sum to 1.0', undefined, {
        totalWeight,
        weights: {
          outcome: this.evaluationCriteria.outcomeAchievement.weight,
          process: this.evaluationCriteria.processQuality.weight,
          efficiency: this.evaluationCriteria.efficiency.weight
        }
      });
      throw new Error(`Evaluation criteria weights must sum to 1.0, got ${totalWeight}`);
    }

    this.logger.info('Evaluation criteria updated', {
      outcomeWeight: this.evaluationCriteria.outcomeAchievement.weight,
      processWeight: this.evaluationCriteria.processQuality.weight,
      efficiencyWeight: this.evaluationCriteria.efficiency.weight,
      totalWeight
    });
  }

  // Method to get current evaluation criteria (readonly)
  getEvaluationCriteria(): Readonly<EvaluationCriteria> {
    return {
      outcomeAchievement: { ...this.evaluationCriteria.outcomeAchievement },
      processQuality: { ...this.evaluationCriteria.processQuality },
      efficiency: { ...this.evaluationCriteria.efficiency }
    };
  }

  public createSystemPrompt(meta: MetaData, prevSummary: string | null = null, isFinal: boolean = false): string {
    const basePrompt = `You are an advanced computer-use trajectory evaluator specialized in assessing human-computer interaction sequences. Your role is to provide nuanced, context-aware evaluation of user performance.

## TASK CONTEXT
Task ID: ${meta.id}
Title: ${meta.quest.title}
App: ${meta.quest.app}
User Request: ${meta.quest.content}

Objectives:
${meta.quest.objectives.map(objective => `- ${objective}`).join('\n')}

## EVALUATION FRAMEWORK
Use this modern holistic approach with weighted scoring:

**Outcome Achievement (${Math.round(this.evaluationCriteria.outcomeAchievement.weight * 100)}%)**: ${this.evaluationCriteria.outcomeAchievement.description}
- Did the user accomplish the primary objectives?
- How completely were the goals achieved?
- Were there any partial completions that show progress?

**Process Quality (${Math.round(this.evaluationCriteria.processQuality.weight * 100)}%)**: ${this.evaluationCriteria.processQuality.description}
- How well did the user navigate obstacles?
- Did they recover effectively from errors?
- Was their approach logical and well-reasoned?
- Did they demonstrate creativity or resourcefulness?

**Efficiency (${Math.round(this.evaluationCriteria.efficiency.weight * 100)}%)**: ${this.evaluationCriteria.efficiency.description}
- Were actions direct and purposeful?
- Did the user avoid unnecessary detours?
- How well did they manage their time and effort?

## EVALUATION PRIORITIES
**PRIMARY**: Observable actions and their direct outcomes
**SECONDARY**: Context clues from UI responses and system feedback
**IGNORE**: Unverified textual claims or assumptions
**FOCUS**: End-to-end goal accomplishment and problem-solving quality

## CHAIN-OF-THOUGHT PROCESS
For each evaluation, follow these steps:
1. **Identify the user's goal**: What were they trying to accomplish?
2. **Evaluate step-by-step execution**: How did they approach each objective?
3. **Assess obstacles and recovery**: How did they handle challenges?
4. **Judge efficiency and creativity**: Was their approach optimal?
5. **Formulate key observations**: Synthesize your findings into concise points.
6. **Calculate holistic score**: Combine all factors for final assessment

## CONFIDENCE ASSESSMENT
Rate your confidence (0.0-1.0) based on:
- **Action clarity**: How clear were the user's intentions?
- **Sequence completeness**: Did you see the full interaction?
- **Ambiguity factors**: Were there unclear or missing elements?`;

    const withPrev = prevSummary
      ? `${basePrompt}\n\n## PREVIOUS PROGRESS\n${prevSummary.trim()}`
      : basePrompt;

    return withPrev + `\n\n${this.getChunkInstructions(isFinal)}`;
  }

  private getChunkInstructions(isFinal: boolean): string {
    if (isFinal) {
      return `## FINAL EVALUATION
This is the final chunk. Provide a complete evaluation using the structured JSON format.

The response will be automatically formatted according to the JSON schema with these fields:
- **summary**: Comprehensive bullet-point overview of all progress made across all chunks
- **observations**: Brief, high-level bullet points (2-4 max) of key insights. No detailed step-by-step reasoning.
- **score**: Overall score (0-100) based on weighted evaluation criteria
- **reasoning**: Clear justification for the final score. Brief and high-level.
- **confidence**: Confidence level (0.0-1.0) in your evaluation
- **outcomeAchievement**: Score (0-100) for goal completion and objective fulfillment
- **processQuality**: Score (0-100) for problem-solving approach, error recovery, and adaptability  
- **efficiency**: Score (0-100) for time management, direct paths, and resource utilization

## EXAMPLE EVALUATION CONTEXT

**Task**: "Order a large pepperoni pizza for delivery"
**Objectives**: ["Navigate to pizza website", "Select large pepperoni pizza", "Add to cart", "Complete checkout with delivery"]

**Actions Observed**: User navigates to Domino's website, browses menu, adds large pepperoni to cart, but gets distracted and closes browser before checkout.

**Expected Response**: 
- Summary would list each step taken
- Observations would note UI proficiency but session abandonment  
- Score around 45 (partial completion)
- Reasoning would explain the 75% completion but critical checkout failure
- Component scores: Outcome 60%, Process 70%, Efficiency 80%`;
    } else {
      return `## CHUNK EVALUATION
Provide a JSON summary of progress combining previous summary (if any) with this chunk's accomplishments.

The response will be automatically formatted with a single field:
- **summary**: Comprehensive bullet-point overview of all progress made so far`;
    }
  }

  private chunkMessages(messages: readonly Message[], chunkSize: number): readonly Message[][] {
    if (!Array.isArray(messages) || messages.length === 0) {
      this.logger.error('Invalid messages array provided to chunkMessages');
      return [];
    }

    // Filter out scroll messages first
    const filteredMessages = messages.filter((msg): msg is Message => {
      if (typeof msg.content === 'string') {
        let content = msg.content;
        // Remove python code block if present
        if (content.startsWith('```python\n')) {
          content = content.slice(10, -4); // Remove ```python\n and \n```
        }
        // Filter out if it starts with scroll
        return !content.startsWith('scroll(');
      }
      // Keep all image messages
      return true;
    });

    this.logger.debug('Messages filtered', {
      originalCount: messages.length,
      filteredCount: filteredMessages.length
    });

    // Then chunk the filtered messages
    const chunks: Message[][] = [];
    for (let i = 0; i < filteredMessages.length; i += chunkSize) {
      chunks.push(filteredMessages.slice(i, i + chunkSize));
    }

    this.logger.debug('Messages chunked', {
      chunkCount: chunks.length,
      chunkSize: chunkSize
    });

    return chunks;
  }

  private extractClickCoordinates(message: string): [number, number] | null {
    const match = message.match(/click\((\d+),\s*(\d+)\)/);
    if (match) {
      return [parseInt(match[1]), parseInt(match[2])];
    }
    return null;
  }

  private formatMessageContent(content: string | { type: string; data: string }, prevMessage?: string): any {
    if (typeof content === 'string') {
      return content;
    }

    if (content.type === 'image') {
      let cropInfo = '';
      if (prevMessage) {
        const coords = this.extractClickCoordinates(prevMessage);
        if (coords) {
          cropInfo = ` The image is cropped to a 768x768 area centered around the cursor at coordinates ${coords}.`;
        }
      }

      return [
        {
          type: 'image_url',
          image_url: {
            url: `data:image/jpeg;base64,${content.data}`
          }
        },
        {
          type: 'text',
          text: `Screenshot of the application.${cropInfo}`
        }
      ];
    }

    return String(content);
  }

  private async evaluateChunk(
    systemPrompt: string,
    messages: readonly Message[],
    isFinal: boolean,
    chunkIndex: number = 0,
    totalChunks: number = 1
  ): Promise<string | null> {
    try {
      // Add chunk metadata to system prompt
      const actionCount = messages.length;

      const enhancedSystemPrompt = `${systemPrompt}

## CHUNK ANALYSIS CONTEXT
**Chunk Progress**: ${chunkIndex + 1} of ${totalChunks} chunks
**Action Density**: ${actionCount} user actions in this segment
**Analysis Scope**: ${actionCount === 0 ? 'Static observation only' : 'Active user interaction sequence'}

## ADVANCED EVALUATION DIRECTIVES

### 🎯 **Action Boundary Recognition**
- **STRICT BOUNDARY**: Only evaluate actions between BEGIN_ACTIONS and END_ACTIONS markers
- **NO HALLUCINATION**: Never infer actions that aren't explicitly shown
- **SCREENSHOT CONTEXT**: Use visual information to understand action outcomes, not to assume actions

### 🧠 **Cognitive Load Assessment**
- **Task Complexity**: Consider the inherent difficulty of the current objectives
- **Context Switching**: Evaluate how well the user manages multiple UI elements
- **Decision Points**: Identify moments where the user had to make strategic choices

### 🔄 **Adaptive Evaluation Framework**
${actionCount === 0 ?
          `**ZERO-ACTION CHUNK PROTOCOL**:
- Focus on environmental observation and context preservation
- Note any UI changes or system responses that occurred
- Maintain continuity with previous progress without fabricating user actions
- Assess whether inaction was strategic (waiting) or problematic (confusion)` :
          `**ACTIVE CHUNK PROTOCOL**:
- Analyze action sequences for logical progression
- Evaluate decision quality at each interaction point
- Assess recovery strategies when encountering obstacles
- Measure efficiency through action-to-outcome ratios`}

### 📊 **Modern Scoring Considerations**
- **Outcome Achievement**: Did actions advance toward stated objectives?
- **Process Intelligence**: Was the approach methodical and well-reasoned?
- **Adaptive Behavior**: How well did the user respond to unexpected situations?
- **Efficiency Metrics**: Were actions direct and purposeful?

### 🔍 **Evidence-Based Analysis**
- **PRIMARY EVIDENCE**: Direct user actions (clicks, types, scrolls)
- **SECONDARY EVIDENCE**: UI responses, system feedback, visual changes
- **CONTEXTUAL CLUES**: Application state, error messages, success indicators
- **IGNORE**: Text overlays claiming completion without corresponding actions

${actionCount === 0 ?
          "⚠️ **CRITICAL**: This chunk contains no user actions. Focus on environmental continuity and context preservation. Do not fabricate user interactions." :
          `✅ **ACTIVE ANALYSIS**: ${actionCount} actions detected. Evaluate the complete interaction sequence for strategic coherence and goal progression.`}`;

      const formattedMessages: Array<{
        role: 'system' | 'user' | 'assistant';
        content: any;
      }> = [{ role: 'system', content: enhancedSystemPrompt }];

      // Add a clear marker for the beginning of actions with context
      const contextualIntro = actionCount === 0
        ? `=== BEGIN_OBSERVATION_WINDOW ===
📸 STATIC ANALYSIS: This chunk contains screenshots only - no user actions to evaluate.
🔍 FOCUS: Environmental context, UI state, and continuity with previous progress.
⚠️  DO NOT infer or hallucinate user actions that aren't explicitly documented.`
        : `=== BEGIN_ACTIONS (${actionCount} user interactions) ===
🎯 ACTIVE ANALYSIS: Evaluating ${actionCount} user action${actionCount > 1 ? 's' : ''} in sequence.
🧠 FOCUS: Action logic, decision quality, goal progression, and efficiency.
📊 CONTEXT: Chunk ${chunkIndex + 1}/${totalChunks} - ${Math.round((actionCount / Math.max(1, totalChunks)) * 100)}% action density.`;

      formattedMessages.push({
        role: 'user',
        content: contextualIntro
      });

      for (let i = 0; i < messages.length; i++) {
        const prevMessage = i > 0 ? messages[i - 1].content : undefined;
        formattedMessages.push({
          role: 'user',
          content: this.formatMessageContent(messages[i].content, typeof prevMessage === 'string' ? prevMessage : undefined)
        });
      }

      // Add a clear marker for the end of actions with summary
      const contextualOutro = actionCount === 0
        ? `=== END_OBSERVATION_WINDOW ===
📋 SUMMARY: Static analysis complete - no user actions detected.
🎯 NEXT: Provide environmental context and continuity assessment in JSON format.`
        : `=== END_ACTIONS (${actionCount} user interactions) ===
📋 SEQUENCE COMPLETE: All ${actionCount} user action${actionCount > 1 ? 's' : ''} documented above.
🎯 NEXT: Provide comprehensive evaluation following the modern framework in JSON format.
⚡ REMINDER: Focus on outcome achievement, process quality, and efficiency.`;

      formattedMessages.push({
        role: 'user',
        content: contextualOutro
      });

      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: formattedMessages,
        max_tokens: isFinal ? 1500 : 800,
        temperature: 0,
        response_format: {
          type: "json_schema",
          json_schema: isFinal ? FINAL_EVALUATION_SCHEMA : CHUNK_EVALUATION_SCHEMA
        }
      });

      return response.choices[0].message.content;
    } catch (error) {
      this.logger.error('Error calling OpenAI API', error as Error, {
        chunkIndex: chunkIndex + 1,
        totalChunks,
        actionCount: messages.length
      });
      return null;
    }
  }

  private parseStructuredResponse(text: string): any {
    try {
      // With Structured Outputs, the response is guaranteed to be valid JSON
      return JSON.parse(text.trim());
    } catch (error) {
      // This should rarely happen with Structured Outputs
      this.logger.error('Failed to parse structured response', error as Error, {
        rawResponse: text?.substring(0, 200) + '...'
      });
      return null;
    }
  }

  async grade(meta: MetaData, sft: readonly Message[]): Promise<GradeResult | null>;
  async grade(metaPath: string, sftPath: string): Promise<GradeResult | null>;
  async grade(
    metaOrPath: MetaData | string,
    sftOrPath: readonly Message[] | string
  ): Promise<GradeResult | null> {
    const startTime = Date.now();

    try {
      let meta: MetaData;
      let sft: readonly Message[];

      if (typeof metaOrPath === 'string' && typeof sftOrPath === 'string') {
        this.logger.info('Reading input files', { metaPath: metaOrPath, sftPath: sftOrPath });

        // Read input files if paths are provided
        const [metaFile, sftFile] = await Promise.all([
          Bun.file(metaOrPath).json(),
          Bun.file(sftOrPath).json()
        ]);

        meta = metaFile as MetaData;
        sft = sftFile as readonly Message[];
      } else {
        // Use provided data directly
        meta = metaOrPath as MetaData;
        sft = sftOrPath as readonly Message[];
      }

      // Validate inputs
      if (!meta?.id || !meta?.quest?.objectives) {
        throw new Error('Invalid metadata: missing required fields');
      }

      if (!Array.isArray(sft) || sft.length === 0) {
        throw new Error('Invalid SFT data: must be a non-empty array');
      }

      // Split messages into chunks
      const chunks = this.chunkMessages(sft, this.chunkSize);
      const totalChunks = chunks.length;

      if (totalChunks === 0) {
        this.logger.error('No chunks to process after filtering messages');
        return null;
      }

      this.logger.info('Starting grading process', {
        sessionId: meta.id,
        totalChunks,
        totalMessages: sft.length,
        objectives: meta.quest.objectives.length
      });

      // Process each chunk
      let prevSummary: string | null = null;

      for (const [i, chunk] of chunks.entries()) {
        const isFinal = i === chunks.length - 1;
        const chunkResult = await this.processChunkWithRetries(
          meta,
          chunk,
          prevSummary,
          isFinal,
          i,
          totalChunks
        );

        if (!chunkResult.success) {
          this.logger.error('Failed to process chunk after retries', undefined, {
            chunkIndex: i + 1,
            totalChunks,
            sessionId: meta.id
          });
          return null;
        }

        if (isFinal && chunkResult.result) {
          const duration = Date.now() - startTime;
          this.logger.info('Grading completed successfully', {
            sessionId: meta.id,
            score: chunkResult.result.score,
            duration: `${duration}ms`
          });
          return chunkResult.result;
        }

        prevSummary = chunkResult.summary ?? null;
      }

      this.logger.error('Unexpected end of chunk processing');
      return null;
    } catch (error) {
      const duration = Date.now() - startTime;
      this.logger.error('Error during grading', error as Error, {
        duration: `${duration}ms`,
        sessionId: typeof metaOrPath === 'object' ? metaOrPath.id : 'unknown'
      });
      return null;
    }
  }

  private async processChunkWithRetries(
    meta: MetaData,
    chunk: readonly Message[],
    prevSummary: string | null,
    isFinal: boolean,
    chunkIndex: number,
    totalChunks: number
  ): Promise<{ success: boolean; result?: GradeResult; summary?: string }> {
    let retries = 0;

    while (retries < this.maxRetries) {
      const isRetry = retries > 0;

      this.logger.info('Processing chunk', {
        chunkIndex: chunkIndex + 1,
        totalChunks,
        attempt: retries + 1,
        maxRetries: this.maxRetries,
        isFinal,
        isRetry
      });

      const systemPrompt = this.createSystemPrompt(meta, prevSummary, isFinal);
      const evaluation = await this.evaluateChunk(systemPrompt, chunk, isFinal, chunkIndex, totalChunks);

      if (!evaluation) {
        this.logger.error('Failed to get evaluation from API', undefined, {
          chunkIndex: chunkIndex + 1,
          attempt: retries + 1
        });
        retries++;
        if (retries < this.maxRetries) {
          const delay = Math.min(8000, 500 * 2 ** (retries - 1)) + Math.random() * 250; // Exponential backoff with jitter
          this.logger.info(`Retrying after ${delay.toFixed(0)}ms delay...`);
          await sleep(delay);
        }
        continue;
      }

      if (isFinal) {
        const result = this.parseFinalEvaluation(evaluation);
        if (result) {
          this.logger.debug('Final evaluation parsed successfully', {
            chunkIndex: chunkIndex + 1,
            score: result.score
          });
          return { success: true, result };
        }

        this.logger.error('Failed to parse final evaluation JSON', undefined, {
          chunkIndex: chunkIndex + 1,
          attempt: retries + 1,
          evaluation: evaluation.substring(0, 200) + '...'
        });
        retries++;
        if (retries < this.maxRetries) {
          const delay = Math.min(8000, 500 * 2 ** (retries - 1)) + Math.random() * 250;
          this.logger.info(`Retrying after ${delay.toFixed(0)}ms delay...`);
          await sleep(delay);
        }
      } else {
        const summary = this.parseChunkEvaluation(evaluation);
        if (summary) {
          this.logger.debug('Intermediate summary extracted', {
            chunkIndex: chunkIndex + 1,
            summary: summary.substring(0, 100) + '...'
          });
          return { success: true, summary };
        }

        this.logger.error('Failed to parse chunk evaluation JSON', undefined, {
          chunkIndex: chunkIndex + 1,
          attempt: retries + 1,
          evaluation: evaluation.substring(0, 200) + '...'
        });
        retries++;
        if (retries < this.maxRetries) {
          const delay = Math.min(8000, 500 * 2 ** (retries - 1)) + Math.random() * 250;
          this.logger.info(`Retrying after ${delay.toFixed(0)}ms delay...`);
          await sleep(delay);
        }
      }
    }

    return { success: false };
  }

  private parseFinalEvaluation(evaluation: string): GradeResult | null {
    const finalEval = this.parseStructuredResponse(evaluation) as FinalEvaluation;

    if (!finalEval) {
      return null;
    }

    // With Structured Outputs, all required fields are guaranteed to be present and of correct type
    // JSON Schema already enforces score/confidence bounds, but we still validate component scores
    const outcomeScore = Math.max(0, Math.min(100, finalEval.outcomeAchievement));
    const processScore = Math.max(0, Math.min(100, finalEval.processQuality));
    const efficiencyScore = Math.max(0, Math.min(100, finalEval.efficiency));
    const confidence = Math.max(0, Math.min(1, finalEval.confidence));

    // Calculate expected weighted score based on criteria
    const expectedScore = Math.round(
      outcomeScore * this.evaluationCriteria.outcomeAchievement.weight +
      processScore * this.evaluationCriteria.processQuality.weight +
      efficiencyScore * this.evaluationCriteria.efficiency.weight
    );

    // Log score validation (allow some tolerance for LLM rounding)
    const scoreDifference = Math.abs(finalEval.score - expectedScore);
    if (scoreDifference > 10) {
      this.logger.debug('Score calculation variance detected', {
        providedScore: finalEval.score,
        expectedScore: expectedScore,
        difference: scoreDifference,
        components: {
          outcome: `${outcomeScore} * ${this.evaluationCriteria.outcomeAchievement.weight}`,
          process: `${processScore} * ${this.evaluationCriteria.processQuality.weight}`,
          efficiency: `${efficiencyScore} * ${this.evaluationCriteria.efficiency.weight}`
        }
      });
    }

    return {
      summary: finalEval.summary,
      observations: finalEval.observations,
      score: finalEval.score,
      reasoning: finalEval.reasoning,
      confidence: confidence,
      outcomeAchievement: outcomeScore,
      processQuality: processScore,
      efficiency: efficiencyScore
    };
  }

  private parseChunkEvaluation(evaluation: string): string | null {
    const chunkEval = this.parseStructuredResponse(evaluation) as ChunkEvaluation;
    // With Structured Outputs, summary is guaranteed to be present
    return chunkEval?.summary || null;
  }
}