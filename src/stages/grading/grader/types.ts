export interface GradeResult {
    /** One-paragraph outcome summary. */
    summary: string;
    /** Brief, high-level observations (2–6 lines). No chain-of-thought. */
    observations: string;
    /** Deterministic score (0–100) computed from component scores and weights. */
    score: number;
    /** Short, final rationale (no step-by-step reasoning). */
    reasoning: string;
    /** Model-reported confidence in [0.0, 1.0]. */
    confidence: number;
    /** Component scores (0–100). */
    outcomeAchievement: number;
    processQuality: number;
    efficiency: number;
}

export interface GraderConfig {
    apiKey: string;
    /** Number of events per chunk (pipeline-level slicing), if applicable. */
    chunkSize?: number;
    /** Model name (e.g., 'gpt-4o'). */
    model?: string;
    /** Per-request timeout in ms (default: 60000). */
    timeout?: number;
    /** Maximum retries per request (default: 3). */
    maxRetries?: number;
    /** Max images included per chunk (default: 3). */
    maxImagesPerChunk?: number;
    /** Max characters per text message (default: 3000). */
    maxTextPerMessage?: number;
    /** Seed for deterministic output (default: 42). */
    seed?: number;
    /** Rate limiter configuration (default: 10 tokens, 2/sec refill). */
    rateLimiter?: {
        maxTokens?: number;
        refillRate?: number;
    };
    /** Optional metrics hook for observability. */
    onMetrics?: MetricsHook;
}

export interface GraderLogger {
    debug(msg: string, err?: Error | undefined, meta?: Record<string, unknown>): void;
    info(msg: string, err?: Error | undefined, meta?: Record<string, unknown>): void;
    warn(msg: string, err?: Error | undefined, meta?: Record<string, unknown>): void;
    error(msg: string, err?: Error | undefined, meta?: Record<string, unknown>): void;
}

export interface RequestMetrics {
    /** OpenAI response ID for tracing */
    responseId?: string;
    /** System fingerprint for model version tracking */
    systemFingerprint?: string;
    /** Token usage details */
    usage?: {
        promptTokens: number;
        completionTokens: number;
        totalTokens: number;
    };
    /** Request timing and retry information */
    timing: {
        startTime: number;
        endTime: number;
        durationMs: number;
        retryCount: number;
        retryDelays: number[];
    };
    /** Request context */
    context: {
        sessionId: string;
        chunkIndex?: number;
        totalChunks?: number;
        isFinal: boolean;
        model: string;
    };
    /** Final outcome */
    outcome: 'success' | 'permanent_error' | 'transient_error' | 'timeout';
    /** Error details if failed */
    error?: {
        type: string;
        message: string;
        statusCode?: number;
    };
}

export interface MetricsHook {
    (metrics: RequestMetrics): void | Promise<void>;
}

export interface MetaData {
    sessionId: string;
    taskDescription?: string;
    platform?: "web" | "desktop" | "other";
}

/** Input items the model can consume for each chunk. */
export type LLMContentItem =
    | { type: "text"; text: string }
    | {
        type: "image";
        /** Base64-encoded image (PNG or JPEG). */
        data: string;
        mime?: "image/jpeg" | "image/png";
        /** Optional short note (e.g., click area / crop context). */
        cropInfo?: string;
    };

/** A chunk is a list of items describing a contiguous slice of the trajectory. */
export type Chunk = LLMContentItem[];

/** Scoring criteria with weights that must sum to 1.0. */
export interface EvaluationCriteria {
    outcomeAchievement: { weight: number };
    processQuality: { weight: number };
    efficiency: { weight: number };
}
