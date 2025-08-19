import { z } from "zod";

export const FINAL_EVALUATION_SCHEMA = {
    name: "final_evaluation",
    strict: true,
    schema: {
        type: "object",
        additionalProperties: false,
        required: [
            "summary",
            "observations",
            "reasoning",
            "score",
            "confidence",
            "outcomeAchievement",
            "processQuality",
            "efficiency",
        ],
        properties: {
            summary: { type: "string", minLength: 1 },
            observations: {
                type: "string",
                minLength: 1,
                description:
                    "Brief, high-level observations (2–6 lines). Do NOT reveal chain-of-thought.",
            },
            reasoning: {
                type: "string",
                minLength: 1,
                description:
                    "Short, final justification of the scores. No chain-of-thought.",
            },
            score: { type: "integer", minimum: 0, maximum: 100 },
            confidence: { type: "integer", minimum: 0, maximum: 100 },
            outcomeAchievement: { type: "integer", minimum: 0, maximum: 100 },
            processQuality: { type: "integer", minimum: 0, maximum: 100 },
            efficiency: { type: "integer", minimum: 0, maximum: 100 },
        },
    },
} as const;

export const CHUNK_EVALUATION_SCHEMA = {
    name: "chunk_evaluation",
    strict: true,
    schema: {
        type: "object",
        additionalProperties: false,
        required: ["summary"],
        properties: {
            summary: { type: "string", minLength: 1 },
        },
    },
} as const;

export const ChunkEvaluationSchema = z.object({
    summary: z.string().min(1, "Summary must not be empty")
});

export const FinalEvaluationSchema = z.object({
    summary: z.string().min(1, "Summary must not be empty"),
    observations: z.string()
        .min(1, "Observations must not be empty")
        .refine((obs) => {
            // More flexible validation: accept bullet points, numbered lists, or line breaks
            const lines = obs.split('\n').filter(line => line.trim().length > 0);
            const bulletPoints = obs.split(/[•\-\*]/).filter(point => point.trim().length > 0);

            // Accept either 2-6 lines OR 2-6 bullet points
            return (lines.length >= 2 && lines.length <= 6) ||
                (bulletPoints.length >= 2 && bulletPoints.length <= 6);
        }, "Observations must contain between 2 and 6 non-empty lines or bullet points"),
    reasoning: z.string().min(1, "Reasoning must not be empty"),
    score: z.number().int().min(0).max(100),
    confidence: z.number().int().min(0).max(100),
    outcomeAchievement: z.number().int().min(0).max(100),
    processQuality: z.number().int().min(0).max(100),
    efficiency: z.number().int().min(0).max(100)
});

export type ChunkEvaluation = z.infer<typeof ChunkEvaluationSchema>;
export type FinalEvaluation = z.infer<typeof FinalEvaluationSchema>;
