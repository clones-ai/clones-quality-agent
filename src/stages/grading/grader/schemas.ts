import { z } from "zod";

export const CHUNK_EVALUATION_SCHEMA = {
    type: "object",
    properties: {
        summary: {
            type: "string",
            description: "A concise, one-paragraph summary of the user's actions in this chunk."
        }
    },
    required: ["summary"]
};

export const FINAL_EVALUATION_SCHEMA = {
    type: "object",
    properties: {
        summary: {
            type: "string",
            description: "A concise, one-paragraph summary of the entire session."
        },
        observations: {
            type: "string",
            description: "2-6 bullet points highlighting the most important user actions, successes, or failures."
        },
        reasoning: {
            type: "string",
            description: "A short (1-3 sentence) rationale for the overall score."
        },
        score: {
            type: "number",
            description: "The overall session score (0-100), based on the weighted average of the component scores. This will be ignored and recalculated deterministically."
        },
        confidence: {
            type: "number",
            description: "Your confidence in the evaluation (0-100), where 100 is absolute certainty."
        },
        outcomeAchievement: {
            type: "number",
            description: "Score (0-100) for outcome achievement."
        },
        processQuality: {
            type: "number",
            description: "Score (0-100) for process quality."
        },
        efficiency: {
            type: "number",
            description: "Score (0-100) for efficiency."
        },
        outcomeAchievementReasoning: {
            type: "string",
            description: "Justification for the outcome achievement score."
        },
        processQualityReasoning: {
            type: "string",
            description: "Justification for the process quality score."
        },
        efficiencyReasoning: {
            type: "string",
            description: "Justification for the efficiency score."
        },
        confidenceReasoning: {
            type: "string",
            description: "Justification for the confidence score."
        }
    },
    required: [
        "summary",
        "observations",
        "reasoning",
        "score",
        "confidence",
        "outcomeAchievement",
        "processQuality",
        "efficiency",
        "outcomeAchievementReasoning",
        "processQualityReasoning",
        "efficiencyReasoning",
        "confidenceReasoning"
    ]
};

export const ChunkEvaluationSchema = z.object({
    summary: z.string().min(1, "Summary must not be empty")
});

export const FinalEvaluationSchema = z.object({
    summary: z.string().min(1, "Summary must not be empty"),
    observations: z.string()
        .min(1, "Observations must not be empty")
        .refine((obs) => {
            // More flexible validation for different model capabilities
            const trimmed = obs.trim();
            
            // Accept if it has basic content (at least 10 characters for meaningful observations)
            if (trimmed.length < 10) return false;
            
            // Check for structured content (bullet points, numbered lists, or line breaks)
            const lines = obs.split('\n').filter(line => line.trim().length > 0);
            const bulletPoints = obs.split(/[•\-\*]/).filter(point => point.trim().length > 0);
            const numberedItems = obs.split(/\d+[\.\)]/).filter(item => item.trim().length > 0);
            
            // More permissive: accept 1-10 lines/points (was 2-6)
            // Check that at least one format is valid, but enforce the 1-10 limit on ALL formats
            const linesValid = lines.length >= 1 && lines.length <= 10;
            const bulletsValid = bulletPoints.length >= 1 && bulletPoints.length <= 10;
            const numberedValid = numberedItems.length >= 1 && numberedItems.length <= 10;
            
            // If content is structured as lines, enforce line limit
            // If content is structured as bullets, enforce bullet limit  
            // If content is structured as numbered items, enforce numbered limit
            return linesValid && bulletsValid && numberedValid;
        }, "Observations must contain meaningful structured content (1-10 lines, bullet points, or numbered items)"),
    reasoning: z.string().min(1, "Reasoning must not be empty"),
    score: z.number().int().min(0).max(100),
    confidence: z.number().int().min(0).max(100),
    outcomeAchievement: z.number().int().min(0).max(100),
    processQuality: z.number().int().min(0).max(100),
    efficiency: z.number().int().min(0).max(100),
    outcomeAchievementReasoning: z.string().min(1, "Outcome achievement reasoning must not be empty"),
    processQualityReasoning: z.string().min(1, "Process quality reasoning must not be empty"),
    efficiencyReasoning: z.string().min(1, "Efficiency reasoning must not be empty"),
    confidenceReasoning: z.string().min(1, "Confidence reasoning must not be empty")
});

export type ChunkEvaluation = z.infer<typeof ChunkEvaluationSchema>;
export type FinalEvaluation = z.infer<typeof FinalEvaluationSchema>;
