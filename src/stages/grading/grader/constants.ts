import { EvaluationCriteria } from "./types";

export const DEFAULT_CRITERIA: EvaluationCriteria = {
    outcomeAchievement: { weight: 0.5 },
    processQuality: { weight: 0.3 },
    efficiency: { weight: 0.2 },
};

console.log('LOADED_CONSTANTS_FILE', import.meta?.url || __filename);
console.log('DEFAULT_CRITERIA_IN_FILE', DEFAULT_CRITERIA);

export const DEFAULT_MODEL = "gpt-4o";
export const DEFAULT_TIMEOUT_MS = 60_000;
export const DEFAULT_MAX_RETRIES = 3;
export const DEFAULT_MAX_IMAGES = 3;
export const DEFAULT_MAX_TEXT_LEN = 3000;
export const DEFAULT_SEED = 42;
