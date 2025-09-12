import { EvaluationCriteria } from "./types";
import constants from './constants.json';

export const DEFAULT_CRITERIA: EvaluationCriteria = constants.DEFAULT_CRITERIA;

export const DEFAULT_MODEL = "gpt-4o-2024-08-06";
export const DEFAULT_TIMEOUT_MS = 60_000;
export const DEFAULT_MAX_RETRIES = 3;
export const DEFAULT_MAX_IMAGES = 3;
export const DEFAULT_MAX_TEXT_LEN = 3000;
export const DEFAULT_SEED = 42;
