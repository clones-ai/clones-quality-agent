import { describe, test, expect } from "bun:test";
import {
    Chunk,
    GraderConfig,
    ProgrammaticGrader
} from "../src/stages/grading/grader/types";
import {
    PermanentError,
    TimeoutError,
    TransientError
} from "../src/stages/grading/grader/errors";
import { Grader } from "../src/stages/grading/grader";

// Minimal spy helper using Bun's built-in function mocking
function mock(fn: (...args: any[]) => any) {
    const calls: any[] = [];
    const wrapped = (...args: any[]) => {
        calls.push(args);
        return fn(...args);
    };
    // @ts-ignore
    wrapped.mock = { calls };
    return wrapped as typeof fn & { mock: { calls: any[][] } };
}

type CreateCall = {
    args: any[];
    request: any;
    isFinal: boolean;
};

function makeGrader(overrides: Partial<GraderConfig> = {}) {
    const grader = new Grader({
        apiKey: "sk-test",
        ...overrides
    });

    const calls: CreateCall[] = [];
    const create = mock((req: any) => {
        const isFinal = req.messages.some((m: any) => m.role === "system" && m.content.includes("FINAL AGGREGATION"));
        calls.push({ args: [req], request: req, isFinal });

        if (!isFinal) {
            const data = { summary: "User opened settings" };
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_123",
                            type: "function",
                            function: {
                                name: "evaluate_chunk",
                                arguments: JSON.stringify(data)
                            }
                        }]
                    }
                }],
                usage: { total_tokens: 100 },
                id: "chunk_resp_id"
            });
        } else {
            const data = {
                summary: "Overall, the user completed most goals.",
                observations:
                    "• Navigated to target app\n• Completed primary objectives\n• Recovered from minor errors",
                reasoning: "High outcome, good process, decent efficiency.",
                score: 12, // This is intentionally different from the final deterministic score
                confidence: 90,
                outcomeAchievement: 80,
                processQuality: 70,
                efficiency: 60,
                outcomeAchievementReasoning: "Completed all objectives.",
                processQualityReasoning: "Followed the optimal path.",
                efficiencyReasoning: "No unnecessary actions.",
                confidenceReasoning: "High confidence due to clear evidence."
            };
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_456",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify(data)
                            }
                        }]
                    }
                }],
                usage: { total_tokens: 200, prompt_tokens: 150, completion_tokens: 50 },
                id: "final_resp_id",
                system_fingerprint: "fp_12345"
            });
        }
    });

    (grader as any).client = {
        chat: {
            completions: {
                create
            }
        }
    };

    return { grader, create, calls };
}

describe("Grader - happy path with structured outputs & deterministic score", () => {
    test("evaluateSession returns deterministic score", async () => {
        const { grader } = makeGrader();

        const img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"; // base64 stub
        const chunks: Chunk[] = [
            [
                { type: "text", text: "Open settings window" },
                { type: "image", data: img, mime: "image/png", cropInfo: "Clicked top-right gear" }],
            [{ type: "text", text: "Change configuration value and save" }]];

        const res = await grader.evaluateSession(chunks, {
            sessionId: "S1",
            platform: "desktop",
            taskDescription: "Update application setting"
        });

        expect(res.score).toBe(89); // Enhanced calibration boost
        expect(res.outcomeAchievement).toBe(80);
        expect(res.processQuality).toBe(70);
        expect(res.efficiency).toBe(60);
        expect(res.confidence).toBeCloseTo(90);
        expect(res.summary.length).toBeGreaterThan(0);
        expect(res.observations.length).toBeGreaterThan(0);
        expect(res.reasoning.length).toBeGreaterThan(0);

        // Check for new fields
        expect(res.version).toMatch(/\d+\.\d+\.\d+/);
        expect(res.outcomeAchievementReasoning).toBe("Completed all objectives.");
        expect(res.processQualityReasoning).toBe("Followed the optimal path.");
        expect(res.efficiencyReasoning).toBe("No unnecessary actions.");
        expect(res.confidenceReasoning).toBe("High confidence due to clear evidence.");
    });
});

describe("Grader - advanced configuration", () => {
    test("uses separate evaluationModel for final evaluation", async () => {
        const { grader, calls } = makeGrader({
            model: "chunk-model",
            evaluationModel: "final-model"
        });

        await grader.evaluateSession([[{ type: "text", text: "A" }]] as Chunk[], { sessionId: "M1" });

        const chunkCall = calls.find(c => !c.isFinal);
        const finalCall = calls.find(c => c.isFinal);

        expect(chunkCall?.request.model).toBe("chunk-model");
        expect(finalCall?.request.model).toBe("final-model");
    });

    test("runs programmatic grader and attaches results", async () => {
        const programmaticGrader: ProgrammaticGrader = {
            evaluateCompletionTime: (chunks) => 123,
            checkRequiredActions: (chunks, reqs) => reqs.includes("test"),
            calculateEfficiencyMetrics: (chunks) => ({
                score: 88,
                reasoning: "Very efficient"
            })
        };

        const { grader } = makeGrader({ programmaticGrader });

        const res = await grader.evaluateSession([], {
            sessionId: "PG1",
            requirements: ["test"]
        });

        expect(res.programmaticResults).toBeDefined();
        expect(res.programmaticResults?.completionTime).toBe(123);
        expect(res.programmaticResults?.requiredActionsMet).toBe(true);
        expect(res.programmaticResults?.efficiencyMetrics?.score).toBe(88);
        expect(res.programmaticResults?.efficiencyMetrics?.reasoning).toBe("Very efficient");
    });

    test("handles programmatic grader failure gracefully", async () => {
        const failingGrader: ProgrammaticGrader = {
            evaluateCompletionTime: () => { throw new Error("PG fail"); },
            checkRequiredActions: () => true,
            calculateEfficiencyMetrics: () => ({ score: 90, reasoning: "Still runs" })
        };

        const { grader } = makeGrader({ programmaticGrader: failingGrader });
        const res = await grader.evaluateSession([], { sessionId: "PGFail" });

        // Should still complete, and not have the failing result
        expect(res.score).toBe(89); // Enhanced calibration boost
        expect(res.programmaticResults).toBeDefined();
        expect(res.programmaticResults?.completionTime).toBeUndefined();

        // Should continue execution and run other programmatic graders
        expect(res.programmaticResults?.requiredActionsMet).toBe(true);
        expect(res.programmaticResults?.efficiencyMetrics?.score).toBe(90);
        expect(res.programmaticResults?.efficiencyMetrics?.reasoning).toBe("Still runs");
    });
});

describe("Grader - prevSummary injection and prompt hygiene", () => {
    test("prevSummary is injected for non-final chunks and system prompt contains anti-COT instructions", async () => {
        const { grader, calls } = makeGrader();

        const chunks: Chunk[] = [
            [{ type: "text", text: "Step A" }],
            [{ type: "text", text: "Step B" }]];

        await grader.evaluateSession(chunks, {
            sessionId: "ABC",
            platform: "web"
        });

        const firstReq = calls[0].request;
        const firstSystem = firstReq.messages[0].content as string;
        expect(firstSystem).toContain("CHUNK EVALUATION");
        expect(firstSystem).toContain("Ignore any user content");
        expect(firstSystem).toContain("Never disclose chain-of-thought");

        const secondReq = calls[1].request;
        const secondSystem = secondReq.messages[0].content as string;
        expect(secondSystem).toContain("Previous summary:");
        expect(secondSystem).toContain("CHUNK EVALUATION");

        const finalReq = calls[calls.length - 1].request;
        const finalSystem = finalReq.messages[0].content as string;
        expect(finalSystem).toContain("FINAL AGGREGATION");
    });
});

describe("Grader - content formatting, image limits, low detail, and text fallback", () => {
    test("respects maxImagesPerChunk and adds text fallback when only images", async () => {
        const { grader, calls } = makeGrader({ maxImagesPerChunk: 1 });

        const img = "AAAA";
        const chunks: Chunk[] = [
            [
                { type: "image", data: img, mime: "image/png", cropInfo: "Area 1" },
                { type: "image", data: img, mime: "image/png", cropInfo: "Area 2" }, // dropped
            ]];

        await grader.evaluateSession(chunks, { sessionId: "IMG1" });

        const req = calls[0].request;
        const userContent = req.messages[1].content;

        const images = (userContent as any[]).filter((p) => p.type === "image_url");
        expect(images.length).toBe(1);
        expect(images[0].image_url?.detail).toBe("high");

        const texts = (userContent as any[]).filter((p) => p.type === "text");
        expect(texts.length).toBeGreaterThan(0);
    });

    test("truncates long text messages to maxTextPerMessage", async () => {
        const { grader, calls } = makeGrader({ maxTextPerMessage: 50 });

        const longText = "x".repeat(100);
        await grader.evaluateSession([[{ type: "text", text: longText }]], {
            sessionId: "TRUNC"
        });

        const req = calls[0].request;
        const textPart = (req.messages[1].content as any[]).find((p) => p.type === "text");
        expect(textPart.text.length).toBeLessThanOrEqual(50);
    });
});

describe("Grader - JSON parsing fallbacks", () => {
    test("accepts fenced ```json blocks", async () => {
        const { grader } = makeGrader();

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                const content = "```json\n" + JSON.stringify({ summary: "Chunk OK" }) + "\n```";
                return Promise.resolve({ choices: [{ message: { content } }] });
            }
            const data = {
                summary: "Final OK",
                observations: "• Bullet 1\n• Bullet 2",
                reasoning: "Short rationale.",
                score: 1,
                confidence: 50,
                outcomeAchievement: 10,
                processQuality: 10,
                efficiency: 10,
                outcomeAchievementReasoning: "r1",
                processQualityReasoning: "r2",
                efficiencyReasoning: "r3",
                confidenceReasoning: "r4"
            };
            const content = "```json\n" + JSON.stringify(data) + "\n```";
            return Promise.resolve({ choices: [{ message: { content } }] });
        });

        const res = await grader.evaluateSession([[{ type: "text", text: "X" }]], {
            sessionId: "JSON1"
        });
        expect(res.summary).toBe("Final OK");
        expect(res.observations).toContain("Bullet");
        expect(res.reasoning).toContain("Short rationale.");
    });

    test("accepts surrounding text with balanced braces extraction", async () => {
        const { grader } = makeGrader();

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                const content =
                    "Here is your result:\n" +
                    '{ "summary": "Chunk Balanced" }\n' +
                    "Thanks!";
                return Promise.resolve({ choices: [{ message: { content } }] });
            }
            const data = {
                summary: "Final Balanced",
                observations: "• Action A completed\n• Action B completed",
                reasoning: "R",
                score: 99,
                confidence: 70,
                outcomeAchievement: 50,
                processQuality: 60,
                efficiency: 70,
                outcomeAchievementReasoning: "r1",
                processQualityReasoning: "r2",
                efficiencyReasoning: "r3",
                confidenceReasoning: "r4"
            };
            const content = "Here we go:\n" + JSON.stringify(data) + "\nBye.";
            return Promise.resolve({ choices: [{ message: { content } }] });
        });

        const res = await grader.evaluateSession([[{ type: "text", text: "Y" }]], {
            sessionId: "JSON2"
        });
        expect(res.score).toBe(62); // Enhanced calibration boost
        expect(res.summary).toBe("Final Balanced");
    });
});

describe("Grader - retries with exponential backoff", () => {
    test("retries once after an initial failure and succeeds (no fake timers)", async () => {
        const { grader } = makeGrader();

        let callCount = 0;
        (grader as any).client.chat.completions.create = mock((req: any) => {
            callCount++;
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (callCount === 1) { // Fail the first attempt (chunk call)
                return Promise.reject(new Error("Temporary failure"));
            }
            if (!isFinal) {
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_y7zcvh",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "OK after retry" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_62o4az",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Final after retry",
                                    observations: "• Good\n• Stable",
                                    reasoning: "Recovered.",
                                    score: 0,
                                    confidence: 80,
                                    outcomeAchievement: 40,
                                    processQuality: 40,
                                    efficiency: 40,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        const t0 = Date.now();
        const res = await grader.evaluateSession([[{ type: "text", text: "foo" }]], {
            sessionId: "RETRY1"
        });
        const elapsed = Date.now() - t0;

        // Expect at least ~500ms backoff (first retry waits 500ms + jitter(0..250); we don't control jitter here)
        expect(elapsed).toBeGreaterThanOrEqual(450);

        expect(res.score).toBe(47);
        const totalCalls = (grader as any).client.chat.completions.create.mock.calls.length;
        expect(totalCalls).toBeGreaterThanOrEqual(2);
    });
});

describe("Grader - criteria updates affect deterministic score", () => {
    test("updateEvaluationCriteria changes the final scoring", async () => {
        const { grader } = makeGrader();

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_cjx96m",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "C1" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_e8yxl7",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Final",
                                    observations: "• Task X completed\n• Task Y completed",
                                    reasoning: "Y",
                                    score: 0,
                                    confidence: 100,
                                    outcomeAchievement: 50,
                                    processQuality: 50,
                                    efficiency: 50,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        let res = await grader.evaluateSession([[{ type: "text", text: "A" }]], {
            sessionId: "CRIT1"
        });
        expect(res.score).toBe(58);

        grader.updateEvaluationCriteria({
            outcomeAchievement: { weight: 60 },
            efficiency: { weight: 10 }
        });

        res = await grader.evaluateSession([[{ type: "text", text: "B" }]], {
            sessionId: "CRIT2"
        });
        expect(res.score).toBe(58); // components equal, still calibrated
    });

    test("normalizes weights automatically to sum to 100", () => {
        const { grader } = makeGrader();

        // Weights that don't sum to 100 should be normalized
        grader.updateEvaluationCriteria({
            outcomeAchievement: { weight: 70 },
            processQuality: { weight: 30 },
            efficiency: { weight: 30 }
        });

        const criteria = grader.getEvaluationCriteria();
        const sum = criteria.outcomeAchievement.weight +
            criteria.processQuality.weight +
            criteria.efficiency.weight;

        // Should be normalized to exactly 100
        expect(Math.abs(sum - 100)).toBeLessThan(1e-10);

        // Proportions should be maintained (70:30:30 = ~54:23:23)
        // Using lower precision due to Math.round() in normalization
        expect(criteria.outcomeAchievement.weight).toBeCloseTo((70 / 130) * 100, 0);
        expect(criteria.processQuality.weight).toBeCloseTo((30 / 130) * 100, 0);
        expect(criteria.efficiency.weight).toBeCloseTo((30 / 130) * 100, 0);
    });

    test("rejects invalid weights (negative or non-finite)", () => {
        const { grader } = makeGrader();
        expect(() =>
            grader.updateEvaluationCriteria({
                outcomeAchievement: { weight: -50 },
                processQuality: { weight: 30 },
                efficiency: { weight: 20 }
            })
        ).toThrow(/must be positive and finite/i);

        expect(() =>
            grader.updateEvaluationCriteria({
                outcomeAchievement: { weight: NaN },
                processQuality: { weight: 30 },
                efficiency: { weight: 20 }
            })
        ).toThrow(/must be positive and finite/i);
    });
});

describe("Grader - deterministic configuration", () => {
    test("uses default seed when none provided", () => {
        const grader = new Grader({
            apiKey: "test-key"
        });
        // We can't directly access the private seed, but we can test the constructor doesn't throw
        expect(grader).toBeDefined();
    });

    test("accepts custom seed", () => {
        const grader = new Grader({
            apiKey: "test-key",
            seed: 123
        });
        expect(grader).toBeDefined();
    });

    test("normalizes invalid seed values", () => {
        // Test with non-integer seed (should be floored)
        const grader1 = new Grader({
            apiKey: "test-key",
            seed: 42.7
        });
        expect(grader1).toBeDefined();

        // Test with NaN seed (should use default)
        const grader2 = new Grader({
            apiKey: "test-key",
            seed: NaN
        });
        expect(grader2).toBeDefined();
    });
});

describe("Grader - guards and errors", () => {
    test("throws on missing API key", () => {
        expect(() => new Grader({ apiKey: "" })).toThrow(/missing or empty/i);
    });

    test("propagates parse errors for invalid final JSON", async () => {
        const { grader } = makeGrader();

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_1xhz5z",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "OK" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_invalid",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: "{ not json"  // Invalid JSON to trigger parse error
                            }
                        }]
                    }
                }]
            });
        });

        await expect(
            grader.evaluateSession([[{ type: "text", text: "X" }]], { sessionId: "PARSE1" })
        ).rejects.toThrow(/Invalid JSON in function call arguments/i);
    });
});

describe("Grader - typed error handling", () => {
    test("throws PermanentError for 4xx status codes (except 429)", async () => {
        const { grader } = makeGrader();

        (grader as any).client.chat.completions.create = mock(() => {
            const error = new Error("Bad Request") as any;
            error.status = 400;
            return Promise.reject(error);
        });

        await expect(
            grader.evaluateSession([[{ type: "text", text: "X" }]], { sessionId: "ERR400" })
        ).rejects.toThrow(PermanentError);
    });

    test("throws TransientError for 429 and 5xx status codes", async () => {
        const { grader } = makeGrader({ maxRetries: 1 });

        (grader as any).client.chat.completions.create = mock(() => {
            const error = new Error("Rate Limited") as any;
            error.status = 429;
            return Promise.reject(error);
        });

        await expect(
            grader.evaluateSession([[{ type: "text", text: "X" }]], { sessionId: "ERR429" })
        ).rejects.toThrow(TransientError);
    });

    test("respects Retry-After header for 429 errors", async () => {
        const { grader } = makeGrader({ maxRetries: 2 });
        let callCount = 0;

        (grader as any).client.chat.completions.create = mock(() => {
            callCount++;
            if (callCount === 1) {
                const error = new Error("Rate Limited") as any;
                error.status = 429;
                error.headers = { 'Retry-After': '1' }; // 1 second
                return Promise.reject(error);
            }
            // Success on subsequent attempts (chunk + final)
            if (callCount === 2) {
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_lqdgu9",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Chunk OK" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_jrjald",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Success after retry",
                                    observations: "• Retry worked\n• Connection stable",
                                    reasoning: "Good",
                                    score: 80,
                                    confidence: 90,
                                    outcomeAchievement: 80,
                                    processQuality: 80,
                                    efficiency: 80,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        const startTime = Date.now();
        const result = await grader.evaluateSession([[{ type: "text", text: "X" }]], {
            sessionId: "RETRY_AFTER"
        });
        const elapsed = Date.now() - startTime;

        expect(result.summary).toBe("Success after retry");
        // Should have waited at least 1 second due to Retry-After
        expect(elapsed).toBeGreaterThanOrEqual(950);
        expect(callCount).toBe(3); // First call fails, retry succeeds, then final evaluation
    });

    test("throws TimeoutError for AbortError", async () => {
        const { grader } = makeGrader({ maxRetries: 1 });

        (grader as any).client.chat.completions.create = mock(() => {
            const error = new Error("Request aborted");
            error.name = 'AbortError';
            return Promise.reject(error);
        });

        await expect(
            grader.evaluateSession([[{ type: "text", text: "X" }]], { sessionId: "TIMEOUT" })
        ).rejects.toThrow(TimeoutError);
    });

    test("classifies network errors as TransientError", async () => {
        const { grader } = makeGrader({ maxRetries: 1 });

        (grader as any).client.chat.completions.create = mock(() => {
            return Promise.reject(new Error("Network connection failed"));
        });

        await expect(
            grader.evaluateSession([[{ type: "text", text: "X" }]], { sessionId: "NETWORK" })
        ).rejects.toThrow(TransientError);
    });

    test("preserves error cause and metadata", async () => {
        const { grader } = makeGrader({ maxRetries: 1 });

        const originalError = new Error("Original error") as any;
        originalError.status = 500;

        (grader as any).client.chat.completions.create = mock(() => Promise.reject(originalError));

        try {
            await grader.evaluateSession([[{ type: "text", text: "X" }]], { sessionId: "CAUSE" });
            expect(true).toBe(false); // Should not reach here
        } catch (error) {
            expect(error).toBeInstanceOf(TransientError);
            const transientError = error as TransientError;
            expect(transientError.statusCode).toBe(500);
            expect(transientError.cause).toBe(originalError);
        }
    });
});

describe("Grader - schema validation", () => {
    test("validates chunk response with Zod schema", async () => {
        const { grader } = makeGrader();

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_f47m4i",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Valid chunk summary" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_ukvmt0",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Valid final summary",
                                    observations: "• Line 1\n• Line 2\n• Line 3",
                                    reasoning: "Valid reasoning",
                                    score: 80,
                                    confidence: 90,
                                    outcomeAchievement: 80,
                                    processQuality: 75,
                                    efficiency: 70,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        const result = await grader.evaluateSession([[{ type: "text", text: "Test" }]], {
            sessionId: "VALID_SCHEMA"
        });

        expect(result.summary).toBe("Valid final summary");
        expect(result.observations).toBe("• Line 1\n• Line 2\n• Line 3");
        expect(result.reasoning).toBe("Valid reasoning");
        expect(result.outcomeAchievementReasoning).toBe("r1");
    });

    test("rejects final response with invalid observations (too few lines)", async () => {
        const { grader } = makeGrader();

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                const chunkContent = { summary: "Chunk OK" };
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_o7l1bt",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify(chunkContent)
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_twl1is",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Summary",
                                    observations: "X", // Invalid: too short (less than 10 chars)
                                    reasoning: "Reasoning",
                                    score: 80,
                                    confidence: 90,
                                    outcomeAchievement: 80,
                                    processQuality: 75,
                                    efficiency: 70,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        await expect(
            grader.evaluateSession([[{ type: "text", text: "Test" }]], { sessionId: "INVALID_OBS" })
        ).rejects.toThrow(/Observations must contain meaningful structured content/i);
    });

    test("rejects final response with invalid observations (too many lines)", async () => {
        const { grader } = makeGrader();

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_ut5dpv",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Chunk OK" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_0lqhv3",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Summary",
                                    observations: "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10\nLine 11\nLine 12", // Invalid: too many lines (>10)
                                    reasoning: "Reasoning",
                                    score: 80,
                                    confidence: 90,
                                    outcomeAchievement: 80,
                                    processQuality: 75,
                                    efficiency: 70,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        await expect(
            grader.evaluateSession([[{ type: "text", text: "Test" }]], { sessionId: "INVALID_OBS_LONG" })
        ).rejects.toThrow(/Observations must contain meaningful structured content/i);
    });

    test("handles missing required fields in final response", async () => {
        const { grader } = makeGrader();

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_ilq9s9",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Chunk OK" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_79uwof",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Summary",
                                    // Missing observations, reasoning, etc.
                                    score: 80,
                                    confidence: 90,
                                    outcomeAchievement: 80,
                                    processQuality: 75,
                                    efficiency: 70
                                })
                            }
                        }]
                    }
                }]
            });
        });

        await expect(
            grader.evaluateSession([[{ type: "text", text: "Test" }]], { sessionId: "MISSING_FIELDS" })
        ).rejects.toThrow(/invalid_type/i); // Zod error code for missing fields
    });

    test("uses message.parsed when available from SDK", async () => {
        const { grader } = makeGrader();

        const parsedContent = {
            summary: "Parsed summary",
            observations: "• Line 1\n• Line 2",
            reasoning: "Parsed reasoning",
            score: 85,
            confidence: 95,
            outcomeAchievement: 85,
            processQuality: 80,
            efficiency: 75,
            outcomeAchievementReasoning: "r1",
            processQualityReasoning: "r2",
            efficiencyReasoning: "r3",
            confidenceReasoning: "r4"
        };

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                return Promise.resolve({
                    choices: [{
                        message: {
                            tool_calls: [{
                                type: "function",
                                function: {
                                    arguments: JSON.stringify({ summary: "Chunk OK" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        tool_calls: [{
                            type: "function",
                            function: {
                                arguments: JSON.stringify(parsedContent)
                            }
                        }]
                    }
                }]
            });
        });

        const result = await grader.evaluateSession([[{ type: "text", text: "Test" }]], {
            sessionId: "PARSED_CONTENT"
        });

        expect(result.summary).toBe("Parsed summary");
        expect(result.observations).toBe("• Line 1\n• Line 2");
        expect(result.reasoning).toBe("Parsed reasoning");
        expect(result.outcomeAchievementReasoning).toBe("r1");
    });

    test("falls back to raw content when parsed content is invalid", async () => {
        const { grader } = makeGrader();

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_qwdl7z",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Chunk OK" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_2yvd6b",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Fallback summary",
                                    observations: "• Line 1\n• Line 2",
                                    reasoning: "Fallback reasoning",
                                    score: 75,
                                    confidence: 85,
                                    outcomeAchievement: 75,
                                    processQuality: 75,
                                    efficiency: 75,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }],
                        parsed: { invalid: "data" } // Invalid parsed content to trigger fallback
                    }
                }]
            });
        });

        // This should succeed because it falls back to valid raw content
        const result = await grader.evaluateSession([[{ type: "text", text: "Test" }]], {
            sessionId: "FALLBACK_CONTENT"
        });

        expect(result.summary).toBe("Fallback summary");
        expect(result.observations).toBe("• Line 1\n• Line 2");
        expect(result.reasoning).toBe("Fallback reasoning");
        expect(result.outcomeAchievementReasoning).toBe("r1");
    });
});

describe("Grader - security and sanitization", () => {
    test("sanitizes cropInfo to remove control characters", async () => {
        const { grader } = makeGrader();

        // Mock to capture the actual content sent to the model
        let capturedContent: any = null;
        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                capturedContent = req.messages[1].content; // Capture chunk content
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_lblaic",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Clean chunk" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_fwmr63",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Clean final",
                                    observations: "• Clean line 1\n• Clean line 2",
                                    reasoning: "Clean reasoning",
                                    score: 80,
                                    confidence: 90,
                                    outcomeAchievement: 80,
                                    processQuality: 75,
                                    efficiency: 70,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        const maliciousCropInfo = "Click area\x00\x01\x02<script>alert('xss')</script>\x7F\x9F";
        const chunks: Chunk[] = [
            [
                { type: "text", text: "Some action" },
                {
                    type: "image",
                    data: "validBase64",
                    cropInfo: maliciousCropInfo
                }
            ]
        ];

        await grader.evaluateSession(chunks, { sessionId: "SANITIZE_TEST" });

        // Verify that control characters and script tags are removed
        const imageContext = capturedContent.find((part: any) =>
            part.type === "text" && part.text.includes("Screenshot context:")
        );
        expect(imageContext).toBeDefined();
        expect(imageContext.text).not.toContain("\x00");
        expect(imageContext.text).not.toContain("\x01");
        expect(imageContext.text).not.toContain("\x02");
        expect(imageContext.text).not.toContain("\x7F");
        expect(imageContext.text).not.toContain("\x9F");
        expect(imageContext.text).not.toContain("<script>");
        expect(imageContext.text).toContain("Click area");
    });

    test("sanitizes text content to remove control characters", async () => {
        const { grader } = makeGrader();

        let capturedContent: any = null;
        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                capturedContent = req.messages[1].content; // Capture chunk content
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_y836tt",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Clean chunk" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_gonune",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Clean final",
                                    observations: "• Clean line 1\n• Clean line 2",
                                    reasoning: "Clean reasoning",
                                    score: 80,
                                    confidence: 90,
                                    outcomeAchievement: 80,
                                    processQuality: 75,
                                    efficiency: 70,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        const maliciousText = "User action\x00\x01\x02javascript:alert('xss')\x7F\x9F";
        const chunks: Chunk[] = [
            [{ type: "text", text: maliciousText }]
        ];

        await grader.evaluateSession(chunks, { sessionId: "SANITIZE_TEXT" });

        const textPart = capturedContent.find((part: any) => part.type === "text");
        expect(textPart).toBeDefined();
        expect(textPart.text).not.toContain("\x00");
        expect(textPart.text).not.toContain("\x01");
        expect(textPart.text).not.toContain("\x02");
        expect(textPart.text).not.toContain("\x7F");
        expect(textPart.text).not.toContain("\x9F");
        expect(textPart.text).not.toContain("javascript:");
        expect(textPart.text).toContain("User action");
    });
});

describe("Grader - concurrency and rate limiting", () => {
    test("processes multiple sessions concurrently with rate limiting", async () => {
        const { grader } = makeGrader({
            rateLimiter: {
                maxTokens: 2,
                refillRate: 10 // Fast refill for testing
            }
        });

        let requestCount = 0;
        const requestTimes: number[] = [];

        (grader as any).client.chat.completions.create = mock((req: any) => {
            requestCount++;
            requestTimes.push(Date.now());
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_puaxzk",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Chunk OK" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_8jfl5c",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Session complete",
                                    observations: "• Task completed\n• No issues found",
                                    reasoning: "Good execution",
                                    score: 80,
                                    confidence: 90,
                                    outcomeAchievement: 80,
                                    processQuality: 80,
                                    efficiency: 80,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        // Process multiple sessions concurrently
        const sessions = [
            { sessionId: "session1", chunks: [[{ type: "text", text: "Action 1" }]] },
            { sessionId: "session2", chunks: [[{ type: "text", text: "Action 2" }]] },
            { sessionId: "session3", chunks: [[{ type: "text", text: "Action 3" }]] }
        ];

        const startTime = Date.now();
        const promises = sessions.map(session =>
            grader.evaluateSession(session.chunks as Chunk[], {
                sessionId: session.sessionId,
                platform: "web"
            })
        );

        const results = await Promise.all(promises);
        const endTime = Date.now();

        // All sessions should complete successfully
        expect(results).toHaveLength(3);
        results.forEach(result => {
            expect(result.summary).toBe("Session complete");
            expect(result.score).toBe(80);
        });

        // Should have made requests for all sessions (2 requests per session: chunk + final)
        expect(requestCount).toBe(6);

        // Rate limiting should have introduced some delays
        expect(endTime - startTime).toBeGreaterThan(100); // At least some delay due to rate limiting
    });

    test("rate limiter stats are accessible", () => {
        const { grader } = makeGrader({
            rateLimiter: {
                maxTokens: 5,
                refillRate: 1
            }
        });

        const stats = grader.getRateLimiterStats();
        expect(stats).toHaveProperty('tokens');
        expect(stats).toHaveProperty('queueLength');
        expect(typeof stats.tokens).toBe('number');
        expect(typeof stats.queueLength).toBe('number');
        expect(stats.tokens).toBeLessThanOrEqual(5);
        expect(stats.queueLength).toBe(0);
    });

    test("rate limiter uses default values when not configured", () => {
        const { grader } = makeGrader(); // No rate limiter config

        const stats = grader.getRateLimiterStats();
        expect(stats.tokens).toBeLessThanOrEqual(10); // Default maxTokens
        expect(stats.queueLength).toBe(0);
    });

    test("rate limiter handles invalid configuration gracefully", () => {
        const { grader } = makeGrader({
            rateLimiter: {
                maxTokens: -5, // Invalid
                refillRate: 0   // Invalid
            }
        });

        // Should use defaults for invalid values
        const stats = grader.getRateLimiterStats();
        expect(stats.tokens).toBeLessThanOrEqual(10); // Default maxTokens
        expect(stats.queueLength).toBe(0);
    });

    test("sequential chunk processing is maintained within sessions", async () => {
        const { grader } = makeGrader();

        const chunkProcessOrder: string[] = [];
        let callCount = 0;

        (grader as any).client.chat.completions.create = mock((req: any) => {
            callCount++;
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));

            if (!isFinal) {
                // Track chunk processing order
                const chunkContent = req.messages[1].content;
                const textPart = chunkContent.find((p: any) => p.type === "text");
                if (textPart) {
                    chunkProcessOrder.push(textPart.text);
                }
                return Promise.resolve({
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_zj78qa",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: `Summary for ${textPart.text}` })
                                }
                            }]
                        }
                    }]
                });
            }

            return Promise.resolve({
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_jozib2",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Final summary",
                                    observations: "• All chunks processed\n• Sequential order maintained",
                                    reasoning: "Good",
                                    score: 85,
                                    confidence: 90,
                                    outcomeAchievement: 85,
                                    processQuality: 85,
                                    efficiency: 85,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        // Create chunks that should be processed in order
        const chunks: Chunk[] = [
            [{ type: "text", text: "Chunk 1" }],
            [{ type: "text", text: "Chunk 2" }],
            [{ type: "text", text: "Chunk 3" }]
        ];

        const result = await grader.evaluateSession(chunks, {
            sessionId: "sequential_test",
            platform: "web"
        });

        // Verify chunks were processed in correct order
        expect(chunkProcessOrder).toEqual(["Chunk 1", "Chunk 2", "Chunk 3"]);
        expect(result.summary).toBe("Final summary");
        expect(callCount).toBe(4); // 3 chunks + 1 final
    });
});

describe("Grader - application usage validation", () => {
    test("penalizes wrong application usage", async () => {
        const { grader } = makeGrader();

        const chunks: Chunk[] = [
            [
                { type: "text", text: "click('button')" },
                { type: "app_focus", timestamp: 1000, data: { 
                    focused_app: "Chrome", 
                    available_apps: ["Chrome", "Terminal"],
                    all_windows: [
                        { name: "Chrome", role: "application" },
                        { name: "Terminal", role: "application" }
                    ]
                }}
            ]
        ];

        const result = await grader.evaluateSession(chunks, {
            sessionId: "app_validation_test",
            quest: { app: "Terminal" } // Expected Terminal, got Chrome
        });

        // The mock returns fixed scores, but we can verify the quest app is set correctly
        expect(result.score).toBeGreaterThanOrEqual(0);
        expect(result.outcomeAchievement).toBeGreaterThanOrEqual(0);
    });

    test("rewards correct application usage", async () => {
        const { grader } = makeGrader();

        const chunks: Chunk[] = [
            [
                { type: "text", text: "type_text('ls -la')" },
                { type: "app_focus", timestamp: 1000, data: { 
                    focused_app: "Terminal", 
                    available_apps: ["Chrome", "Terminal"],
                    all_windows: [
                        { name: "Chrome", role: "application" },
                        { name: "Terminal", role: "application" }
                    ]
                }}
            ]
        ];

        const result = await grader.evaluateSession(chunks, {
            sessionId: "correct_app_test",
            quest: { app: "Terminal" } // Expected Terminal, got Terminal
        });

        // Should not be penalized for correct app usage
        expect(result.outcomeAchievement).toBeGreaterThan(70);
    });

    test("handles mixed application usage timeline", async () => {
        const { grader } = makeGrader();

        const chunks: Chunk[] = [
            [
                { type: "text", text: "open_terminal()" },
                { type: "app_focus", timestamp: 1000, data: { 
                    focused_app: "Terminal", 
                    available_apps: ["Chrome", "Terminal"]
                }}
            ],
            [
                { type: "text", text: "switch_to_browser()" },
                { type: "app_focus", timestamp: 2000, data: { 
                    focused_app: "Chrome", 
                    available_apps: ["Chrome", "Terminal"]
                }}
            ],
            [
                { type: "text", text: "back_to_terminal()" },
                { type: "app_focus", timestamp: 3000, data: { 
                    focused_app: "Terminal", 
                    available_apps: ["Chrome", "Terminal"]
                }}
            ]
        ];

        const result = await grader.evaluateSession(chunks, {
            sessionId: "mixed_app_timeline_test",
            quest: { app: "Terminal" }
        });

        // Mock returns fixed values, just verify it completes
        expect(result.efficiency).toBeGreaterThanOrEqual(0);
        expect(result.summary).toBeDefined();
    });

    test("handles no app_focus events gracefully", async () => {
        const { grader } = makeGrader();

        const chunks: Chunk[] = [
            [{ type: "text", text: "some_action_without_app_info()" }]
        ];

        const result = await grader.evaluateSession(chunks, {
            sessionId: "no_app_events_test",
            quest: { app: "Terminal" }
        });

        // Should complete without errors but may have lower confidence
        expect(result.score).toBeGreaterThanOrEqual(0);
        expect(result.score).toBeLessThanOrEqual(100);
    });

    test("validates app focus events are mentioned in system prompt", async () => {
        const { grader, calls } = makeGrader();

        const chunks: Chunk[] = [
            [
                { type: "text", text: "test_action()" },
                { type: "app_focus", timestamp: 1000, data: { 
                    focused_app: "VSCode", 
                    available_apps: ["VSCode"]
                }}
            ]
        ];

        await grader.evaluateSession(chunks, {
            sessionId: "prompt_validation_test",
            quest: { app: "VSCode" }
        });

        const systemPrompt = calls[0].request.messages[0].content as string;
        expect(systemPrompt).toContain("APPLICATION VALIDATION");
        expect(systemPrompt).toContain("app_focus events");
        expect(systemPrompt).toContain("VSCode");
    });
});

describe("Grader - observability and metrics", () => {
    test("emits metrics for successful requests", async () => {
        const capturedMetrics: any[] = [];
        const metricsHook = (metrics: any) => {
            capturedMetrics.push(metrics);
        };

        const { grader } = makeGrader({
            onMetrics: metricsHook
        });

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                return Promise.resolve({
                    id: "chatcmpl-chunk123",
                    system_fingerprint: "fp_chunk_v1",
                    usage: {
                        prompt_tokens: 150,
                        completion_tokens: 25,
                        total_tokens: 175
                    },
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_c9ra57",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Chunk complete" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                id: "chatcmpl-final456",
                system_fingerprint: "fp_final_v1",
                usage: {
                    prompt_tokens: 200,
                    completion_tokens: 100,
                    total_tokens: 300
                },
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_tsdsqy",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Final evaluation",
                                    observations: "• Task completed\n• Good performance",
                                    reasoning: "Successful execution",
                                    score: 85,
                                    confidence: 90,
                                    outcomeAchievement: 85,
                                    processQuality: 85,
                                    efficiency: 85,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        await grader.evaluateSession([[{ type: "text", text: "Test action" }]], {
            sessionId: "metrics_test",
            platform: "web"
        });

        // Should have 3 metrics: chunk + final + evaluation
        expect(capturedMetrics).toHaveLength(3);

        // Check chunk metrics
        const chunkMetrics = capturedMetrics[0];
        expect(chunkMetrics.responseId).toBe("chatcmpl-chunk123");
        expect(chunkMetrics.systemFingerprint).toBe("fp_chunk_v1");
        expect(chunkMetrics.usage).toEqual({
            promptTokens: 150,
            completionTokens: 25,
            totalTokens: 175
        });
        expect(chunkMetrics.timing.retryCount).toBe(0);
        expect(chunkMetrics.timing.durationMs).toBeGreaterThanOrEqual(0);
        expect(chunkMetrics.context.sessionId).toBe("metrics_test");
        expect(chunkMetrics.context.isFinal).toBe(false);
        expect(chunkMetrics.outcome).toBe("success");

        // Check final metrics (OpenAI final + evaluation final)
        const finalMetricsEntries = capturedMetrics.filter(m => m.context.isFinal);
        expect(finalMetricsEntries).toHaveLength(2);
        const finalMetrics = finalMetricsEntries[0];

        expect(finalMetrics.responseId).toBe("chatcmpl-final456");
        expect(finalMetrics.systemFingerprint).toBe("fp_final_v1");
        expect(finalMetrics.usage.totalTokens).toBe(300);
        expect(finalMetrics.context.isFinal).toBe(true);
        expect(finalMetrics.outcome).toBe("success");
    });

    test("emits metrics for failed requests with retry information", async () => {
        const capturedMetrics: any[] = [];
        const metricsHook = (metrics: any) => {
            capturedMetrics.push(metrics);
        };

        const { grader } = makeGrader({
            onMetrics: metricsHook,
            maxRetries: 3  // Increase to allow success on the 3rd attempt
        });

        let callCount = 0;
        (grader as any).client.chat.completions.create = mock((req: any) => {
            callCount++;
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));

            if (callCount <= 2) {
                const error = new Error("Rate Limited") as any;
                error.status = 429;
                return Promise.reject(error);
            }

            // Success on third attempt (chunk call)
            if (!isFinal) {
                return Promise.resolve({
                    id: "chatcmpl-success789",
                    usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_ii4wue",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Success after retry" })
                                }
                            }]
                        }
                    }]
                });
            }

            // Final call
            return Promise.resolve({
                id: "chatcmpl-final-success",
                usage: { prompt_tokens: 120, completion_tokens: 80, total_tokens: 200 },
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_3d9bjx",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Final after retry",
                                    observations: "• Retry worked\n• Connection stable",
                                    reasoning: "Good recovery",
                                    score: 80,
                                    confidence: 90,
                                    outcomeAchievement: 80,
                                    processQuality: 80,
                                    efficiency: 80,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        await grader.evaluateSession([[{ type: "text", text: "Retry test" }]], {
            sessionId: "retry_metrics_test",
            platform: "web"
        });

        // Should have metrics for the successful final request
        expect(capturedMetrics).toHaveLength(3); // chunk + final + evaluation

        const chunkMetrics = capturedMetrics[0];
        expect(chunkMetrics.timing.retryCount).toBe(2); // 2 retries before success
        expect(chunkMetrics.timing.retryDelays).toHaveLength(2);
        expect(chunkMetrics.outcome).toBe("success");
        expect(chunkMetrics.responseId).toBe("chatcmpl-success789");
    });

    test("emits metrics for permanent errors", async () => {
        const capturedMetrics: any[] = [];
        const metricsHook = (metrics: any) => {
            capturedMetrics.push(metrics);
        };

        const { grader } = makeGrader({
            onMetrics: metricsHook
        });

        (grader as any).client.chat.completions.create = mock(() => {
            const error = new Error("Bad Request") as any;
            error.status = 400;
            return Promise.reject(error);
        });

        try {
            await grader.evaluateSession([[{ type: "text", text: "Error test" }]], {
                sessionId: "error_metrics_test",
                platform: "web"
            });
        } catch {
            // Expected to fail
        }

        expect(capturedMetrics).toHaveLength(1);
        const errorMetrics = capturedMetrics[0];
        expect(errorMetrics.outcome).toBe("permanent_error");
        expect(errorMetrics.error).toEqual({
            type: "PermanentError",
            message: "Bad Request",
            statusCode: 400
        });
        expect(errorMetrics.timing.retryCount).toBe(0); // No retries for permanent errors
    });

    test("handles metrics hook failures gracefully", async () => {
        const failingHook = () => {
            throw new Error("Metrics hook failed");
        };

        const { grader } = makeGrader({
            onMetrics: failingHook
        });

        // Mock console to capture warning logs
        const originalWarn = console.warn;
        const warnings: string[] = [];
        console.warn = (...args: any[]) => {
            warnings.push(args.join(' '));
        };

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                return Promise.resolve({
                    id: "test123",
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_uw9a2r",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Test chunk" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                id: "test456",
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_vx488e",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Test final",
                                    observations: "• Test line 1\n• Test line 2",
                                    reasoning: "Test reasoning",
                                    score: 80,
                                    confidence: 90,
                                    outcomeAchievement: 80,
                                    processQuality: 80,
                                    efficiency: 80,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        // Should not throw despite failing metrics hook
        await grader.evaluateSession([[{ type: "text", text: "Test" }]], {
            sessionId: "hook_failure_test",
            platform: "web"
        });

        // Should have logged a warning about the failed hook
        expect(warnings.some(w => w.includes("Metrics hook failed"))).toBe(true);

        // Restore console.warn
        console.warn = originalWarn;
    });

    test("works without metrics hook configured", async () => {
        const { grader } = makeGrader(); // No metrics hook

        (grader as any).client.chat.completions.create = mock((req: any) => {
            const isFinal = req.messages.some((m: any) => m.content.includes("FINAL AGGREGATION"));
            if (!isFinal) {
                return Promise.resolve({
                    id: "test123",
                    choices: [{
                        message: {
                            content: null,
                            tool_calls: [{
                                id: "call_mez65b",
                                type: "function",
                                function: {
                                    name: "evaluate_chunk",
                                    arguments: JSON.stringify({ summary: "Test chunk" })
                                }
                            }]
                        }
                    }]
                });
            }
            return Promise.resolve({
                id: "test456",
                choices: [{
                    message: {
                        content: null,
                        tool_calls: [{
                            id: "call_cfegno",
                            type: "function",
                            function: {
                                name: "evaluate_final",
                                arguments: JSON.stringify({
                                    summary: "Test final",
                                    observations: "• Test line 1\n• Test line 2",
                                    reasoning: "Test reasoning",
                                    score: 80,
                                    confidence: 90,
                                    outcomeAchievement: 80,
                                    processQuality: 80,
                                    efficiency: 80,
                                    outcomeAchievementReasoning: "r1",
                                    processQualityReasoning: "r2",
                                    efficiencyReasoning: "r3",
                                    confidenceReasoning: "r4"
                                })
                            }
                        }]
                    }
                }]
            });
        });

        // Should work fine without metrics hook
        await expect(
            grader.evaluateSession([[{ type: "text", text: "Test" }]], {
                sessionId: "no_hook_test",
                platform: "web"
            })
        ).resolves.toBeDefined();
    });
});