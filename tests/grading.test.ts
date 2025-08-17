// tests/grading.test.ts
import { describe, test, expect, beforeEach } from "bun:test";
// 🔧 adjust the import path to your project layout
import { Grader, type Chunk, type GraderConfig } from "../src/stages/grading/grader";

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
    isFinal: boolean;
    request: any;
};

function makeGrader(overrides: Partial<GraderConfig> = {}) {
    const grader = new Grader({
        apiKey: "sk-test",
        ...overrides,
    });

    const calls: CreateCall[] = [];
    const create = mock((req: any) => {
        const isFinal =
            req?.response_format?.json_schema?.name === "final_evaluation";
        calls.push({ args: [req], isFinal, request: req });

        if (!isFinal) {
            const content = JSON.stringify({ summary: "User opened settings" });
            return Promise.resolve({
                choices: [{ message: { content } }],
            });
        } else {
            const content = JSON.stringify({
                summary: "Overall, the user completed most goals.",
                observations:
                    "• Navigated to target app\n• Completed primary objectives\n• Recovered from minor errors",
                reasoning: "High outcome, good process, decent efficiency.",
                score: 12, // ignored (deterministic score used)
                confidence: 0.9,
                outcomeAchievement: 80,
                processQuality: 70,
                efficiency: 60,
            });
            return Promise.resolve({
                choices: [{ message: { content } }],
            });
        }
    });

    (grader as any).client = {
        chat: {
            completions: {
                create,
            },
        },
    };

    return { grader, create, calls };
}

describe("Grader - happy path with structured outputs & deterministic score", () => {
    test("evaluateSession returns deterministic score and uses JSON Schema response_format", async () => {
        const { grader, calls } = makeGrader();

        const img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"; // base64 stub
        const chunks: Chunk[] = [
            [
                { type: "text", text: "Open settings window" },
                { type: "image", data: img, mime: "image/png", cropInfo: "Clicked top-right gear" },
            ],
            [{ type: "text", text: "Change configuration value and save" }],
        ];

        const res = await grader.evaluateSession(chunks, {
            sessionId: "S1",
            platform: "desktop",
            taskDescription: "Update application setting",
        });

        expect(res.score).toBe(73); // 0.5*80 + 0.3*70 + 0.2*60
        expect(res.outcomeAchievement).toBe(80);
        expect(res.processQuality).toBe(70);
        expect(res.efficiency).toBe(60);
        expect(res.confidence).toBeCloseTo(0.9);
        expect(res.summary.length).toBeGreaterThan(0);
        expect(res.observations.length).toBeGreaterThan(0);
        expect(res.reasoning.length).toBeGreaterThan(0);

        const usedSchemas = calls.map((c) => c.request?.response_format?.json_schema?.name);
        expect(usedSchemas).toContain("chunk_evaluation");
        expect(usedSchemas).toContain("final_evaluation");
    });
});

describe("Grader - prevSummary injection and prompt hygiene", () => {
    test("prevSummary is injected for non-final chunks and system prompt contains anti-COT instructions", async () => {
        const { grader, calls } = makeGrader();

        const chunks: Chunk[] = [
            [{ type: "text", text: "Step A" }],
            [{ type: "text", text: "Step B" }],
        ];

        await grader.evaluateSession(chunks, {
            sessionId: "ABC",
            platform: "web",
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
            ],
        ];

        await grader.evaluateSession(chunks, { sessionId: "IMG1" });

        const req = calls[0].request;
        const userContent = req.messages[1].content;

        const images = (userContent as any[]).filter((p) => p.type === "image_url");
        expect(images.length).toBe(1);
        expect(images[0].image_url?.detail).toBe("low");

        const texts = (userContent as any[]).filter((p) => p.type === "text");
        expect(texts.length).toBeGreaterThan(0);
    });

    test("truncates long text messages to maxTextPerMessage", async () => {
        const { grader, calls } = makeGrader({ maxTextPerMessage: 50 });

        const longText = "x".repeat(100);
        await grader.evaluateSession([[{ type: "text", text: longText }]], {
            sessionId: "TRUNC",
        });

        const req = calls[0].request;
        const textPart = (req.messages[1].content as any[]).find((p) => p.type === "text");
        expect(textPart.text.length).toBeLessThanOrEqual(50);
    });
});

describe("Grader - JSON parsing fallbacks", () => {
    test("accepts fenced ```json blocks", async () => {
        const { grader } = makeGrader();

        (grader as any).client = {
            chat: {
                completions: {
                    create: mock((req: any) => {
                        const isFinal =
                            req?.response_format?.json_schema?.name === "final_evaluation";
                        if (!isFinal) {
                            const content = "```json\n" + JSON.stringify({ summary: "Chunk OK" }) + "\n```";
                            return Promise.resolve({ choices: [{ message: { content } }] });
                        }
                        const content =
                            "```json\n" +
                            JSON.stringify({
                                summary: "Final OK",
                                observations: "• Bullet 1\n• Bullet 2",
                                reasoning: "Short rationale.",
                                score: 1,
                                confidence: 0.5,
                                outcomeAchievement: 10,
                                processQuality: 10,
                                efficiency: 10,
                            }) +
                            "\n```";
                        return Promise.resolve({ choices: [{ message: { content } }] });
                    }),
                },
            },
        };

        const res = await grader.evaluateSession([[{ type: "text", text: "X" }]], {
            sessionId: "JSON1",
        });
        expect(res.summary).toBe("Final OK");
        expect(res.observations).toContain("Bullet");
        expect(res.reasoning).toContain("Short rationale.");
    });

    test("accepts surrounding text with balanced braces extraction", async () => {
        const { grader } = makeGrader();

        (grader as any).client = {
            chat: {
                completions: {
                    create: mock((req: any) => {
                        const isFinal =
                            req?.response_format?.json_schema?.name === "final_evaluation";
                        if (!isFinal) {
                            const content =
                                "Here is your result:\n" +
                                '{ "summary": "Chunk Balanced" }\n' +
                                "Thanks!";
                            return Promise.resolve({ choices: [{ message: { content } }] });
                        }
                        const content =
                            "Here we go:\n" +
                            "{ " +
                            '"summary":"Final Balanced",' +
                            '"observations":"• A\\n• B",' +
                            '"reasoning":"R",' +
                            '"score":99,' +
                            '"confidence":0.7,' +
                            '"outcomeAchievement":50,' +
                            '"processQuality":60,' +
                            '"efficiency":70' +
                            "}\nBye.";
                        return Promise.resolve({ choices: [{ message: { content } }] });
                    }),
                },
            },
        };

        const res = await grader.evaluateSession([[{ type: "text", text: "Y" }]], {
            sessionId: "JSON2",
        });
        expect(res.score).toBe(57); // 0.5*50 + 0.3*60 + 0.2*70
        expect(res.summary).toBe("Final Balanced");
    });
});

describe("Grader - retries with exponential backoff", () => {
    test("retries once after an initial failure and succeeds (no fake timers)", async () => {
        const { grader } = makeGrader();

        let callCount = 0;
        (grader as any).client = {
            chat: {
                completions: {
                    create: mock((req: any) => {
                        callCount++;
                        const isFinal =
                            req?.response_format?.json_schema?.name === "final_evaluation";
                        if (callCount === 1) {
                            // Fail the first attempt (chunk call)
                            return Promise.reject(new Error("Temporary failure"));
                        }
                        if (!isFinal) {
                            return Promise.resolve({
                                choices: [{ message: { content: JSON.stringify({ summary: "OK after retry" }) } }],
                            });
                        }
                        return Promise.resolve({
                            choices: [{
                                message: {
                                    content: JSON.stringify({
                                        summary: "Final after retry",
                                        observations: "• Good\n• Stable",
                                        reasoning: "Recovered.",
                                        score: 0,
                                        confidence: 0.8,
                                        outcomeAchievement: 40,
                                        processQuality: 40,
                                        efficiency: 40,
                                    }),
                                },
                            }],
                        });
                    }),
                },
            },
        };

        const t0 = Date.now();
        const res = await grader.evaluateSession([[{ type: "text", text: "foo" }]], {
            sessionId: "RETRY1",
        });
        const elapsed = Date.now() - t0;

        // Expect at least ~500ms backoff (first retry waits 500ms + jitter(0..250); we don't control jitter here)
        expect(elapsed).toBeGreaterThanOrEqual(450);

        expect(res.score).toBe(40);
        const totalCalls = (grader as any).client.chat.completions.create.mock.calls.length;
        expect(totalCalls).toBeGreaterThanOrEqual(2);
    });
});

describe("Grader - criteria updates affect deterministic score", () => {
    test("updateEvaluationCriteria changes the final scoring", async () => {
        const { grader } = makeGrader();

        (grader as any).client = {
            chat: {
                completions: {
                    create: mock((req: any) => {
                        const isFinal =
                            req?.response_format?.json_schema?.name === "final_evaluation";
                        if (!isFinal) {
                            return Promise.resolve({
                                choices: [{ message: { content: JSON.stringify({ summary: "C1" }) } }],
                            });
                        }
                        return Promise.resolve({
                            choices: [{
                                message: {
                                    content: JSON.stringify({
                                        summary: "Final",
                                        observations: "• X",
                                        reasoning: "Y",
                                        score: 0,
                                        confidence: 1,
                                        outcomeAchievement: 50,
                                        processQuality: 50,
                                        efficiency: 50,
                                    }),
                                },
                            }],
                        });
                    }),
                },
            },
        };

        let res = await grader.evaluateSession([[{ type: "text", text: "A" }]], {
            sessionId: "CRIT1",
        });
        expect(res.score).toBe(50);

        grader.updateEvaluationCriteria({
            outcomeAchievement: { weight: 0.6 },
            efficiency: { weight: 0.1 },
        });

        res = await grader.evaluateSession([[{ type: "text", text: "B" }]], {
            sessionId: "CRIT2",
        });
        expect(res.score).toBe(50); // components equal, still 50
    });

    test("rejects invalid weights that do not sum to 1.0", () => {
        const { grader } = makeGrader();
        expect(() =>
            grader.updateEvaluationCriteria({
                outcomeAchievement: { weight: 0.7 },
                processQuality: { weight: 0.3 },
                efficiency: { weight: 0.3 },
            })
        ).toThrow(/must sum to 1\.0/i);
    });
});

describe("Grader - guards and errors", () => {
    test("throws on missing API key", () => {
        expect(() => new Grader({ apiKey: "" })).toThrow(/missing or empty/i);
    });

    test("propagates parse errors for invalid final JSON", async () => {
        const { grader } = makeGrader();

        (grader as any).client = {
            chat: {
                completions: {
                    create: mock((req: any) => {
                        const isFinal =
                            req?.response_format?.json_schema?.name === "final_evaluation";
                        if (!isFinal) {
                            return Promise.resolve({
                                choices: [{ message: { content: JSON.stringify({ summary: "OK" }) } }],
                            });
                        }
                        return Promise.resolve({
                            choices: [{ message: { content: "{ not json" } }],
                        });
                    }),
                },
            },
        };

        await expect(
            grader.evaluateSession([[{ type: "text", text: "X" }]], { sessionId: "PARSE1" })
        ).rejects.toThrow(/invalid JSON/i);
    });
});