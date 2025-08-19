import { describe, it, expect } from "bun:test";
import { spawn } from "child_process";
import * as fs from "fs/promises";
import * as path from "path";
import {
    GraderConfig,
    GraderLogger,
    Chunk,
    MetaData,
} from "../src/stages/grading/grader/types";
import { Grader } from "../src/stages/grading/grader";

/** Simple integration logger implementing the GraderLogger interface. */
class IntegrationLogger implements GraderLogger {
    public logs: Array<{
        level: "debug" | "info" | "warn" | "error";
        message: string;
        error?: Error;
        meta?: Record<string, unknown>;
        ts: number;
    }> = [];

    debug(message: string, err?: Error, meta?: Record<string, unknown>) {
        this.logs.push({ level: "debug", message, error: err, meta, ts: Date.now() });
        // console.debug(`[INT-DEBUG] ${message}`, err?.message ?? "", meta ?? "");
    }
    info(message: string, err?: Error, meta?: Record<string, unknown>) {
        this.logs.push({ level: "info", message, error: err, meta, ts: Date.now() });
        // console.log(`[INT-INFO] ${message}`, err?.message ?? "", meta ?? "");
    }
    warn(message: string, err?: Error, meta?: Record<string, unknown>) {
        this.logs.push({ level: "warn", message, error: err, meta, ts: Date.now() });
        // console.warn(`[INT-WARN] ${message}`, err?.message ?? "", meta ?? "");
    }
    error(message: string, err?: Error, meta?: Record<string, unknown>) {
        this.logs.push({ level: "error", message, error: err, meta, ts: Date.now() });
        // console.error(`[INT-ERROR] ${message}`, err?.message ?? "", meta ?? "");
    }
}

/** Convert a list of textual "actions" into Grader chunks of size N. */
function toChunks(actions: string[], chunkSize = 2): Chunk[] {
    const chunks: Chunk[] = [];
    for (let i = 0; i < actions.length; i += chunkSize) {
        const slice = actions.slice(i, i + chunkSize);
        chunks.push(slice.map((t) => ({ type: "text", text: t })));
    }
    return chunks;
}

/** Sample text-only actions for a realistic, but simple, web task. */
const sftActions = [
    'open_browser()',
    'navigate_to("https://google.com")',
    'click_element("search_box")',
    'type_text("OpenAI")',
    'click_element("search_button")',
    "wait_for_results()",
];

const meta: MetaData = {
    sessionId: "integration-test-001",
    platform: "web",
    taskDescription: 'Navigate to google.com and search for "OpenAI"',
};

describe("Grader Pipeline (spawn)", () => {
    it(
        "grades a real session and checks for metrics",
        async () => {
            if (!process.env.OPENAI_API_KEY) {
                console.log("⏭️  Skipping: OPENAI_API_KEY not set");
                return;
            }

            const testSessionDir = "data/tests/grader/20250817_232256";
            const scoresPath = path.join(testSessionDir, "scores.json");
            const metricsPath = path.join(testSessionDir, "metrics.json");

            // Clean up previous run's output
            await fs.rm(scoresPath, { force: true });
            await fs.rm(metricsPath, { force: true });

            await new Promise<void>((resolve, reject) => {
                const pipeline = spawn("bun", [
                    "run",
                    "src/index.ts",
                    "-f",
                    "desktop",
                    "-i",
                    testSessionDir,
                    "--grade",
                ]);

                let stdout = "";
                let stderr = "";

                pipeline.stdout.on("data", (data) => {
                    stdout += data;
                    console.log("▶️", data.toString().trim());
                });

                pipeline.stderr.on("data", (data) => {
                    stderr += data;
                    console.error("⛔️", data.toString().trim());
                });

                pipeline.on("close", (code: number) => {
                    if (code === 0) {
                        resolve();
                    } else {
                        reject(
                            new Error(
                                `Pipeline failed:\nstdout: ${stdout}\nstderr: ${stderr}`
                            )
                        );
                    }
                });

                pipeline.on("error", (err) => {
                    reject(err);
                });
            });

            // Check scores.json
            const scoresContent = await fs.readFile(scoresPath, "utf8");
            const gradeResult = JSON.parse(scoresContent);

            expect(gradeResult.summary.length).toBeGreaterThan(0);
            expect(gradeResult.observations.length).toBeGreaterThan(0);
            expect(gradeResult.reasoning.length).toBeGreaterThan(0);
            expect(gradeResult.score).toBeGreaterThanOrEqual(0);
            expect(gradeResult.score).toBeLessThanOrEqual(100);
            expect(gradeResult.outcomeAchievement).toBeGreaterThanOrEqual(0);
            expect(gradeResult.outcomeAchievement).toBeLessThanOrEqual(100);
            expect(gradeResult.processQuality).toBeGreaterThanOrEqual(0);
            expect(gradeResult.processQuality).toBeLessThanOrEqual(100);
            expect(gradeResult.efficiency).toBeGreaterThanOrEqual(0);
            expect(gradeResult.efficiency).toBeLessThanOrEqual(100);
            expect(gradeResult.confidence).toBeGreaterThanOrEqual(0);
            expect(gradeResult.confidence).toBeLessThanOrEqual(100);

            // Check metrics.json
            const metricsContent = await fs.readFile(metricsPath, "utf8");
            const metricsResult = JSON.parse(metricsContent);
            expect(metricsResult).toBeDefined();
            expect(metricsResult.totalTokens).toBeGreaterThan(0);

            console.log("✅ Pipeline test completed successfully.");
        },
        120_000 // 2 minute timeout for the whole test
    );
});

describe("Grader Integration (real API)", () => {
    it(
        "smoke: evaluates a short session successfully",
        async () => {
            if (!process.env.OPENAI_API_KEY) {
                console.log("⏭️  Skipping: OPENAI_API_KEY not set");
                return;
            }

            const logger = new IntegrationLogger();
            const config: GraderConfig = {
                apiKey: process.env.OPENAI_API_KEY!,
                chunkSize: 2,
                model: "gpt-4o-mini", // fast & vision-capable model
                timeout: 30_000,
                maxRetries: 2,
            };

            const grader = new Grader(config, logger);
            const chunks = toChunks(sftActions, config.chunkSize ?? 2);

            const t0 = Date.now();
            const result = await grader.evaluateSession(chunks, meta);
            const elapsed = Date.now() - t0;

            // Basic sanity checks
            expect(result.summary.length).toBeGreaterThan(0);
            expect(result.observations.length).toBeGreaterThan(0);
            expect(result.reasoning.length).toBeGreaterThan(0);
            expect(result.score).toBeGreaterThanOrEqual(0);
            expect(result.score).toBeLessThanOrEqual(100);
            expect(result.outcomeAchievement).toBeGreaterThanOrEqual(0);
            expect(result.outcomeAchievement).toBeLessThanOrEqual(100);
            expect(result.processQuality).toBeGreaterThanOrEqual(0);
            expect(result.processQuality).toBeLessThanOrEqual(100);
            expect(result.efficiency).toBeGreaterThanOrEqual(0);
            expect(result.efficiency).toBeLessThanOrEqual(100);
            expect(result.confidence).toBeGreaterThanOrEqual(0);
            expect(result.confidence).toBeLessThanOrEqual(100);

            // Should finish within the timeout budget (plus small overhead)
            expect(elapsed).toBeLessThan(35_000);

            console.log(`✅ Smoke test ok in ${elapsed}ms — score=${result.score}/100`);
        },
        60_000
    );

    it(
        "handles very short timeout by throwing (graceful failure)",
        async () => {
            if (!process.env.OPENAI_API_KEY) {
                console.log("⏭️  Skipping: OPENAI_API_KEY not set");
                return;
            }

            const logger = new IntegrationLogger();
            const config: GraderConfig = {
                apiKey: process.env.OPENAI_API_KEY!,
                chunkSize: 2,
                model: "gpt-4o-mini",
                timeout: 100, // force fast abort
                maxRetries: 1,
            };
            const grader = new Grader(config, logger);
            const chunks = toChunks(sftActions, 2);

            const t0 = Date.now();
            let threw = false;
            try {
                await grader.evaluateSession(chunks, meta);
            } catch (e) {
                threw = true;
                // We expect an AbortError or a wrapped error after retries.
                expect(String((e as Error).message).toLowerCase()).toContain("");
            }
            const elapsed = Date.now() - t0;

            expect(threw).toBe(true);
            expect(elapsed).toBeLessThan(10_000);

            // Optionally, assert that some error logs were captured
            expect(
                logger.logs.some((log) => log.level === "error")
            ).toBeTruthy();

            console.log(`✅ Timeout path ok in ${elapsed}ms (expected throw)`);
        },
        15_000
    );

    it(
        "can perform multiple runs consistently (performance sanity)",
        async () => {
            if (!process.env.OPENAI_API_KEY) {
                console.log("⏭️  Skipping: OPENAI_API_KEY not set");
                return;
            }

            const logger = new IntegrationLogger();
            const config: GraderConfig = {
                apiKey: process.env.OPENAI_API_KEY!,
                chunkSize: 2,
                model: "gpt-4o-mini",
                timeout: 30_000,
                maxRetries: 2,
            };
            const grader = new Grader(config, logger);
            const chunks = toChunks(sftActions, 2);

            const runs = 3;
            const durations: number[] = [];
            for (let i = 0; i < runs; i++) {
                const t0 = Date.now();
                const res = await grader.evaluateSession(chunks, meta);
                const dt = Date.now() - t0;
                durations.push(dt);
                expect(res.score).toBeGreaterThanOrEqual(0);
                expect(res.score).toBeLessThanOrEqual(100);
                await new Promise((r) => setTimeout(r, 800)); // small gap to avoid rate-limits
            }

            const max = Math.max(...durations);
            const min = Math.min(...durations);
            const avg = Math.round(durations.reduce((a, b) => a + b, 0) / runs);

            console.log(
                `📈 Perf over ${runs} runs — min=${min}ms avg=${avg}ms max=${max}ms`
            );
            expect(max).toBeLessThan(45_000);
            expect(min).toBeGreaterThan(0);
        },
        120_000
    );

    it(
        "invalid API key → throws (auth failure path)",
        async () => {
            // This case does not require a real API key
            const logger = new IntegrationLogger();
            const config: GraderConfig = {
                apiKey: "invalid-api-key-12345",
                chunkSize: 2,
                model: "gpt-4o-mini",
                timeout: 10_000,
                maxRetries: 1,
            };
            const grader = new Grader(config, logger);
            const chunks = toChunks(sftActions, 2);

            let threw = false;
            try {
                await grader.evaluateSession(chunks, meta);
            } catch (e) {
                threw = true;
                // Error message text can vary; just ensure we did throw.
                expect(String((e as Error).message).length).toBeGreaterThan(0);
            }
            expect(threw).toBe(true);

            // Expect at least one error log recorded
            expect(logger.logs.some((l) => l.level === "error")).toBeTruthy();

            console.log("✅ Invalid API key handled with throw (as expected)");
        },
        30_000
    );
});
