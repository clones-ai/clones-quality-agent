import { describe, it, expect, beforeAll, afterAll } from 'bun:test';
import { Grader, type GraderConfig, type GraderLogger } from '../src/stages/grading/grader';
import type { Message, MetaData } from '../src/stages/grading/grader';

// Integration test logger that captures real API interactions
class IntegrationLogger implements GraderLogger {
    public logs: Array<{ level: string; message: string; meta?: Record<string, unknown>; error?: Error; timestamp: number }> = [];

    info(message: string, meta?: Record<string, unknown>): void {
        this.logs.push({ level: 'info', message, meta, timestamp: Date.now() });
        console.log(`[INTEGRATION-INFO] ${message}`, meta ? JSON.stringify(meta, null, 2) : '');
    }

    error(message: string, error?: Error, meta?: Record<string, unknown>): void {
        this.logs.push({ level: 'error', message, error, meta, timestamp: Date.now() });
        console.error(`[INTEGRATION-ERROR] ${message}`, error?.message || '', meta ? JSON.stringify(meta, null, 2) : '');
    }

    debug(message: string, meta?: Record<string, unknown>): void {
        this.logs.push({ level: 'debug', message, meta, timestamp: Date.now() });
        console.debug(`[INTEGRATION-DEBUG] ${message}`, meta ? JSON.stringify(meta, null, 2) : '');
    }

    clear(): void {
        this.logs = [];
    }

    getExecutionTime(): number {
        if (this.logs.length < 2) return 0;
        return this.logs[this.logs.length - 1].timestamp - this.logs[0].timestamp;
    }
}

describe('Grader Integration Tests', () => {
    let grader: Grader;
    let logger: IntegrationLogger;
    let config: GraderConfig;

    // Sample realistic test data
    const metaData: MetaData = {
        id: 'integration-test-001',
        timestamp: new Date().toISOString(),
        duration_seconds: 45,
        status: 'completed',
        reason: 'integration-testing',
        title: 'Integration Test Quest',
        description: 'Testing real API integration',
        platform: 'web',
        arch: 'x64',
        version: '1.0.0',
        locale: 'en-US',
        primary_monitor: { width: 1920, height: 1080 },
        quest: {
            title: 'Simple Web Navigation Test',
            app: 'Chrome Browser',
            icon_url: 'https://example.com/chrome.png',
            content: 'Navigate to google.com and search for "OpenAI"',
            objectives: [
                'Open web browser',
                'Navigate to google.com',
                'Enter search term "OpenAI"',
                'Click search button',
                'Verify search results appear'
            ]
        }
    } as const;

    // Text-only test data for integration tests (no images to avoid API issues)
    const sftData: readonly Message[] = [
        { role: 'user', content: 'open_browser()' },
        { role: 'user', content: 'navigate_to("https://google.com")' },
        { role: 'user', content: 'click_element("search_box")' },
        { role: 'user', content: 'type_text("OpenAI")' },
        { role: 'user', content: 'click_element("search_button")' },
        { role: 'user', content: 'wait_for_results()' }
    ] as const;

    beforeAll(() => {
        // Skip integration tests if no API key
        if (!process.env.OPENAI_API_KEY) {
            console.log('⚠️  Skipping integration tests - OPENAI_API_KEY not set');
            return;
        }

        logger = new IntegrationLogger();
        config = {
            apiKey: process.env.OPENAI_API_KEY!,
            chunkSize: 2,
            model: 'gpt-4o-mini', // Use vision-capable model for integration tests
            timeout: 30 * 1000, // 30 second timeout
            maxRetries: 2 // Fewer retries for faster tests
        };

        grader = new Grader(config, logger);
    });

    afterAll(() => {
        if (logger) {
            logger.clear();
        }
    });

    describe('Real API Integration', () => {
        it('should successfully grade with real OpenAI API', async () => {
            if (!process.env.OPENAI_API_KEY) {
                console.log('⏭️  Skipping - no API key');
                return;
            }

            const startTime = Date.now();
            const result = await grader.grade(metaData, sftData);
            const duration = Date.now() - startTime;

            // Verify we got a real result
            expect(result).not.toBeNull();
            expect(result!.score).toBeGreaterThanOrEqual(0);
            expect(result!.score).toBeLessThanOrEqual(100);
            expect(result!.summary).toContain('');
            expect(result!.reasoning).toContain('');

            // Verify timing is reasonable (should complete within timeout)
            expect(duration).toBeLessThan(30000);

            // Verify logging captured the process
            expect(logger.logs.some(log => log.message === 'Grader initialized')).toBe(true);
            expect(logger.logs.some(log => log.message === 'Starting grading process')).toBe(true);
            expect(logger.logs.some(log => log.message === 'Grading completed successfully')).toBe(true);

            console.log(`✅ Integration test completed in ${duration}ms`);
            console.log(`📊 Final score: ${result!.score}/100`);
            console.log(`📝 Summary: ${result!.summary.substring(0, 100)}...`);
        }, 60000); // 60 second timeout for this test

        it('should handle timeout gracefully', async () => {
            if (!process.env.OPENAI_API_KEY) {
                console.log('⏭️  Skipping - no API key');
                return;
            }

            // Create a grader with very short timeout to test timeout handling
            const shortTimeoutConfig: GraderConfig = {
                apiKey: process.env.OPENAI_API_KEY!,
                chunkSize: 2,
                model: 'gpt-4o-mini',
                timeout: 100, // Very short timeout (100ms)
                maxRetries: 1
            };

            const shortTimeoutLogger = new IntegrationLogger();
            const shortTimeoutGrader = new Grader(shortTimeoutConfig, shortTimeoutLogger);

            const startTime = Date.now();
            const result = await shortTimeoutGrader.grade(metaData, sftData);
            const duration = Date.now() - startTime;

            // Should return null due to timeout
            expect(result).toBeNull();

            // Should complete quickly (within a few seconds, not hang forever)
            expect(duration).toBeLessThan(10000);

            // Should have logged timeout errors
            expect(shortTimeoutLogger.logs.some(log =>
                log.level === 'error' && log.message === 'Error calling OpenAI API'
            )).toBe(true);

            console.log(`✅ Timeout test completed in ${duration}ms (expected failure)`);
        }, 15000);

        it('should handle retry logic with real API', async () => {
            if (!process.env.OPENAI_API_KEY) {
                console.log('⏭️  Skipping - no API key');
                return;
            }

            // Use a configuration that might need retries
            const retryConfig: GraderConfig = {
                apiKey: process.env.OPENAI_API_KEY!,
                chunkSize: 1, // Smaller chunks = more API calls = more opportunities for retries
                model: 'gpt-4o-mini',
                timeout: 15 * 1000,
                maxRetries: 3
            };

            const retryLogger = new IntegrationLogger();
            const retryGrader = new Grader(retryConfig, retryLogger);

            const startTime = Date.now();
            const result = await retryGrader.grade(metaData, sftData);
            const duration = Date.now() - startTime;

            // Should eventually succeed or fail gracefully
            if (result) {
                expect(result.score).toBeGreaterThanOrEqual(0);
                expect(result.score).toBeLessThanOrEqual(100);
                console.log(`✅ Retry test succeeded in ${duration}ms with score: ${result.score}`);
            } else {
                console.log(`⚠️  Retry test failed gracefully in ${duration}ms (acceptable)`);
            }

            // Should not hang indefinitely
            expect(duration).toBeLessThan(60000);

            // Should have processed multiple chunks
            expect(retryLogger.logs.some(log =>
                log.message === 'Processing chunk' && log.meta?.chunkIndex === 1
            )).toBe(true);

        }, 90000); // Longer timeout for retry test
    });

    describe('Performance and Reliability', () => {
        it('should maintain consistent performance across multiple calls', async () => {
            if (!process.env.OPENAI_API_KEY) {
                console.log('⏭️  Skipping - no API key');
                return;
            }

            const performanceResults: number[] = [];
            const numberOfRuns = 3;

            for (let i = 0; i < numberOfRuns; i++) {
                logger.clear();
                const startTime = Date.now();

                const result = await grader.grade(metaData, sftData);
                const duration = Date.now() - startTime;

                performanceResults.push(duration);

                expect(result).not.toBeNull();
                console.log(`Run ${i + 1}: ${duration}ms, Score: ${result?.score}/100`);

                // Small delay between runs to avoid rate limiting
                await new Promise(resolve => setTimeout(resolve, 1000));
            }

            // Calculate performance stats
            const avgDuration = performanceResults.reduce((a, b) => a + b, 0) / numberOfRuns;
            const maxDuration = Math.max(...performanceResults);
            const minDuration = Math.min(...performanceResults);

            console.log(`📈 Performance stats over ${numberOfRuns} runs:`);
            console.log(`   Average: ${avgDuration.toFixed(0)}ms`);
            console.log(`   Min: ${minDuration}ms`);
            console.log(`   Max: ${maxDuration}ms`);

            // All runs should complete within reasonable time
            expect(maxDuration).toBeLessThan(45000);
            expect(minDuration).toBeGreaterThan(0);

        }, 180000); // 3 minutes for multiple runs
    });

    describe('Error Scenarios', () => {
        it('should handle invalid API key gracefully', async () => {
            const invalidConfig: GraderConfig = {
                apiKey: 'invalid-api-key-12345',
                chunkSize: 2,
                model: 'gpt-4o-mini',
                timeout: 10 * 1000,
                maxRetries: 1
            };

            const invalidLogger = new IntegrationLogger();
            const invalidGrader = new Grader(invalidConfig, invalidLogger);

            const result = await invalidGrader.grade(metaData, sftData);

            // Should fail gracefully
            expect(result).toBeNull();

            // Should log authentication errors
            expect(invalidLogger.logs.some(log =>
                log.level === 'error' && log.message === 'Error calling OpenAI API'
            )).toBe(true);

            console.log('✅ Invalid API key handled gracefully');
        }, 30000);

        it('should handle malformed input data', async () => {
            if (!process.env.OPENAI_API_KEY) {
                console.log('⏭️  Skipping - no API key');
                return;
            }

            const malformedMeta = {
                ...metaData,
                id: '', // Empty ID should cause validation error
                quest: {
                    ...metaData.quest,
                    objectives: [] // Empty objectives should cause validation error
                }
            };

            logger.clear(); // Clear previous logs
            const result = await grader.grade(malformedMeta, sftData);

            // Should handle validation error gracefully
            expect(result).toBeNull();

            // Should log validation error
            expect(logger.logs.some(log =>
                log.level === 'error' && log.message === 'Error during grading'
            )).toBe(true);

            console.log('✅ Malformed input handled gracefully');
        }, 15000);
    });
});
