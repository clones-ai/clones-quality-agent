import { describe, it, expect, mock, beforeEach, afterEach, spyOn } from 'bun:test';
import { Grader, type GraderConfig, type GraderLogger } from '../src/stages/grading/grader';
import type { Message, MetaData, GradeResult } from '../src/stages/grading/grader';
import type { OpenAI } from 'openai';

// Create a proper mock for OpenAI chat completions
const mockChatCompletionsCreate = mock();

// Mock the OpenAI module
mock.module('openai', () => ({
    OpenAI: class MockOpenAI {
        chat = {
            completions: {
                create: mockChatCompletionsCreate
            }
        };

        constructor(_config: any) {
            // Mock constructor
        }
    }
}));

// Mock logger for testing - suppresses console output during tests
class TestLogger implements GraderLogger {
    public logs: Array<{ level: string; message: string; meta?: Record<string, unknown>; error?: Error }> = [];

    info(message: string, meta?: Record<string, unknown>): void {
        this.logs.push({ level: 'info', message, meta });
        // Suppress console output during tests
    }

    error(message: string, error?: Error, meta?: Record<string, unknown>): void {
        this.logs.push({ level: 'error', message, error, meta });
        // Suppress console output during tests
    }

    debug(message: string, meta?: Record<string, unknown>): void {
        this.logs.push({ level: 'debug', message, meta });
        // Suppress console output during tests
    }

    clear(): void {
        this.logs = [];
    }
}

describe('Grader', () => {
    let grader: Grader;
    let testLogger: TestLogger;
    let config: GraderConfig;

    // Sample data with proper typing
    const metaData: MetaData = {
        id: 'test-id',
        timestamp: new Date().toISOString(),
        duration_seconds: 120,
        status: 'completed',
        reason: 'testing',
        title: 'Test Quest',
        description: 'A test quest',
        platform: 'test-platform',
        arch: 'test-arch',
        version: '1.0',
        locale: 'en-US',
        primary_monitor: { width: 1920, height: 1080 },
        quest: {
            title: 'Test Quest',
            app: 'Test App',
            icon_url: 'http://example.com/icon.png',
            content: 'Do the test.',
            objectives: ['Objective 1', 'Objective 2']
        }
    } as const;

    const sftData: readonly Message[] = [
        { role: 'user', content: { type: 'image', data: 'img1' } },
        { role: 'user', content: 'click(1,1)' },
        { role: 'user', content: { type: 'image', data: 'img2' } },
        { role: 'user', content: 'type(hello)' }
    ] as const;

    beforeEach(() => {
        // Reset mocks before each test
        mockChatCompletionsCreate.mockReset();

        // Create test logger
        testLogger = new TestLogger();

        // Create config
        config = {
            apiKey: 'test-api-key',
            chunkSize: 2,
            timeout: 5000,
            maxRetries: 3
        };

        // Instantiate Grader with test logger
        grader = new Grader(config, testLogger);
    });

    afterEach(() => {
        testLogger.clear();
    });

    describe('Constructor', () => {
        it('should create grader with valid config', () => {
            expect(grader).toBeInstanceOf(Grader);
            expect(testLogger.logs.some(log => log.message === 'Grader initialized')).toBe(true);
        });

        it('should throw error with empty API key', () => {
            expect(() => new Grader({ ...config, apiKey: '' })).toThrow('OpenAI API key is required and cannot be empty');
        });

        it('should use legacy constructor', () => {
            const legacyGrader = Grader.create('test-key', 4, 'gpt-4');
            expect(legacyGrader).toBeInstanceOf(Grader);
        });

        it('should handle invalid chunk size', () => {
            const graderWithInvalidSize = new Grader({ ...config, chunkSize: -1 }, testLogger);
            expect(graderWithInvalidSize).toBeInstanceOf(Grader);
            // Should default to minimum of 1
        });
    });

    describe('Grade method', () => {
        it('should process successfully on the first try', async () => {
            const finalEvaluation = `
        <summary>Final Summary</summary>
        <scratchpad>Final Scratchpad</scratchpad>
        <answer>95</answer>
        <reasoning>Final Reasoning</reasoning>
      `;

            // Mock API responses with proper typing
            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: '<summary>Intermediate Summary</summary>' } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: finalEvaluation } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).toEqual({
                summary: 'Final Summary',
                scratchpad: 'Final Scratchpad',
                score: 95,
                reasoning: 'Final Reasoning'
            });

            // Should be called twice (once for each chunk)
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(2);

            // Check logging
            expect(testLogger.logs.some(log => log.message === 'Starting grading process')).toBe(true);
            expect(testLogger.logs.some(log => log.message === 'Grading completed successfully')).toBe(true);
        });

        it('should succeed after one retry on a chunk', async () => {
            const finalEvaluation = `
        <summary>Final Summary</summary>
        <scratchpad>Final Scratchpad</scratchpad>
        <answer>90</answer>
        <reasoning>Final Reasoning</reasoning>
      `;

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: '<summary>Intermediate Summary</summary>' } }]
                })
                .mockRejectedValueOnce(new Error('API Error'))
                .mockResolvedValueOnce({
                    choices: [{ message: { content: finalEvaluation } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).not.toBeNull();
            expect(result?.score).toBe(90);

            // Called 3 times: chunk 1 (1), chunk 2 (2 attempts)
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(3);

            // Check error logging
            expect(testLogger.logs.some(log =>
                log.level === 'error' && log.message === 'Failed to get evaluation from API'
            )).toBe(true);
        });

        it('should fail after max retries', async () => {
            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: '<summary>Intermediate Summary</summary>' } }]
                })
                .mockRejectedValue(new Error('API Error'));

            const result = await grader.grade(metaData, sftData);

            expect(result).toBeNull();

            // Called 4 times: chunk 1 (1), chunk 2 (3 attempts)
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(4);

            // Check failure logging
            expect(testLogger.logs.some(log =>
                log.level === 'error' && log.message === 'Failed to process chunk after retries'
            )).toBe(true);
        });

        it('should retry if the response format is invalid', async () => {
            const finalEvaluation = `
        <summary>Final Summary</summary>
        <scratchpad>Final Scratchpad</scratchpad>
        <answer>90</answer>
        <reasoning>Final Reasoning</reasoning>
      `;

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: '<summary>Intermediate Summary</summary>' } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: 'invalid format' } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: finalEvaluation } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).not.toBeNull();
            expect(result?.score).toBe(90);

            // Called 3 times: chunk 1 (1), chunk 2 (2 attempts)
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(3);
        });

        it('should fail if the response format is always invalid after max retries', async () => {
            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: '<summary>Intermediate Summary</summary>' } }]
                })
                .mockResolvedValue({
                    choices: [{ message: { content: 'always invalid' } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).toBeNull();

            // Called 4 times: chunk 1 (1), chunk 2 (3 attempts)
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(4);
        });

        it('should handle invalid metadata', async () => {
            const invalidMeta = { ...metaData, id: '', quest: { ...metaData.quest, objectives: [] } };

            const result = await grader.grade(invalidMeta, sftData);

            expect(result).toBeNull();
            expect(testLogger.logs.some(log =>
                log.level === 'error' && log.message === 'Error during grading'
            )).toBe(true);
        });

        it('should handle empty messages array', async () => {
            const result = await grader.grade(metaData, []);

            expect(result).toBeNull();
            expect(testLogger.logs.some(log =>
                log.level === 'error' && log.message === 'Error during grading'
            )).toBe(true);
        });

        it('should handle invalid score in evaluation', async () => {
            const invalidScoreEvaluation = `
        <summary>Final Summary</summary>
        <scratchpad>Final Scratchpad</scratchpad>
        <answer>invalid</answer>
        <reasoning>Final Reasoning</reasoning>
      `;

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: '<summary>Intermediate Summary</summary>' } }]
                })
                .mockResolvedValue({
                    choices: [{ message: { content: invalidScoreEvaluation } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).toBeNull();
            expect(testLogger.logs.some(log =>
                log.level === 'error' && log.message === 'Invalid score in evaluation'
            )).toBe(true);
        });
    });

    describe('Message filtering', () => {
        it('should filter out scroll messages', async () => {
            const messagesWithScroll: readonly Message[] = [
                { role: 'user', content: { type: 'image', data: 'img1' } },
                { role: 'user', content: 'scroll(0, 100)' },
                { role: 'user', content: 'click(1,1)' },
                { role: 'user', content: '```python\nscroll(0, -50)\n```' }
            ];

            const finalEvaluation = `
        <summary>Final Summary</summary>
        <scratchpad>Final Scratchpad</scratchpad>
        <answer>85</answer>
        <reasoning>Final Reasoning</reasoning>
      `;

            mockChatCompletionsCreate.mockResolvedValue({
                choices: [{ message: { content: finalEvaluation } }]
            });

            const result = await grader.grade(metaData, messagesWithScroll);

            expect(result).not.toBeNull();
            expect(result?.score).toBe(85);

            // Should only process non-scroll messages (2 messages = 1 chunk with chunkSize 2)
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(1);

            // Check filtering logs
            expect(testLogger.logs.some(log =>
                log.level === 'debug' && log.message === 'Messages filtered'
            )).toBe(true);
        });
    });
});