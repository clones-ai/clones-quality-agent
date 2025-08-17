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
            const finalEvaluation = JSON.stringify({
                summary: "Final Summary",
                analysis: "Final Analysis",
                score: 95,
                reasoning: "Final Reasoning",
                confidence: 0.9,
                outcomeAchievement: 90,
                processQuality: 95,
                efficiency: 85
            });

            // Mock API responses with proper typing
            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: finalEvaluation } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).toEqual({
                summary: "Final Summary",
                analysis: "Final Analysis",
                score: 95,
                reasoning: "Final Reasoning",
                confidence: 0.9,
                outcomeAchievement: 90,
                processQuality: 95,
                efficiency: 85
            });

            // Should be called twice (once for each chunk)
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(2);

            // Check logging
            expect(testLogger.logs.some(log => log.message === 'Starting grading process')).toBe(true);
            expect(testLogger.logs.some(log => log.message === 'Grading completed successfully')).toBe(true);
        });

        it('should succeed after one retry on a chunk', async () => {
            const finalEvaluation = JSON.stringify({
                summary: "Final Summary",
                analysis: "Final Analysis",
                score: 90,
                reasoning: "Final Reasoning",
                confidence: 0.85,
                outcomeAchievement: 85,
                processQuality: 90,
                efficiency: 80
            });

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
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
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
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
            const finalEvaluation = JSON.stringify({
                summary: "Final Summary",
                analysis: "Final Analysis",
                score: 90,
                reasoning: "Final Reasoning",
                confidence: 0.8,
                outcomeAchievement: 85,
                processQuality: 88,
                efficiency: 82
            });

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
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
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
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
            const invalidScoreEvaluation = JSON.stringify({
                summary: "Final Summary",
                analysis: "Final Analysis",
                score: "invalid",  // Invalid score type
                reasoning: "Final Reasoning",
                confidence: 0.7,
                outcomeAchievement: 70,
                processQuality: 75,
                efficiency: 65
            });

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
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

            const finalEvaluation = JSON.stringify({
                summary: "Final Summary",
                analysis: "Final Analysis",
                score: 85,
                reasoning: "Final Reasoning",
                confidence: 0.9,
                outcomeAchievement: 80,
                processQuality: 85,
                efficiency: 90
            });

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

    describe('Evaluation Criteria Management', () => {
        it('should get current evaluation criteria', () => {
            const criteria = grader.getEvaluationCriteria();

            expect(criteria).toEqual({
                outcomeAchievement: {
                    weight: 0.5,
                    description: 'Goal completion and objective fulfillment'
                },
                processQuality: {
                    weight: 0.3,
                    description: 'Problem-solving approach, error recovery, and adaptability'
                },
                efficiency: {
                    weight: 0.2,
                    description: 'Time management, direct paths, and resource utilization'
                }
            });
        });

        it('should update evaluation criteria successfully', () => {
            grader.updateEvaluationCriteria({
                outcomeAchievement: { weight: 0.6 },
                efficiency: { weight: 0.1 }
            });

            const criteria = grader.getEvaluationCriteria();
            expect(criteria.outcomeAchievement.weight).toBe(0.6);
            expect(criteria.processQuality.weight).toBe(0.3); // Unchanged
            expect(criteria.efficiency.weight).toBe(0.1);

            // Check logging
            expect(testLogger.logs.some(log =>
                log.level === 'info' && log.message === 'Evaluation criteria updated'
            )).toBe(true);
        });

        it('should validate that weights sum to 1.0', () => {
            expect(() => {
                grader.updateEvaluationCriteria({
                    outcomeAchievement: { weight: 0.8 },
                    processQuality: { weight: 0.3 },
                    efficiency: { weight: 0.2 }
                });
            }).toThrow('Evaluation criteria weights must sum to 1.0, got 1.3');

            // Check error logging
            expect(testLogger.logs.some(log =>
                log.level === 'error' && log.message === 'Evaluation criteria weights do not sum to 1.0'
            )).toBe(true);
        });

        it('should allow small tolerance in weight validation', () => {
            // Should not throw with small rounding errors
            expect(() => {
                grader.updateEvaluationCriteria({
                    outcomeAchievement: { weight: 0.501 },
                    processQuality: { weight: 0.299 },
                    efficiency: { weight: 0.2 }
                });
            }).not.toThrow();
        });
    });

    describe('Modern JSON Parsing', () => {
        it('should parse JSON from code blocks', async () => {
            const jsonInCodeBlock = `\`\`\`json
{
  "summary": "Test Summary",
  "analysis": "Test Analysis",
  "score": 75,
  "reasoning": "Test Reasoning",
  "confidence": 0.8,
  "outcomeAchievement": 70,
  "processQuality": 80,
  "efficiency": 75
}
\`\`\``;

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: jsonInCodeBlock } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).not.toBeNull();
            expect(result?.score).toBe(75);
            expect(result?.confidence).toBe(0.8);
        });

        it('should handle malformed JSON gracefully', async () => {
            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockResolvedValue({
                    choices: [{ message: { content: '{ invalid json }' } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).toBeNull();
            expect(testLogger.logs.some(log =>
                log.level === 'error' && log.message === 'Failed to parse JSON response'
            )).toBe(true);
        });
    });

    describe('Score Validation and Variance Detection', () => {
        it('should detect score calculation variance', async () => {
            const inconsistentEvaluation = JSON.stringify({
                summary: "Test Summary",
                analysis: "Test Analysis",
                score: 50, // Inconsistent with components
                reasoning: "Test Reasoning",
                confidence: 0.9,
                outcomeAchievement: 90, // 90 * 0.5 = 45
                processQuality: 80,     // 80 * 0.3 = 24  
                efficiency: 70          // 70 * 0.2 = 14
                // Expected: 45 + 24 + 14 = 83, but score is 50 (variance > 10)
            });

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: inconsistentEvaluation } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).not.toBeNull();
            expect(result?.score).toBe(50); // Uses provided score despite variance

            // Check variance logging
            expect(testLogger.logs.some(log =>
                log.level === 'debug' && log.message === 'Score calculation variance detected'
            )).toBe(true);
        });

        it('should clamp component scores to valid ranges', async () => {
            const extremeEvaluation = JSON.stringify({
                summary: "Test Summary",
                analysis: "Test Analysis",
                score: 75,
                reasoning: "Test Reasoning",
                confidence: 1.5, // > 1.0
                outcomeAchievement: 150, // > 100
                processQuality: -10,     // < 0
                efficiency: 80
            });

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: extremeEvaluation } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).not.toBeNull();
            expect(result?.confidence).toBe(1.0); // Clamped to max
            expect(result?.outcomeAchievement).toBe(100); // Clamped to max
            expect(result?.processQuality).toBe(0); // Clamped to min
            expect(result?.efficiency).toBe(80); // Within range
        });
    });

    describe('Enhanced Logging', () => {
        it('should log initialization with detailed config', () => {
            expect(testLogger.logs.some(log =>
                log.level === 'info' &&
                log.message === 'Grader initialized' &&
                log.meta?.model === 'gpt-4o' &&
                log.meta?.chunkSize === 2 &&
                log.meta?.maxRetries === 3
            )).toBe(true);
        });

        it('should log grading process with metrics', async () => {
            const finalEvaluation = JSON.stringify({
                summary: "Test Summary",
                analysis: "Test Analysis",
                score: 85,
                reasoning: "Test Reasoning",
                confidence: 0.9,
                outcomeAchievement: 80,
                processQuality: 90,
                efficiency: 85
            });

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: finalEvaluation } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).not.toBeNull();

            // Check process logging
            expect(testLogger.logs.some(log =>
                log.level === 'info' && log.message === 'Starting grading process'
            )).toBe(true);

            expect(testLogger.logs.some(log =>
                log.level === 'info' && log.message === 'Grading completed successfully'
            )).toBe(true);
        });
    });
});