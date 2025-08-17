import { describe, it, expect, mock, beforeEach, afterEach, spyOn, jest } from 'bun:test';
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

// Mock the sleep function
const mockSleep = mock((ms: number) => Promise.resolve());
mock.module('../src/shared/utils/sleep', () => ({
    sleep: mockSleep,
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
        mockSleep.mockClear();
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
                observations: "Key observations",
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
                observations: "Key observations",
                score: 91, // Calculated score: 90*0.5 + 95*0.3 + 85*0.2 = 45 + 28.5 + 17 = 90.5 ≈ 91
                modelScoreRaw: 95,
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
                observations: "Key observations after retry",
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
            expect(result?.score).toBe(86); // Calculated: 85*0.5 + 90*0.3 + 80*0.2 = 42.5 + 27 + 16 = 85.5 ≈ 86
            expect(result?.modelScoreRaw).toBe(90);

            // Called 3 times: chunk 1 (1), chunk 2 (2 attempts)
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(3);
            expect(mockSleep).toHaveBeenCalledTimes(1); // Called once for the delay

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
            expect(mockSleep).toHaveBeenCalledTimes(2); // Called for the first 2 retries

            // Check failure logging
            expect(testLogger.logs.some(log =>
                log.level === 'error' && log.message === 'Failed to process chunk after retries'
            )).toBe(true);
        });

        it('should retry if the response format is invalid', async () => {
            const finalEvaluation = JSON.stringify({
                summary: "Final Summary",
                observations: "Observations after invalid format",
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
            expect(result?.score).toBe(85); // Calculated: 85*0.5 + 88*0.3 + 82*0.2 = 42.5 + 26.4 + 16.4 = 85.3 ≈ 85
            expect(result?.modelScoreRaw).toBe(90);

            // Called 3 times: chunk 1 (1), chunk 2 (2 attempts)
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(3);
            expect(mockSleep).toHaveBeenCalledTimes(1);
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
            expect(mockSleep).toHaveBeenCalledTimes(2);
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

        it('should handle edge case scores within valid bounds', async () => {
            const edgeCaseEvaluation = JSON.stringify({
                summary: "Final Summary",
                observations: "Observations with edge case scores",
                score: 0,  // Valid minimum score
                reasoning: "Final Reasoning",
                confidence: 0.0,  // Valid minimum confidence
                outcomeAchievement: 100,  // Valid maximum
                processQuality: 0,   // Valid minimum
                efficiency: 50
            });

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockResolvedValue({
                    choices: [{ message: { content: edgeCaseEvaluation } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).not.toBeNull();
            expect(result?.score).toBe(60); // Calculated: 100*0.5 + 0*0.3 + 50*0.2 = 50 + 0 + 10 = 60
            expect(result?.modelScoreRaw).toBe(0);
            expect(result?.confidence).toBe(0.0);
            expect(result?.outcomeAchievement).toBe(100);
            expect(result?.processQuality).toBe(0);
        });

        it('should use exponential backoff with jitter for retries', async () => {
            // Mock sleep to capture the delay values
            const sleepDelays: number[] = [];
            mockSleep.mockImplementation((ms: number) => {
                sleepDelays.push(ms);
                return Promise.resolve();
            });

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockRejectedValueOnce(new Error('API Error'))
                .mockRejectedValueOnce(new Error('API Error'))
                .mockResolvedValueOnce({
                    choices: [{
                        message: {
                            content: JSON.stringify({
                                summary: "Final Summary",
                                observations: "Final observations",
                                score: 85,
                                reasoning: "Final Reasoning",
                                confidence: 0.9,
                                outcomeAchievement: 80,
                                processQuality: 90,
                                efficiency: 85
                            })
                        }
                    }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).not.toBeNull();
            expect(mockSleep).toHaveBeenCalledTimes(2);

            // Verify exponential backoff: 500 * 2^(retries-1) + jitter
            // First retry: 500 * 2^0 = 500ms base + jitter (0-250ms) = 500-750ms
            // Second retry: 500 * 2^1 = 1000ms base + jitter (0-250ms) = 1000-1250ms
            expect(sleepDelays[0]).toBeGreaterThanOrEqual(500);
            expect(sleepDelays[0]).toBeLessThanOrEqual(750);
            expect(sleepDelays[1]).toBeGreaterThanOrEqual(1000);
            expect(sleepDelays[1]).toBeLessThanOrEqual(1250);

            // Verify delays are different (jitter working)
            expect(sleepDelays[0]).not.toBe(sleepDelays[1]);
        });

        it('should correctly chain summaries between chunks', async () => {
            const chunk1Summary = "Chunk 1 summary.";
            const chunk2Summary = "Chunk 2 summary, building on chunk 1.";
            const finalEvaluation = JSON.stringify({
                summary: "Final Summary",
                observations: "Final observations",
                score: 95,
                reasoning: "Final Reasoning",
                confidence: 0.9,
                outcomeAchievement: 90,
                processQuality: 95,
                efficiency: 85
            });

            // Create a larger dataset to force 3 chunks with chunkSize = 2
            const multiChunkSftData: readonly Message[] = [
                ...sftData,
                { role: 'user', content: { type: 'image', data: 'img3' } },
                { role: 'user', content: 'click(2,2)' },
            ];

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: chunk1Summary }) } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: chunk2Summary }) } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: finalEvaluation } }]
                });

            // Spy on createSystemPrompt to check its input
            const createSystemPromptSpy = spyOn(grader, 'createSystemPrompt');

            const result = await grader.grade(metaData, multiChunkSftData);

            expect(result).not.toBeNull();
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(3);

            // Check that prevSummary is passed correctly
            // First call, no prevSummary
            expect(createSystemPromptSpy.mock.calls[0][1]).toBeNull();
            // Second call, receives summary from the first
            expect(createSystemPromptSpy.mock.calls[1][1]).toBe(chunk1Summary);
            // Third call, receives summary from the second
            expect(createSystemPromptSpy.mock.calls[2][1]).toBe(chunk2Summary);

            createSystemPromptSpy.mockRestore();
        });
    });

    describe('Message filtering', () => {
        it('should filter out scroll messages', async () => {
            const messagesWithScroll: readonly Message[] = [
                { role: 'user', content: { type: 'image', data: 'img1' } },
                { role: 'user', content: 'scroll(0, 100)' },
                { role: 'user', content: 'click(1,1)' },
                { role: 'user', content: '```python\nscroll(0, -50)\n```' },
                { role: 'user', content: '```js\r\nscroll(100, 0)\r\n```' }, // CRLF line endings
                { role: 'user', content: '```\nscroll(50, 50)\n```' },      // No language specified
                { role: 'user', content: '```PYTHON\nscroll(0, 200)\n```' }, // Uppercase language
                { role: 'user', content: '```typescript\n  scroll(10, 20)  \n```' } // Whitespace around scroll
            ];

            const finalEvaluation = JSON.stringify({
                summary: "Final Summary",
                observations: "Observations from filtered messages",
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
            expect(result?.score).toBe(84); // Calculated: 80*0.5 + 85*0.3 + 90*0.2 = 40 + 25.5 + 18 = 83.5 ≈ 84
            expect(result?.modelScoreRaw).toBe(85);

            // Should only process non-scroll messages (2 messages: 1 image + 1 click = 1 chunk with chunkSize 2)
            // All scroll messages should be filtered out (5 total scroll messages in various formats)
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(1);

            // Check filtering logs
            expect(testLogger.logs.some(log =>
                log.level === 'debug' && log.message === 'Messages filtered'
            )).toBe(true);
        });

        it('should handle various code block formats robustly', async () => {
            const codeBlockVariations: readonly Message[] = [
                { role: 'user', content: { type: 'image', data: 'img1' } },
                { role: 'user', content: '```python\nclick(1,1)\n```' },           // Standard Python
                { role: 'user', content: '```js\r\nclick(2,2)\r\n```' },           // JavaScript with CRLF
                { role: 'user', content: '```\nclick(3,3)\n```' },                 // No language specified
                { role: 'user', content: '```TYPESCRIPT\nclick(4,4)\n```' },       // Uppercase language
                { role: 'user', content: '```py\n  click(5,5)  \n```' },           // Python alias with whitespace
                { role: 'user', content: 'click(6,6)' },                           // No code block
                // These should be filtered out (scroll commands)
                { role: 'user', content: '```bash\nscroll(0, 100)\n```' },         // Bash scroll
                { role: 'user', content: '```\r\nscroll(50, 0)\r\n```' },          // No lang + CRLF
                { role: 'user', content: '```SHELL\n\tscroll(0, -50)\t\n```' }      // Uppercase + tabs
            ];

            const finalEvaluation = JSON.stringify({
                summary: "Final Summary",
                observations: "Observations from robust parsing",
                score: 88,
                reasoning: "Final Reasoning",
                confidence: 0.95,
                outcomeAchievement: 85,
                processQuality: 90,
                efficiency: 88
            });

            mockChatCompletionsCreate.mockResolvedValue({
                choices: [{ message: { content: finalEvaluation } }]
            });

            const result = await grader.grade(metaData, codeBlockVariations);

            expect(result).not.toBeNull();
            expect(result?.score).toBe(87); // Calculated: 85*0.5 + 90*0.3 + 88*0.2 = 42.5 + 27 + 17.6 = 87.1 ≈ 87
            expect(result?.modelScoreRaw).toBe(88);

            // Should process 7 non-scroll messages (1 image + 6 click commands)
            // With chunkSize=2, that's 4 chunks total (2+2+2+1)
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(4);

            // Verify filtering worked - should have filtered out 3 scroll commands
            const filteredLog = testLogger.logs.find(log =>
                log.level === 'debug' && log.message === 'Messages filtered'
            );
            expect(filteredLog?.meta?.originalCount).toBe(10); // Total messages
            expect(filteredLog?.meta?.filteredCount).toBe(7);  // After filtering
        });

        it('should calculate action density correctly', async () => {
            // Create test data with specific action distribution
            const testMessages: readonly Message[] = [
                // Chunk 1: 3 actions (image + 2 commands)
                { role: 'user', content: { type: 'image', data: 'img1' } },
                { role: 'user', content: 'click(1,1)' },
                { role: 'user', content: 'type(hello)' },
                // Chunk 2: 2 actions  
                { role: 'user', content: { type: 'image', data: 'img2' } },
                { role: 'user', content: 'click(2,2)' },
                // Chunk 3: 1 action
                { role: 'user', content: 'click(3,3)' }
            ];
            // Total: 6 actions across 3 chunks (chunkSize=2: [3,2,1])

            const finalEvaluation = JSON.stringify({
                summary: "Final Summary",
                observations: "Action density test observations",
                score: 85,
                reasoning: "Final Reasoning",
                confidence: 0.9,
                outcomeAchievement: 80,
                processQuality: 90,
                efficiency: 85
            });

            mockChatCompletionsCreate.mockResolvedValue({
                choices: [{ message: { content: finalEvaluation } }]
            });

            // Spy on evaluateChunk to capture the context messages
            const evaluateChunkSpy = spyOn(grader, 'evaluateChunk');

            const result = await grader.grade(metaData, testMessages);

            expect(result).not.toBeNull();
            expect(evaluateChunkSpy).toHaveBeenCalledTimes(3); // 3 chunks

            // Verify action density calculations:
            // Chunk 1: 3 actions / 6 total = 50%
            // Chunk 2: 2 actions / 6 total = 33% 
            // Chunk 3: 1 action / 6 total = 17%
            const calls = evaluateChunkSpy.mock.calls;

            // Check that totalActions (6) is passed correctly to each chunk
            expect(calls[0][5]).toBe(6); // totalActions parameter (index 5) for chunk 1
            expect(calls[1][5]).toBe(6); // totalActions parameter (index 5) for chunk 2  
            expect(calls[2][5]).toBe(6); // totalActions parameter (index 5) for chunk 3

            evaluateChunkSpy.mockRestore();
        });

        it('should use low detail for image optimization', async () => {
            const imageMessage: readonly Message[] = [
                { role: 'user', content: { type: 'image', data: 'base64imagedata' } },
                { role: 'user', content: 'click(100,200)' }
            ];

            const finalEvaluation = JSON.stringify({
                summary: "Image optimization test",
                observations: "Low detail optimization working",
                score: 80,
                reasoning: "Test reasoning",
                confidence: 0.9,
                outcomeAchievement: 75,
                processQuality: 85,
                efficiency: 80
            });

            mockChatCompletionsCreate.mockResolvedValue({
                choices: [{ message: { content: finalEvaluation } }]
            });

            await grader.grade(metaData, imageMessage);

            // Verify the API call used low detail for images
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(1);
            const apiCall = mockChatCompletionsCreate.mock.calls[0][0];

            // Find the message with image content
            const messageWithImage = apiCall.messages.find((msg: any) =>
                Array.isArray(msg.content) &&
                msg.content.some((part: any) => part.type === 'image_url')
            );

            expect(messageWithImage).toBeDefined();
            const imageContent = messageWithImage.content.find((part: any) => part.type === 'image_url');
            expect(imageContent.image_url.detail).toBe('low');
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

    describe('Structured Outputs', () => {
        it('should handle structured JSON responses correctly', async () => {
            const structuredEvaluation = JSON.stringify({
                summary: "Test Summary",
                observations: "Test Observations",
                score: 75,
                reasoning: "Test Reasoning",
                confidence: 0.8,
                outcomeAchievement: 70,
                processQuality: 80,
                efficiency: 75
            });

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: structuredEvaluation } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).not.toBeNull();
            expect(result?.score).toBe(74); // Calculated: 70*0.5 + 80*0.3 + 75*0.2 = 35 + 24 + 15 = 74
            expect(result?.modelScoreRaw).toBe(75);
            expect(result?.confidence).toBe(0.8);
            expect(result?.observations).toBe("Test Observations");
        });

        it('should handle parsing failures gracefully', async () => {
            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockResolvedValue({
                    choices: [{ message: { content: 'malformed response' } }]
                });

            const result = await grader.grade(metaData, sftData);

            expect(result).toBeNull();
            expect(testLogger.logs.some(log =>
                log.level === 'error' && log.message === 'Failed to parse structured response even after repair attempt'
            )).toBe(true);
        });

        it('should repair truncated JSON responses', async () => {
            // Test with a simple case: missing closing brace
            const truncatedJson = '{"summary": "Test Summary", "observations": "Test obs", "score": 85, "reasoning": "Test reasoning", "confidence": 0.9, "outcomeAchievement": 80, "processQuality": 90, "efficiency": 85';

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: truncatedJson } }]
                });

            const result = await grader.grade(metaData, sftData);

            // Should successfully repair and parse
            expect(result).not.toBeNull();
            expect(result?.summary).toBe("Test Summary");

            // Check that repair was attempted (debug log should be present)
            const hasDebugLog = testLogger.logs.some(log =>
                log.level === 'debug' && log.message === 'Attempting JSON repair for potentially truncated response'
            );
            expect(hasDebugLog).toBe(true);
        });

        it('should use correct JSON schemas for API calls', async () => {
            const finalEvaluation = JSON.stringify({
                summary: "Final Summary",
                observations: "Final observations",
                score: 90,
                reasoning: "Final Reasoning",
                confidence: 0.95,
                outcomeAchievement: 85,
                processQuality: 95,
                efficiency: 90
            });

            mockChatCompletionsCreate
                .mockResolvedValueOnce({
                    choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                })
                .mockResolvedValueOnce({
                    choices: [{ message: { content: finalEvaluation } }]
                });

            await grader.grade(metaData, sftData);

            // Verify API calls used structured outputs
            expect(mockChatCompletionsCreate).toHaveBeenCalledTimes(2);

            // First call (chunk) should use chunk schema
            const firstCall = mockChatCompletionsCreate.mock.calls[0][0];
            expect(firstCall.response_format.type).toBe("json_schema");
            expect(firstCall.response_format.json_schema.name).toBe("chunk_evaluation");
            expect(firstCall.response_format.json_schema.strict).toBe(true);

            // Second call (final) should use final schema
            const secondCall = mockChatCompletionsCreate.mock.calls[1][0];
            expect(secondCall.response_format.type).toBe("json_schema");
            expect(secondCall.response_format.json_schema.name).toBe("final_evaluation");
            expect(secondCall.response_format.json_schema.strict).toBe(true);
        });
    });

    describe('Score Validation and Variance Detection', () => {
        it('should calculate deterministic scores from components', async () => {
            const testCases = [
                {
                    name: "Perfect scores",
                    components: { outcome: 100, process: 100, efficiency: 100 },
                    expectedScore: 100 // 100*0.5 + 100*0.3 + 100*0.2 = 100
                },
                {
                    name: "Mixed scores",
                    components: { outcome: 80, process: 60, efficiency: 40 },
                    expectedScore: 66 // 80*0.5 + 60*0.3 + 40*0.2 = 40 + 18 + 8 = 66
                },
                {
                    name: "Edge case with decimals",
                    components: { outcome: 83, process: 77, efficiency: 91 },
                    expectedScore: 83 // 83*0.5 + 77*0.3 + 91*0.2 = 41.5 + 23.1 + 18.2 = 82.8 ≈ 83
                }
            ];

            for (const testCase of testCases) {
                const evaluation = JSON.stringify({
                    summary: `Test Summary for ${testCase.name}`,
                    observations: `Test observations for ${testCase.name}`,
                    score: 999, // Intentionally wrong to test that calculated score is used
                    reasoning: "Test Reasoning",
                    confidence: 0.9,
                    outcomeAchievement: testCase.components.outcome,
                    processQuality: testCase.components.process,
                    efficiency: testCase.components.efficiency
                });

                mockChatCompletionsCreate
                    .mockResolvedValueOnce({
                        choices: [{ message: { content: JSON.stringify({ summary: "Intermediate Summary" }) } }]
                    })
                    .mockResolvedValueOnce({
                        choices: [{ message: { content: evaluation } }]
                    });

                const result = await grader.grade(metaData, sftData);

                expect(result).not.toBeNull();
                expect(result?.score).toBe(testCase.expectedScore);
                expect(result?.modelScoreRaw).toBe(999);
                expect(result?.outcomeAchievement).toBe(testCase.components.outcome);
                expect(result?.processQuality).toBe(testCase.components.process);
                expect(result?.efficiency).toBe(testCase.components.efficiency);

                // Reset mocks for next iteration
                mockChatCompletionsCreate.mockClear();
            }
        });

        it('should detect score calculation variance', async () => {
            const inconsistentEvaluation = JSON.stringify({
                summary: "Test Summary",
                observations: "Inconsistent observations",
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
            expect(result?.score).toBe(83); // Uses calculated score: 90*0.5 + 80*0.3 + 70*0.2 = 45 + 24 + 14 = 83
            expect(result?.modelScoreRaw).toBe(50); // Original model score preserved for diagnostics

            // Check score comparison logging (now logs all differences > 0)
            expect(testLogger.logs.some(log =>
                log.level === 'debug' && log.message === 'Score calculation comparison'
            )).toBe(true);
        });

        it('should clamp component scores to valid ranges', async () => {
            const extremeEvaluation = JSON.stringify({
                summary: "Test Summary",
                observations: "Extreme observations",
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
                observations: "Logged observations",
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