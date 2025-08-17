/**
 * Utility function for creating delays in async operations
 * @param ms - Number of milliseconds to wait
 * @returns Promise that resolves after the specified delay
 */
export const sleep = (ms: number): Promise<void> => new Promise(res => setTimeout(res, ms));
