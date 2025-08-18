export class RateLimiter {
    private tokens: number;
    private lastRefill: number;
    private readonly maxTokens: number;
    private readonly refillRate: number; // tokens per second
    private readonly queue: Array<{ resolve: () => void; timestamp: number }> = [];

    constructor(maxTokens: number = 10, refillRate: number = 2) {
        this.maxTokens = maxTokens;
        this.refillRate = refillRate;
        this.tokens = maxTokens;
        this.lastRefill = Date.now();
    }

    async acquire(): Promise<void> {
        return new Promise((resolve) => {
            this.queue.push({ resolve, timestamp: Date.now() });
            this.processQueue();
        });
    }

    private processQueue(): void {
        this.refillTokens();

        while (this.queue.length > 0 && this.tokens > 0) {
            const request = this.queue.shift()!;
            this.tokens--;
            request.resolve();
        }

        // Schedule next processing if there are waiting requests
        if (this.queue.length > 0) {
            const nextRefillTime = Math.max(0, 1000 / this.refillRate);
            setTimeout(() => this.processQueue(), nextRefillTime);
        }
    }

    private refillTokens(): void {
        const now = Date.now();
        const timePassed = (now - this.lastRefill) / 1000;
        const tokensToAdd = Math.floor(timePassed * this.refillRate);

        if (tokensToAdd > 0) {
            this.tokens = Math.min(this.maxTokens, this.tokens + tokensToAdd);
            this.lastRefill = now;
        }
    }

    getStats(): { tokens: number; queueLength: number } {
        this.refillTokens();
        return {
            tokens: this.tokens,
            queueLength: this.queue.length
        };
    }
}
