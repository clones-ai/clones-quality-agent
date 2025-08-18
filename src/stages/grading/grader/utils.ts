import { GraderError, TimeoutError, PermanentError, TransientError } from "./errors";

/** Promise-based sleep utility. */
export const sleep = (ms: number) => new Promise<void>((res) => setTimeout(res, ms));

/** Numeric clamp with finite check. */
export const clamp = (v: number, min: number, max: number) =>
    Number.isFinite(v) ? Math.min(max, Math.max(min, v)) : min;

/** Comprehensive data sanitization for logging and security. */
export const redact = (s: string) => {
    if (!s || typeof s !== 'string') return "";

    return s
        // Email addresses
        .replace(/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g, "[EMAIL]")

        // API Keys - OpenAI
        .replace(/sk-[A-Za-z0-9]{20,}/g, "[OPENAI-KEY]")
        .replace(/pk-[A-Za-z0-9]{20,}/g, "[OPENAI-PUB-KEY]")

        // JWT tokens (process first to avoid conflicts with other patterns)
        .replace(/eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+/g, "[JWT-TOKEN]")

        // API Keys - AWS
        .replace(/AKIA[0-9A-Z]{16}/g, "[AWS-ACCESS-KEY]")
        .replace(/(?:^|\s)([A-Za-z0-9/+=]{40})(?:\s|$)/g, (match, key) => {
            // AWS Secret Access Key pattern (40 chars base64-like, standalone)
            if (/^[A-Za-z0-9/+=]{40}$/.test(key)) return match.replace(key, "[AWS-SECRET-KEY]");
            return match;
        })

        // API Keys - Google
        .replace(/AIza[0-9A-Za-z\-_]{35}/g, "[GOOGLE-API-KEY]")
        .replace(/ya29\.[0-9A-Za-z\-_]+/g, "[GOOGLE-OAUTH-TOKEN]")

        // API Keys - GitHub
        .replace(/ghp_[A-Za-z0-9]{36}/g, "[GITHUB-PAT]")
        .replace(/gho_[A-Za-z0-9]{36}/g, "[GITHUB-OAUTH]")
        .replace(/ghu_[A-Za-z0-9]{36}/g, "[GITHUB-USER-TOKEN]")
        .replace(/ghs_[A-Za-z0-9]{36}/g, "[GITHUB-SERVER-TOKEN]")
        .replace(/ghr_[A-Za-z0-9]{36}/g, "[GITHUB-REFRESH-TOKEN]")

        // API Keys - Stripe
        .replace(/sk_live_[0-9a-zA-Z]{24,}/g, "[STRIPE-SECRET-LIVE]")
        .replace(/sk_test_[0-9a-zA-Z]{24,}/g, "[STRIPE-SECRET-TEST]")
        .replace(/pk_live_[0-9a-zA-Z]{24,}/g, "[STRIPE-PUBLIC-LIVE]")
        .replace(/pk_test_[0-9a-zA-Z]{24,}/g, "[STRIPE-PUBLIC-TEST]")

        // API Keys - Slack
        .replace(/xoxb-[0-9]{11,13}-[0-9]{11,13}-[a-zA-Z0-9]{24}/g, "[SLACK-BOT-TOKEN]")
        .replace(/xoxp-[0-9]{11,13}-[0-9]{11,13}-[0-9]{11,13}-[a-zA-Z0-9]{32}/g, "[SLACK-USER-TOKEN]")

        // OAuth and Bearer tokens
        .replace(/Bearer\s+[A-Za-z0-9\-_\.]+/gi, "[BEARER-TOKEN]")
        .replace(/OAuth\s+[A-Za-z0-9\-_\.]+/gi, "[OAUTH-TOKEN]")

        // PEM private keys
        .replace(/-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----/gi, "[PEM-PRIVATE-KEY]")
        .replace(/-----BEGIN\s+ENCRYPTED\s+PRIVATE\s+KEY-----[\s\S]*?-----END\s+ENCRYPTED\s+PRIVATE\s+KEY-----/gi, "[PEM-ENCRYPTED-KEY]")
        .replace(/-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----[\s\S]*?-----END\s+OPENSSH\s+PRIVATE\s+KEY-----/gi, "[OPENSSH-PRIVATE-KEY]")

        // SSH private keys
        .replace(/ssh-rsa\s+[A-Za-z0-9+/=]+/g, "[SSH-PUBLIC-KEY]")
        .replace(/ssh-ed25519\s+[A-Za-z0-9+/=]+/g, "[SSH-ED25519-KEY]")

        // Database connection strings
        .replace(/(?:mongodb|mysql|postgresql|postgres):\/\/[^\s]+/gi, "[DATABASE-URL]")

        // Generic secrets (common patterns)
        .replace(/(?:password|passwd|pwd|secret|token|key)\s*[:=]\s*['"]*[A-Za-z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?]+['"]*(?:\s|$)/gi,
            (match) => match.replace(/['"]*[A-Za-z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?]+['"]*/, "[REDACTED]"))

        // Base64 patterns (potential sensitive data)
        .replace(/data:image\/[^;]+;base64,[A-Za-z0-9+/=]+/g, "[BASE64-IMAGE]")
        .replace(/[A-Za-z0-9+/]{50,}={0,2}/g, "[BASE64-DATA]")

        // IP addresses (optional - might be too aggressive)
        .replace(/\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b/g, "[IP-ADDRESS]")

        // Remove control characters except common whitespace
        .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/g, "")

        // Truncate for log safety
        .slice(0, 200);
};

/** Sanitize cropInfo and other user-provided strings. */
export const sanitizeUserInput = (input: string): string => {
    if (!input || typeof input !== 'string') return "";

    return input
        // Remove control characters except common whitespace (\t, \n, \r)
        .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/g, "")
        // Remove potential injection patterns
        .replace(/<script[^>]*>.*?<\/script>/gi, "")
        .replace(/javascript:/gi, "")
        .replace(/data:/gi, "")
        // Normalize whitespace
        .replace(/\s+/g, " ")
        .trim()
        // Limit length
        .slice(0, 500);
};

/** Safe logging helper that prevents data leakage. */
export const safeLog = (data: unknown): string => {
    if (data === null || data === undefined) return "[NULL]";
    if (typeof data === 'string') return redact(data);
    if (typeof data === 'number' || typeof data === 'boolean') return String(data);

    try {
        const stringified = JSON.stringify(data);
        // Don't log large objects or potential sensitive content
        if (stringified.length > 300) return "[LARGE-OBJECT]";
        return redact(stringified);
    } catch {
        return "[UNSTRINGIFIABLE]";
    }
};

/**
 * Extract JSON from a ```json fenced block or by scanning for balanced braces.
 * Returns null if no plausible JSON is found.
 */
export function safeExtractJson(text: string): string | null {
    if (!text) return null;
    const fence = text.match(/```json\s*([\s\S]*?)```/i);
    if (fence) return fence[1];

    let depth = 0;
    let start = -1;
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (ch === "{") {
            if (depth++ === 0) start = i;
        } else if (ch === "}") {
            if (--depth === 0 && start !== -1) return text.slice(start, i + 1);
        }
    }
    return null;
}

/**
 * Analyze an error and return the appropriate typed error.
 * Handles HTTP status codes, timeouts, and Retry-After headers.
 */
export function classifyError(error: unknown): GraderError {
    // Handle AbortController timeout
    if (error instanceof Error && (error.name === 'AbortError' || error.message.includes('aborted'))) {
        return new TimeoutError("Request timed out", error);
    }

    // Handle OpenAI SDK errors
    if (error && typeof error === 'object' && 'status' in error) {
        const status = (error as any).status;
        const message = (error as any).message || `HTTP ${status} error`;

        // Extract Retry-After header if present
        let retryAfter: number | undefined;
        if ('headers' in error && error.headers) {
            const headers = error.headers as any;
            const retryAfterHeader = headers['retry-after'] || headers['Retry-After'];
            if (retryAfterHeader) {
                const parsed = parseInt(retryAfterHeader, 10);
                if (!isNaN(parsed)) {
                    retryAfter = parsed * 1000; // Convert seconds to milliseconds
                }
            }
        }

        if (status === 429 || (status >= 500 && status < 600)) {
            // Transient errors: rate limits and server errors
            return new TransientError(message, status, retryAfter, error);
        } else if (status >= 400 && status < 500) {
            // Permanent errors: client errors (except 429)
            return new PermanentError(message, status, error);
        }
    }

    // Handle network errors and other transient issues
    if (error instanceof Error) {
        const message = error.message.toLowerCase();
        if (message.includes('network') || message.includes('connection') ||
            message.includes('timeout') || message.includes('econnreset')) {
            return new TransientError(error.message, undefined, undefined, error);
        }
    }

    // Default to transient for unknown errors (conservative approach)
    const message = error instanceof Error ? error.message : String(error);
    return new TransientError(`Unknown error: ${message}`, undefined, undefined, error);
}
