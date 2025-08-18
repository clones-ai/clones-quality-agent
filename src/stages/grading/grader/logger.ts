import { GraderLogger } from "./types";
import { redact, safeLog } from "./utils";

export class DefaultLogger implements GraderLogger {
    debug(msg: string, err?: Error, meta?: Record<string, unknown>) {
        // Intentionally quiet in production; uncomment for deep troubleshooting.
        // console.debug(`[grader][debug] ${msg}`, this.safeMeta(meta), err ? redact(err.message) : "");
    }
    info(msg: string, err?: Error, meta?: Record<string, unknown>) {
        console.log(`[grader][info] ${msg}`, this.safeMeta(meta));
        if (err) console.log(`[grader][info][err]`, redact(err.message));
    }
    warn(msg: string, err?: Error, meta?: Record<string, unknown>) {
        console.warn(`[grader][warn] ${msg}`, this.safeMeta(meta));
        if (err) console.warn(`[grader][warn][err]`, redact(err.message));
    }
    error(msg: string, err?: Error, meta?: Record<string, unknown>) {
        console.error(`[grader][error] ${msg}`, this.safeMeta(meta));
        if (err) console.error(`[grader][error][err]`, redact(err.message));
    }

    private safeMeta(meta?: Record<string, unknown>): Record<string, string> {
        if (!meta) return {};

        const safe: Record<string, string> = {};
        for (const [key, value] of Object.entries(meta)) {
            // Skip potentially sensitive keys
            if (key.toLowerCase().includes('content') ||
                key.toLowerCase().includes('data') ||
                key.toLowerCase().includes('response') ||
                key.toLowerCase().includes('message')) {
                safe[key] = "[REDACTED]";
            } else {
                safe[key] = safeLog(value);
            }
        }
        return safe;
    }
}
