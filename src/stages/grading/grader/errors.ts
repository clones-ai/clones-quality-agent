export class GraderError extends Error {
    constructor(message: string, public readonly cause?: unknown) {
        super(message);
        this.name = this.constructor.name;
    }
}

export class TimeoutError extends GraderError {
    constructor(message: string = "Request timed out", cause?: unknown) {
        super(message, cause);
    }
}

export class PermanentError extends GraderError {
    constructor(message: string, public readonly statusCode?: number, cause?: unknown) {
        super(message, cause);
    }
}

export class TransientError extends GraderError {
    constructor(
        message: string,
        public readonly statusCode?: number,
        public readonly retryAfter?: number,
        cause?: unknown
    ) {
        super(message, cause);
    }
}
