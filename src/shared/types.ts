export interface RawSession {
    _id: { $oid: string };
    address: string;
    challenge: string;
    prompt: string;
    category: string;
    vm_ip: string;
    vm_port: number;
    vm_password: string;
    vm_region: string;
    vm_credentials: {
        guacToken: string;
        guacConnectionId: string;
        guacClientId: string;
        username: string;
        password: string;
    };
    status: string;
    created_at: { $date: string };
}

export interface RawEventsFile {
    session_id: string;
    challenge: string;
    category: string;
    transaction_signature: string;
    timestamp?: number;
    events: RawEvent[];
}

export interface RawEvent {
    type: 'quest' | 'hint';
    timestamp: number;
    message?: string;
    frame?: number;
}

export interface GuacInstruction {
    opcode: string;
    args: string[];
    timestamp: number;
}

export interface Message {
    role: 'user' | 'assistant';
    content: string | {
        type: 'image';
        data: string;
    };
    timestamp: number;
}

export interface ProcessedEvent {
    type: 'keydown' | 'keyup' | 'mousedown' | 'mouseup' | 'mousedrag' | 'mousewheel' | 'frame' | 'quest' | 'hint' | 'mouseclick' | 'type' | 'hotkey' | 'dense_caption' | 'state_transition' | 'structured_data' | 'reasoning' | 'app_focus';
    timestamp: number;
    data: {
        delta?: number;
        x?: number;
        y?: number;
        keyCode?: number;
        text?: string;
        frame?: string;
        beforeFrame?: string;
        afterFrame?: string;
        message?: string;
        coordinates?: Array<{
            time: number;  // relative ms from start of drag
            x: number;
            y: number;
        }>;
        focused_app?: string;
        available_apps?: string[];
        all_windows?: Array<{
            name: string;
            role?: string;
            children?: any[];
        }>;
    };
}

export interface PipelineStage<T, U> {
    process(input: T): Promise<U>;
}

export interface PipelineConfig {
    dataDir: string;
    outputDir: string;
    sessionIds: string[];
    extractors: PipelineStage<string, ProcessedEvent[]>[];
    augmenters: PipelineStage<ProcessedEvent[], ProcessedEvent[]>[];
}
