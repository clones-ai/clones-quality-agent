import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';

async function runGradingProcess(sessionDir: string, extraArgs: string[]) {
    console.log(`Starting Clones Quality Agent on directory: ${sessionDir}`);

    // 1. Verify the directory exists
    try {
        await fs.access(sessionDir);
    } catch (error) {
        console.error(`Error: Directory not found at '${sessionDir}'`);
        process.exit(1);
    }

    // 2. Define the command to be executed
    const cqaExecutable = 'bun'; // We use bun to run the ts file
    const cqaScriptPath = path.resolve(__dirname, '../src/index.ts');
    const args = ['run', cqaScriptPath, '-f', 'desktop', '-i', sessionDir, '--grade', ...extraArgs];

    console.log(`Executing: ${cqaExecutable} ${args.join(' ')}`);

    // 3. Spawn the child process
    await new Promise<void>((resolve, reject) => {
        const pipeline = spawn(cqaExecutable, args, {
            stdio: 'pipe', // Use pipe to capture stdio
        });

        let stdout = '';
        let stderr = '';

        // 4. Capture and log stdout/stderr in real-time
        pipeline.stdout.on('data', (data) => {
            const output = data.toString();
            stdout += output;
            console.log(output.trim()); // Log directly to see progress
        });

        pipeline.stderr.on('data', (data) => {
            const output = data.toString();
            stderr += output;
            console.error(output.trim()); // Log directly to see errors
        });

        // 5. Handle process exit
        pipeline.on('close', (code: number) => {
            if (code === 0) {
                console.log('\n✅ Clones Quality Agent finished successfully.');
                resolve();
            } else {
                console.error('\n❌ Clones Quality Agent failed.');
                // The full stdout/stderr is already logged above, but we can summarize here
                const error = new Error(`Clones Quality Agent exited with code ${code}.`);
                console.error(error.message);
                reject(error);
            }
        });

        // 6. Handle spawn errors
        pipeline.on('error', (err) => {
            console.error('❌ Failed to spawn Clones Quality Agent process.', err);
            reject(err);
        });
    });

    // 7. Verify and read output files (similar to backend)
    try {
        const scoresPath = path.join(sessionDir, 'scores.json');
        const scoresContent = await fs.readFile(scoresPath, 'utf8');
        const gradeResult = JSON.parse(scoresContent);

        console.log('\n--- Grading Results (scores.json) ---');
        console.log(JSON.stringify(gradeResult, null, 2));

        const metricsPath = path.join(sessionDir, 'metrics.json');
        const metricsContent = await fs.readFile(metricsPath, 'utf8');
        const metricsResult = JSON.parse(metricsContent);

        console.log('\n--- Metrics (metrics.json) ---');
        console.log(JSON.stringify(metricsResult, null, 2));
    } catch (error) {
        console.error('\nCould not read output files (scores.json/metrics.json).', error);
    }
}

// --- Script Entry Point ---
if (process.argv.length < 3) {
    console.error('Usage: bun run scripts/run-grading.ts <path_to_session_directory> [--model model_name] [--evaluation-model model_name]');
    process.exit(1);
}

// The first argument without a dash is the directory
const dirIndex = process.argv.slice(2).findIndex(arg => !arg.startsWith('--'));
if (dirIndex === -1) {
    console.error('Error: Session directory path is required.');
    console.error('Usage: bun run scripts/run-grading.ts <path_to_session_directory> [--model model_name] [--evaluation-model model_name]');
    process.exit(1);
}

const rawArgs = process.argv.slice(2);
const sessionDirectory = path.resolve(rawArgs[dirIndex]);
const extraArgs = rawArgs.filter((_, i) => i !== dirIndex);

runGradingProcess(sessionDirectory, extraArgs).catch(() => process.exit(1));
