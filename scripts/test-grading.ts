#!/usr/bin/env bun

/**
 * Script to run only grading-related tests
 * Usage: bun run scripts/test-grading.ts [--integration] [--unit] [--all]
 */

import { parseArgs } from 'util';

const { values } = parseArgs({
    args: Bun.argv.slice(2),
    options: {
        integration: {
            type: 'boolean',
            default: false,
            short: 'i'
        },
        unit: {
            type: 'boolean',
            default: false,
            short: 'u'
        },
        all: {
            type: 'boolean',
            default: false,
            short: 'a'
        },
        help: {
            type: 'boolean',
            default: false,
            short: 'h'
        }
    },
    allowPositionals: false
});

if (values.help) {
    console.log(`
🧪 Grading Test Runner

Usage: bun run scripts/test-grading.ts [options]

Options:
  -u, --unit         Run unit tests only (mocked)
  -i, --integration  Run integration tests only (real API)  
  -a, --all          Run both unit and integration tests
  -h, --help         Show this help message

Examples:
  bun run scripts/test-grading.ts --unit
  bun run scripts/test-grading.ts --integration  
  bun run scripts/test-grading.ts --all

Environment:
  OPENAI_API_KEY     Required for integration tests
`);
    process.exit(0);
}

async function runTests() {
    const testFiles: string[] = [];

    // Determine which tests to run
    if (values.all) {
        testFiles.push('tests/grading.test.ts', 'tests/grading.integration.test.ts');
    } else if (values.unit) {
        testFiles.push('tests/grading.test.ts');
    } else if (values.integration) {
        testFiles.push('tests/grading.integration.test.ts');
    } else {
        // Default: run unit tests
        testFiles.push('tests/grading.test.ts');
    }

    console.log('🚀 Starting grading tests...\n');

    if (values.integration || values.all) {
        if (!process.env.OPENAI_API_KEY) {
            console.log('⚠️  WARNING: OPENAI_API_KEY not set - integration tests will be skipped');
        } else {
            console.log('✅ OPENAI_API_KEY found - integration tests will run');
        }
        console.log('');
    }

    for (const testFile of testFiles) {
        console.log(`📝 Running: ${testFile}`);

        const proc = Bun.spawn(['bun', 'test', testFile, '--timeout', '180000'], {
            stdio: ['inherit', 'inherit', 'inherit'],
            env: process.env
        });

        const exitCode = await proc.exited;

        if (exitCode !== 0) {
            console.error(`❌ Tests failed in ${testFile}`);
            process.exit(exitCode);
        }

        console.log(`✅ Tests passed in ${testFile}\n`);
    }

    console.log('🎉 All grading tests completed successfully!');
}

// Run tests - this script is designed to be executed directly
runTests().catch((error) => {
    console.error('❌ Test runner failed:', error);
    process.exit(1);
});
