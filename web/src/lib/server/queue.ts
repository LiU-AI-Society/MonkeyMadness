import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import * as db from './db';
import { config } from './config';
import { broadcast, snapshot } from './live';
import type { ClassStat } from '$lib/types';

export const uploadDir = (id: number) => path.join(config.dataDir, 'uploads', String(id));

type QueueState = { waiting: number[]; running: number; started: boolean };
const g = globalThis as unknown as { __monkeyQueue?: QueueState };
const q = (g.__monkeyQueue ??= { waiting: [], running: 0, started: false });

if (!q.started) {
	q.started = true;
	q.waiting.push(...db.pendingIds());
	setTimeout(pump, 0);
}

export function enqueue(id: number) {
	q.waiting.push(id);
	pump();
}

export function queueLength(): number {
	return q.waiting.length;
}

export function queuePosition(id: number): number {
	return q.waiting.indexOf(id) + 1;
}

function pump() {
	while (q.running < config.concurrency && q.waiting.length) {
		const id = q.waiting.shift()!;
		q.running++;
		score(id)
			.catch((err) => console.error(`[queue] submission ${id} crashed the scorer`, err))
			.finally(() => {
				q.running--;
				fs.rmSync(uploadDir(id), { recursive: true, force: true });
				pump();
			});
	}
}

type WorkerResult =
	| { ok: true; accuracy: number; precision: number; recall: number; f1_score: number; per_class?: ClassStat[] }
	| { ok: false; error?: string };

async function score(id: number) {
	const before = snapshot().leaderboard;
	const sub = db.allSubmissions(null).find((s) => s.id === id);
	if (!sub) return; // deleted by an admin while waiting
	db.markScoring(id);
	broadcast();

	const dir = uploadDir(id);
	const model = fs.existsSync(dir) ? fs.readdirSync(dir).find((f) => f.toLowerCase().endsWith('.onnx')) : undefined;
	if (!model) {
		db.markFailed(id, 'Upload went missing before scoring, please submit again.');
		broadcast({ kind: 'failed', team: sub.team });
		return;
	}

	const result = config.mockScorer ? await mockScore() : await runWorker(id, path.join(dir, model));
	if (!result.ok) {
		db.markFailed(id, friendlyError(result.error ?? ''));
		broadcast({ kind: 'failed', team: sub.team });
		return;
	}

	db.markDone(id, result);
	const after = snapshot();
	const mine = after.submissions.find((s) => s.id === id)!;
	const prevBest = before.find((r) => r.team.toLowerCase() === sub.team.toLowerCase());
	const row = after.leaderboard.find((r) => r.team.toLowerCase() === sub.team.toLowerCase());
	broadcast({
		kind: 'scored',
		team: sub.team,
		accuracy: mine.accuracy,
		delta: mine.accuracy !== null && prevBest ? mine.accuracy - prevBest.accuracy : null,
		rank: mine.hidden ? null : (row?.rank ?? null),
		newLeader: !mine.hidden && row?.rank === 1 && before[0]?.team.toLowerCase() !== sub.team.toLowerCase()
	});
}

function runWorker(id: number, modelPath: string): Promise<WorkerResult> {
	return new Promise((resolve) => {
		const args = [
			config.worker,
			'--model_path', modelPath,
			'--image_dir', config.testDir,
			'--gold', config.goldCsv,
			'--labels_path', config.labelsTxt,
			'--predictions_out', path.join(path.dirname(modelPath), 'predictions.csv')
		];
		const child = spawn(config.python, args, { cwd: config.repoRoot, stdio: ['ignore', 'pipe', 'pipe'] });
		let stdout = '';
		let stderr = '';
		child.stdout.on('data', (d) => (stdout += d));
		child.stderr.on('data', (d) => (stderr += d));
		let timedOut = false;
		const timer = setTimeout(() => {
			timedOut = true;
			child.kill('SIGKILL');
		}, config.timeoutSeconds * 1000);

		const finish = (code: number | null, spawnError?: Error) => {
			clearTimeout(timer);
			if (timedOut) return resolve({ ok: false, error: '__timeout__' });
			const parsed = parseWorkerOutput(stdout);
			if (!parsed?.ok) {
				// Participants only see the friendly message; log the real detail for organizers.
				console.warn(`[queue] scoring failed for submission ${id} (exit ${code})`, spawnError ?? '', `\nstdout:\n${stdout}\nstderr:\n${stderr}`);
			}
			resolve(parsed ?? { ok: false, error: stderr });
		};
		child.on('error', (err) => finish(null, err));
		child.on('close', (code) => finish(code));
	});
}

/** worker.py prints one JSON line; scan from the end in case anything else leaked onto stdout. */
function parseWorkerOutput(stdout: string): WorkerResult | null {
	for (const line of stdout.trim().split('\n').reverse()) {
		try {
			return JSON.parse(line.trim());
		} catch {
			continue;
		}
	}
	return null;
}

function friendlyError(detail: string): string {
	if (detail === '__timeout__') return `Your model took longer than ${config.timeoutSeconds}s to run and was stopped.`;
	if (detail.includes('should be stored in'))
		return 'Your ONNX file references external weight data that was not uploaded (usually <yourmodel>.onnx.data). Select both files together, or re-export as a single .onnx file.';
	return "Could not score your model. Make sure it's a valid ONNX classifier with the expected input shape.";
}

async function mockScore(): Promise<WorkerResult> {
	await new Promise((r) => setTimeout(r, 2000 + Math.random() * 4000));
	const accuracy = 0.3 + Math.random() * 0.6;
	const f1 = Math.max(0, accuracy - Math.random() * 0.05);
	const labels = mockLabels();
	const jitter = (x: number) => Math.max(0, Math.min(1, x + (Math.random() - 0.5) * 0.5));
	const per_class = labels.map((label, i) => ({
		label,
		precision: jitter(accuracy),
		recall: jitter(accuracy),
		f1: jitter(f1),
		support: 10,
		confused_with: labels[(i + 1 + Math.floor(Math.random() * (labels.length - 1))) % labels.length],
		confused_count: Math.ceil(Math.random() * 4)
	}));
	return { ok: true, accuracy, precision: f1 + 0.01, recall: f1 - 0.01, f1_score: f1, per_class };
}

/** Class names from monkey_labels.txt (column 3), sorted the way the evaluator sorts them. */
function mockLabels(): string[] {
	try {
		return fs
			.readFileSync(config.labelsTxt, 'utf8')
			.split('\n')
			.slice(1)
			.map((line) => line.split(',')[2]?.trim())
			.filter((x): x is string => !!x)
			.sort();
	} catch {
		return Array.from({ length: 10 }, (_, i) => `class_${i}`);
	}
}
