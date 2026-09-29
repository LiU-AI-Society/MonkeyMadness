import fs from 'node:fs';
import path from 'node:path';
import { env } from '$env/dynamic/private';

// Same env var names as the Flask leaderboard, so an existing setup carries over.
const REPO_ROOT = path.resolve(env.MONKEY_REPO_ROOT ?? path.join(process.cwd(), '..'));

const VENV_PYTHON = path.join(REPO_ROOT, 'leaderboard', 'venv', 'bin', 'python');

export const config = {
	repoRoot: REPO_ROOT,
	// leaderboard/venv is where the README sets up the scoring dependencies.
	python: env.PYTHON ?? (fs.existsSync(VENV_PYTHON) ? VENV_PYTHON : 'python3'),
	worker: path.join(REPO_ROOT, 'leaderboard', 'worker.py'),
	testDir: env.LEADERBOARD_TEST_DIR ?? path.join(REPO_ROOT, 'hidden_test'),
	goldCsv: env.LEADERBOARD_GOLD_CSV ?? path.join(REPO_ROOT, 'gold_labels.csv'),
	labelsTxt: env.LEADERBOARD_LABELS_TXT ?? path.join(REPO_ROOT, 'Monkey', 'monkey_labels.txt'),
	dataDir: path.resolve(env.DATA_DIR ?? 'data'),
	maxUploadMb: Number(env.LEADERBOARD_MAX_UPLOAD_MB ?? 50),
	cooldownSeconds: Number(env.LEADERBOARD_COOLDOWN_SECONDS ?? 60),
	timeoutSeconds: Number(env.LEADERBOARD_TIMEOUT_SECONDS ?? 120),
	concurrency: Math.max(1, Number(env.SCORING_CONCURRENCY ?? 2)),
	/** Refuse new uploads while this many are waiting, so a flood can't bury everyone else. */
	maxQueue: Number(env.MAX_QUEUE ?? 30),
	/** Per client IP; generous because a venue network may put every team behind one IP. 0 disables. */
	ipLimitPerMinute: Number(env.IP_LIMIT_PER_MINUTE ?? 10),
	adminPassword: env.ADMIN_PASSWORD ?? '',
	/** SCORER=mock returns random scores without Python: for UI rehearsal only. */
	mockScorer: env.SCORER === 'mock',
	maxTeamNameLen: 64
};
