import fs from 'node:fs';
import path from 'node:path';
import Database from 'better-sqlite3';
import { config } from './config';
import type { ClassStat, Status, Submission, TeamRow } from '$lib/types';

fs.mkdirSync(config.dataDir, { recursive: true });

// Survive Vite HMR in dev without opening the file twice.
const g = globalThis as unknown as { __monkeyDb?: Database.Database };
const db = (g.__monkeyDb ??= new Database(path.join(config.dataDir, 'monkeymadness.db')));
db.pragma('journal_mode = WAL');
db.exec(`
CREATE TABLE IF NOT EXISTS submissions (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	team TEXT NOT NULL COLLATE NOCASE,
	status TEXT NOT NULL,
	accuracy REAL,
	precision_macro REAL,
	recall_macro REAL,
	f1_macro REAL,
	error TEXT,
	created_at TEXT NOT NULL,
	scored_at TEXT
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
`);
// Added after the first release: per-class stats as JSON, fetched on demand (kept out of live snapshots).
const columns = db.prepare('PRAGMA table_info(submissions)').all() as { name: string }[];
if (!columns.some((c) => c.name === 'per_class')) db.exec('ALTER TABLE submissions ADD COLUMN per_class TEXT');
// The upgrade shop was removed; drop its columns from databases created before that.
for (const old of ['upgrades', 'cost']) if (columns.some((c) => c.name === old)) db.exec(`ALTER TABLE submissions DROP COLUMN ${old}`);

type Row = {
	id: number;
	team: string;
	status: Status;
	accuracy: number | null;
	precision_macro: number | null;
	recall_macro: number | null;
	f1_macro: number | null;
	error: string | null;
	created_at: string;
	scored_at: string | null;
};

const now = () => new Date().toISOString();

function toSubmission(row: Row, frozenAt: string | null): Submission {
	const hidden = !!frozenAt && row.status === 'done' && !!row.scored_at && row.scored_at > frozenAt;
	return {
		id: row.id,
		team: row.team,
		status: row.status,
		accuracy: hidden ? null : row.accuracy,
		precision: hidden ? null : row.precision_macro,
		recall: hidden ? null : row.recall_macro,
		f1: hidden ? null : row.f1_macro,
		error: row.error,
		createdAt: row.created_at,
		scoredAt: row.scored_at,
		hidden
	};
}

export function getSetting(key: string): string | null {
	const row = db.prepare('SELECT value FROM settings WHERE key = ?').get(key) as { value: string } | undefined;
	return row?.value ?? null;
}

export function setSetting(key: string, value: string | null) {
	if (value === null) db.prepare('DELETE FROM settings WHERE key = ?').run(key);
	else db.prepare('INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value').run(key, value);
}

export function createSubmission(team: string): number {
	const info = db.prepare("INSERT INTO submissions (team, status, created_at) VALUES (?, 'queued', ?)").run(team, now());
	return Number(info.lastInsertRowid);
}

export function markScoring(id: number) {
	db.prepare("UPDATE submissions SET status = 'scoring' WHERE id = ?").run(id);
}

export function markDone(
	id: number,
	s: { accuracy: number; precision: number; recall: number; f1_score: number; per_class?: ClassStat[] }
) {
	db.prepare(
		"UPDATE submissions SET status = 'done', accuracy = ?, precision_macro = ?, recall_macro = ?, f1_macro = ?, per_class = ?, scored_at = ? WHERE id = ?"
	).run(s.accuracy, s.precision, s.recall, s.f1_score, s.per_class ? JSON.stringify(s.per_class) : null, now(), id);
}

export function perClass(id: number): ClassStat[] | null {
	const row = db.prepare('SELECT per_class FROM submissions WHERE id = ?').get(id) as { per_class: string | null } | undefined;
	return row?.per_class ? JSON.parse(row.per_class) : null;
}

export function markFailed(id: number, error: string) {
	db.prepare("UPDATE submissions SET status = 'failed', error = ?, scored_at = ? WHERE id = ?").run(error, now(), id);
}

export function deleteSubmission(id: number) {
	db.prepare('DELETE FROM submissions WHERE id = ?').run(id);
}

export function deleteAll() {
	db.exec('DELETE FROM submissions');
}

/** Submissions a restart interrupted: their uploads are still on disk, so re-run them. */
export function pendingIds(): number[] {
	return (db.prepare("SELECT id FROM submissions WHERE status IN ('queued', 'scoring') ORDER BY id").all() as { id: number }[]).map(
		(r) => r.id
	);
}

export function lastSubmission(team: string): { createdAt: string; status: Status } | null {
	const row = db.prepare('SELECT created_at, status FROM submissions WHERE team = ? ORDER BY id DESC LIMIT 1').get(team) as
		| { created_at: string; status: Status }
		| undefined;
	return row ? { createdAt: row.created_at, status: row.status } : null;
}

/** `frozenAt` null means the full, unfiltered view (admin, or not frozen). */
export function allSubmissions(frozenAt: string | null): Submission[] {
	const rows = db.prepare('SELECT * FROM submissions ORDER BY id').all() as Row[];
	return rows.map((r) => toSubmission(r, frozenAt));
}

/** Each team's best visible submission, by accuracy then macro F1 (same as the Flask board). */
export function leaderboard(submissions: Submission[]): TeamRow[] {
	const byTeam = new Map<string, { best: Submission | null; count: number; name: string }>();
	for (const s of submissions) {
		const key = s.team.toLowerCase();
		const entry = byTeam.get(key) ?? { best: null, count: 0, name: s.team };
		entry.count++;
		if (s.status === 'done' && s.accuracy !== null && s.f1 !== null) {
			const b = entry.best;
			if (!b || s.accuracy > b.accuracy! || (s.accuracy === b.accuracy && s.f1 > b.f1!)) entry.best = s;
		}
		byTeam.set(key, entry);
	}
	const rows = [...byTeam.values()]
		.filter((e) => e.best)
		.map((e) => ({ team: e.best!.team, submissions: e.count, best: e.best! }))
		.sort((a, b) => b.best.accuracy! - a.best.accuracy! || b.best.f1! - a.best.f1! || a.best.id - b.best.id);
	return rows.map((r, i) => ({
		rank: i + 1,
		team: r.team,
		accuracy: r.best.accuracy!,
		f1: r.best.f1!,
		precision: r.best.precision!,
		recall: r.best.recall!,
		submissions: r.submissions,
		best: r.best
	}));
}
