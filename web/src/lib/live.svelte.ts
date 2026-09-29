import { resolve } from '$app/paths';
import type { LiveEvent, Snapshot, Submission } from './types';

/** Live view of the leaderboard, kept current over Server-Sent Events. */
export class Live {
	snapshot = $state<Snapshot | null>(null);
	connected = $state(false);
	/** Server clock minus local clock, so countdowns agree across screens. */
	clockOffset = 0;
	#source: EventSource | null = null;
	#listeners = new Set<(e: LiveEvent) => void>();

	connect() {
		this.#source = new EventSource(resolve('/api/stream'));
		this.#source.onopen = () => (this.connected = true);
		this.#source.onerror = () => (this.connected = false); // EventSource retries on its own
		this.#source.addEventListener('snapshot', (e) => {
			const snap: Snapshot = JSON.parse(e.data);
			this.clockOffset = Date.parse(snap.serverTime) - Date.now();
			this.snapshot = snap;
			this.connected = true;
		});
		this.#source.addEventListener('live', (e) => {
			const event: LiveEvent = JSON.parse(e.data);
			for (const fn of this.#listeners) fn(event);
		});
		return () => this.#source?.close();
	}

	onEvent(fn: (e: LiveEvent) => void) {
		this.#listeners.add(fn);
		return () => this.#listeners.delete(fn);
	}
}

/** One shared connection for the whole app; the root layout connects it. */
export const live = new Live();

export const pct = (x: number | null | undefined, digits = 1) => (x == null ? '—' : `${(x * 100).toFixed(digits)}%`);

export function timeAgo(iso: string, now: number) {
	const s = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
	if (s < 60) return `${s}s ago`;
	if (s < 3600) return `${Math.floor(s / 60)}m ago`;
	return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m ago`;
}

/**
 * For each scored submission: how much it beat (or missed) the team's best *before* it.
 * null for a team's first score or when the score is hidden.
 */
export function improvementDeltas(submissions: Submission[]): Map<number, number | null> {
	const best = new Map<string, number>();
	const out = new Map<number, number | null>();
	for (const s of submissions) {
		if (s.status !== 'done' || s.accuracy === null) continue;
		const key = s.team.toLowerCase();
		const prev = best.get(key);
		out.set(s.id, prev === undefined ? null : s.accuracy - prev);
		if (prev === undefined || s.accuracy > prev) best.set(key, s.accuracy);
	}
	return out;
}

/** A clock that ticks on the server's time, for countdowns. */
export class Clock {
	now = $state(Date.now());
	constructor(live: Live, intervalMs = 250) {
		$effect(() => {
			const id = setInterval(() => (this.now = Date.now() + live.clockOffset), intervalMs);
			return () => clearInterval(id);
		});
	}
}
