import { EventEmitter } from 'node:events';
import * as db from './db';
import { config } from './config';
import type { LiveEvent, Snapshot } from '$lib/types';

const g = globalThis as unknown as { __monkeyBus?: EventEmitter };
export const bus = (g.__monkeyBus ??= new EventEmitter().setMaxListeners(0));

export function snapshot(opts: { admin?: boolean } = {}): Snapshot {
	const frozenAt = db.getSetting('frozen_at');
	const submissions = db.allSubmissions(opts.admin ? null : frozenAt);
	return {
		serverTime: new Date().toISOString(),
		endTime: db.getSetting('end_time'),
		frozen: !!frozenAt,
		cooldownSeconds: config.cooldownSeconds,
		submissions,
		leaderboard: db.leaderboard(submissions)
	};
}

/** Tell every connected screen that state changed, optionally with a moment to celebrate. */
export function broadcast(event?: LiveEvent) {
	bus.emit('change', event ?? null);
}
