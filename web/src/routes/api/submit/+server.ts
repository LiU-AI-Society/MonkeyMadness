import fs from 'node:fs';
import path from 'node:path';
import { json } from '@sveltejs/kit';
import { config } from '$lib/server/config';
import * as db from '$lib/server/db';
import { broadcast } from '$lib/server/live';
import { enqueue, queueLength, uploadDir } from '$lib/server/queue';
import type { RequestHandler } from './$types';

/**
 * Keep the original name exactly (spaces included): a model exported with external
 * data references its .onnx.data file by that literal name. Only strip what could
 * escape the upload folder or hide the file.
 */
const safeName = (name: string) =>
	path
		.basename(name)
		.replace(/[\/\\\0]/g, '_')
		.replace(/^\.+/, '')
		.slice(0, 200) || 'file';

const recentByIp = new Map<string, number[]>();

/** Sliding one-minute window per client IP. */
function overIpLimit(ip: string): boolean {
	if (!config.ipLimitPerMinute) return false;
	const now = Date.now();
	const recent = (recentByIp.get(ip) ?? []).filter((t) => now - t < 60_000);
	if (recent.length >= config.ipLimitPerMinute) return true;
	recent.push(now);
	recentByIp.set(ip, recent);
	return false;
}

export const POST: RequestHandler = async ({ request, getClientAddress }) => {
	const reject = (message: string, status = 400) => json({ error: message }, { status });

	if (queueLength() >= config.maxQueue) return reject('The scoring queue is full right now. Try again in a minute.', 503);
	let ip = 'unknown';
	try {
		ip = getClientAddress();
	} catch {
		// ADDRESS_HEADER is set but the proxy didn't send it: don't fail the upload over it.
	}
	if (overIpLimit(ip)) return reject('Too many uploads from your network. Try again in a minute.', 429);

	const form = await request.formData();
	const team = String(form.get('team') ?? '').trim().replace(/\s+/g, ' ');
	const files = form.getAll('model').filter((f): f is File => f instanceof File && f.size > 0);

	if (!team) return reject('Team name is required.');
	if (team.length > config.maxTeamNameLen) return reject(`Team name is too long (max ${config.maxTeamNameLen} characters).`);

	const onnx = files.filter((f) => f.name.toLowerCase().endsWith('.onnx'));
	if (onnx.length !== 1)
		return reject('Upload exactly one .onnx model. If your export also produced a companion file (e.g. model.onnx.data), select both together.');
	const totalMb = files.reduce((sum, f) => sum + f.size, 0) / 1024 / 1024;
	if (totalMb > config.maxUploadMb) return reject(`Upload is too large (max ${config.maxUploadMb} MB).`, 413);

	const last = db.lastSubmission(team);
	if (last && (last.status === 'queued' || last.status === 'scoring'))
		return reject('Your previous submission is still being scored. Hang tight!', 429);
	if (last) {
		const wait = config.cooldownSeconds - (Date.now() - Date.parse(last.createdAt)) / 1000;
		if (wait > 0) return reject(`Please wait ${Math.ceil(wait)}s before submitting again.`, 429);
	}

	const id = db.createSubmission(team);
	const dir = uploadDir(id);
	fs.mkdirSync(dir, { recursive: true });
	for (const f of files) fs.writeFileSync(path.join(dir, safeName(f.name)), Buffer.from(await f.arrayBuffer()));

	enqueue(id);
	broadcast({ kind: 'submitted', team });
	return json({ id, team });
};
