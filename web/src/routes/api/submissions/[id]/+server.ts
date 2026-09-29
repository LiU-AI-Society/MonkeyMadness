import { error, json } from '@sveltejs/kit';
import { isAdmin } from '$lib/server/auth';
import * as db from '$lib/server/db';
import type { RequestHandler } from './$types';

/** Per-class breakdown for one submission; withheld while its score is hidden by a freeze. */
export const GET: RequestHandler = ({ params, cookies }) => {
	const id = Number(params.id);
	const frozenAt = isAdmin(cookies) ? null : db.getSetting('frozen_at');
	const submission = db.allSubmissions(frozenAt).find((s) => s.id === id);
	if (!submission) error(404, 'No such submission');
	return json({ submission, perClass: submission.hidden ? null : db.perClass(id) });
};
