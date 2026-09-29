import { fail } from '@sveltejs/kit';
import { isAdmin, login, logout } from '$lib/server/auth';
import { config } from '$lib/server/config';
import * as db from '$lib/server/db';
import { broadcast, snapshot } from '$lib/server/live';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = ({ cookies }) => {
	if (!isAdmin(cookies)) return { admin: false as const, configured: !!config.adminPassword };
	return { admin: true as const, snapshot: snapshot({ admin: true }) };
};

const guard = <T>(fn: (form: FormData) => T) =>
	(async ({ cookies, request }) => {
		if (!isAdmin(cookies)) return fail(401, { error: 'Not logged in' });
		return fn(await request.formData());
	}) satisfies Actions[string];

export const actions: Actions = {
	login: async ({ cookies, request, url }) => {
		const form = await request.formData();
		if (!login(cookies, String(form.get('password') ?? ''), url.protocol === 'https:')) return fail(401, { error: 'Wrong password' });
	},
	logout: ({ cookies }) => logout(cookies),
	setEnd: guard((form) => {
		// The browser sends local time plus its UTC offset, so the server stores an unambiguous instant.
		const value = String(form.get('end') ?? '');
		const offset = Number(form.get('tzOffset') ?? 0);
		if (!value) db.setSetting('end_time', null);
		else db.setSetting('end_time', new Date(Date.parse(value + 'Z') + offset * 60000).toISOString());
		broadcast();
	}),
	freeze: guard(() => {
		db.setSetting('frozen_at', new Date().toISOString());
		broadcast();
	}),
	reveal: guard(() => {
		db.setSetting('frozen_at', null);
		broadcast({ kind: 'reveal' });
	}),
	delete: guard((form) => {
		db.deleteSubmission(Number(form.get('id')));
		broadcast();
	}),
	reset: guard((form) => {
		if (form.get('confirm') !== 'RESET') return fail(400, { error: 'Type RESET to confirm' });
		db.deleteAll();
		broadcast();
	})
};
