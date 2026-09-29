import { createHash, timingSafeEqual } from 'node:crypto';
import type { Cookies } from '@sveltejs/kit';
import { config } from './config';

const COOKIE = 'mm_admin';
const token = () => createHash('sha256').update(`monkeymadness:${config.adminPassword}`).digest('hex');

export function isAdmin(cookies: Cookies): boolean {
	const value = cookies.get(COOKIE);
	if (!config.adminPassword || !value) return false;
	const a = Buffer.from(value);
	const b = Buffer.from(token());
	return a.length === b.length && timingSafeEqual(a, b);
}

/** `secure` must follow the real protocol: browsers drop Secure cookies sent over plain http (e.g. a LAN IP). */
export function login(cookies: Cookies, password: string, secure: boolean): boolean {
	if (!config.adminPassword || password !== config.adminPassword) return false;
	cookies.set(COOKIE, token(), { path: '/', httpOnly: true, sameSite: 'strict', secure, maxAge: 60 * 60 * 24 * 3 });
	return true;
}

export function logout(cookies: Cookies) {
	cookies.delete(COOKIE, { path: '/' });
}
