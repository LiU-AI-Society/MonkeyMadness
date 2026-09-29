import { json } from '@sveltejs/kit';
import { snapshot } from '$lib/server/live';

export const GET = () => json(snapshot().leaderboard);
