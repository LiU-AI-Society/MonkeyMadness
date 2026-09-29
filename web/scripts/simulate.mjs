// Rehearsal helper: submits the baseline model under a handful of fake teams so the
// scoreboard has something to show. Run the server with SCORER=mock for varied scores.
// Usage: node scripts/simulate.mjs [baseUrl] [rounds]
import { readFile } from 'node:fs/promises';

const base = process.argv[2] ?? 'http://localhost:5173';
const rounds = Number(process.argv[3] ?? 4);
const teams = ['Banana Brigade', 'Gradient Gorillas', 'Overfit Orangutans', 'ReLU Rhesus', 'Capuchin Crunchers', 'Tensor Tamarins', 'Dropout Drills'];
const model = new Blob([await readFile(new URL('../../saved_models/base_line.onnx', import.meta.url))]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

for (let round = 0; round < rounds; round++) {
	for (const team of teams.sort(() => Math.random() - 0.5)) {
		const form = new FormData();
		form.set('team', team);
		form.append('model', model, 'model.onnx');
		// SvelteKit rejects cross-site form posts; a browser sends Origin, fetch in Node doesn't.
		const res = await fetch(`${base}/api/submit`, { method: 'POST', body: form, headers: { origin: new URL(base).origin } });
		console.log(team.padEnd(20), res.status, (await res.json()).error ?? 'ok');
		await sleep(1500 + Math.random() * 3000);
	}
	await sleep(8000);
}
