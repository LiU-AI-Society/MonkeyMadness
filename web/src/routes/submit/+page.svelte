<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { scale } from 'svelte/transition';
	import { backOut } from 'svelte/easing';
	import { resolve } from '$app/paths';
	import { Clock, improvementDeltas, live, pct } from '$lib/live.svelte';
	import { flowBananas, streamBananas } from '$lib/bananas';
	import { openStats } from '$lib/stats.svelte';
	import AnimatedNumber from '$lib/components/AnimatedNumber.svelte';

	const clock = new Clock(live, 1000);
	const STORE_KEY = 'monkeymadness:v1';

	let team = $state('');
	let files = $state<File[]>([]);
	let busy = $state(false);
	let error = $state<string | null>(null);
	let trackedId = $state<number | null>(null);

	let zone = $state<HTMLElement>();
	let fileChip = $state<HTMLElement>();
	let submitButton = $state<HTMLElement>();
	let trackerCard = $state<HTMLElement>();
	let charged = $state(false);

	// --- While a file is dragged over the page, bananas stream from the cursor into the drop zone.
	let dragging = $state(false);
	let pointer = { x: 0, y: 0 };
	let dragTimer: ReturnType<typeof setTimeout>;

	$effect(() => {
		if (!dragging || !zone) return;
		return streamBananas(() => pointer, zone);
	});

	function onWindowDragOver(e: DragEvent) {
		if (!e.dataTransfer?.types.includes('Files')) return;
		e.preventDefault();
		dragging = true;
		pointer = { x: e.clientX, y: e.clientY };
		clearTimeout(dragTimer);
		dragTimer = setTimeout(() => (dragging = false), 150); // dragover stops firing once the drag leaves
	}

	function onWindowDrop(e: DragEvent) {
		if (!e.dataTransfer?.files.length) return;
		e.preventDefault();
		dragging = false;
		setFiles([...e.dataTransfer.files]);
	}

	const isModel = (f: File) => f.name.toLowerCase().endsWith('.onnx');

	/**
	 * Drops add up, so the .onnx and its .onnx.data can arrive in separate drags:
	 * a new .onnx replaces the old .onnx, a same-named file replaces its old copy,
	 * anything else is added. (A stale .data file is harmless: the server only
	 * reads the one the model references.)
	 */
	async function setFiles(list: File[]) {
		if (!list.length) return;
		const newModel = list.some(isModel);
		const kept = files.filter((f) => !list.some((n) => n.name === f.name) && !(newModel && isModel(f)));
		files = [...kept, ...list];
		error = null;
		charged = false;
		await tick();
		if (fileChip && submitButton) {
			await flowBananas(fileChip, submitButton);
			charged = true;
		}
	}

	onMount(() => {
		try {
			const saved = JSON.parse(localStorage.getItem(STORE_KEY) ?? '{}');
			team = saved.team ?? '';
			trackedId = saved.trackedId ?? null;
		} catch {
			/* storage unavailable: start fresh */
		}
	});

	$effect(() => {
		try {
			localStorage.setItem(STORE_KEY, JSON.stringify({ team, trackedId }));
		} catch {
			/* ignore */
		}
	});

	async function submit(e: SubmitEvent) {
		e.preventDefault();
		if (!files.length) return;
		busy = true;
		error = null;
		const form = new FormData();
		form.set('team', team);
		for (const f of files) form.append('model', f);
		try {
			const res = await fetch(resolve('/api/submit'), { method: 'POST', body: form });
			const body = await res.json();
			if (!res.ok) {
				error = body.error;
				return;
			}
			trackedId = body.id;
			files = [];
			charged = false;
			await tick();
			if (submitButton && trackerCard) flowBananas(submitButton, trackerCard, { count: 10, spread: 500 });
		} catch {
			error = 'Upload failed, check your connection.';
		} finally {
			busy = false;
		}
	}

	const snap = $derived(live.snapshot);
	const tracked = $derived(snap?.submissions.find((s) => s.id === trackedId) ?? null);
	const deltas = $derived(improvementDeltas(snap?.submissions ?? []));
	const myKey = $derived(team.trim().toLowerCase());
	const myRow = $derived(snap?.leaderboard.find((r) => r.team.toLowerCase() === myKey) ?? null);
	const mySubs = $derived((snap?.submissions ?? []).filter((s) => s.team.toLowerCase() === myKey).reverse());
	const queueAhead = $derived.by(() => {
		if (!tracked || tracked.status !== 'queued' || !snap) return 0;
		return snap.submissions.filter((s) => s.status === 'queued' && s.id < tracked.id).length;
	});
	const cooldownLeft = $derived.by(() => {
		const last = mySubs[0];
		if (!last || !snap) return 0;
		return Math.max(0, Math.ceil(snap.cooldownSeconds - (clock.now - Date.parse(last.createdAt)) / 1000));
	});
	const step = $derived(tracked ? { queued: 1, scoring: 2, done: 3, failed: 3 }[tracked.status] : 0);
	const sizeLabel = (bytes: number) => (bytes > 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.ceil(bytes / 1e3)} KB`);
	const ready = $derived(files.some(isModel) && !!team.trim() && cooldownLeft === 0 && !busy);
</script>

<svelte:head><title>Submit · MonkeyMadness</title></svelte:head>
<svelte:window ondragover={onWindowDragOver} ondrop={onWindowDrop} />

<main class="mx-auto grid max-w-5xl grid-cols-[1fr_20rem] gap-12 px-8 py-12">
	<form class="space-y-6" onsubmit={submit}>
		<header>
			<h1 class="text-3xl font-semibold tracking-tight">Submit a model</h1>
			<p class="text-base-content/60 mt-1">Drop your exported .onnx and it gets scored on the hidden test set.</p>
		</header>

		<fieldset class="fieldset">
			<legend class="fieldset-legend text-base-content/50 text-xs font-normal tracking-[0.2em] uppercase">Team</legend>
			<input class="input input-lg w-full" type="text" bind:value={team} required maxlength="64" placeholder="Team name" />
		</fieldset>

		<!-- Drop zone -->
		<!-- Drag-and-drop only (no file picker): the banana stream is the point. -->
		<div
			bind:this={zone}
			role="region"
			aria-label="Drop zone for your model file"
			class="rounded-box relative flex min-h-72 flex-col items-center justify-center gap-4 overflow-hidden border border-dashed p-10 text-center transition-colors duration-300 {dragging ? 'border-primary' : 'border-base-300'}"
		>
			{#if files.length}
				<div bind:this={fileChip} in:scale={{ start: 1.4, duration: 500, easing: backOut }} class="relative flex flex-col items-center gap-3">
					<svg class="text-base-content size-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.2">
						<path stroke-linejoin="round" d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5z" /><path d="M14 3v5h5" />
					</svg>
					{#each files as f (f.name)}
						<div class="num text-sm">{f.name} <span class="text-base-content/40">· {sizeLabel(f.size)}</span></div>
					{/each}
					<span class="text-base-content/40 text-xs">
						{files.some(isModel) ? 'Drop a .onnx.data file to add it, or another .onnx to replace' : 'Now drop the .onnx model file'}
					</span>
					<button type="button" class="btn btn-ghost btn-xs text-base-content/50" onclick={() => ((files = []), (charged = false))}>Clear</button>
				</div>
			{:else}
				<div class="relative flex flex-col items-center gap-3">
					<svg
						class="size-10 transition-colors {dragging ? 'text-primary' : 'text-base-content/40'}"
						fill="none"
						viewBox="0 0 24 24"
						stroke="currentColor"
						stroke-width="1.2"
					>
						<path stroke-linecap="round" stroke-linejoin="round" d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
					</svg>
					<span class="text-lg font-medium">{dragging ? 'Let go' : 'Drag your model here'}</span>
					<span class="text-base-content/40 text-sm">.onnx, plus the .onnx.data file if your export made one</span>
				</div>
			{/if}
		</div>

		<button bind:this={submitButton} class="btn btn-primary btn-lg w-full transition-shadow" class:charged={charged && ready} disabled={!ready}>
			{#if busy}
				<span class="loading loading-spinner"></span> Uploading
			{:else if cooldownLeft > 0}
				Wait {cooldownLeft}s
			{:else}
				Submit
			{/if}
		</button>
		{#if error}<div role="alert" class="alert alert-error alert-soft">{error}</div>{/if}
	</form>

	<!-- Status column -->
	<aside class="space-y-8 pt-2">
		<div>
			<div class="text-base-content/50 border-base-300 border-b pb-2 text-xs tracking-[0.2em] uppercase">Your standing</div>
			{#if myRow}
				<div class="mt-3 flex items-baseline justify-between">
					<span class="num text-base-content/50">#{myRow.rank}</span>
					<span class="text-3xl font-semibold"><AnimatedNumber value={myRow.accuracy} /></span>
				</div>
			{:else}
				<p class="text-base-content/40 mt-3 text-sm">No score yet{team.trim() ? ` for ${team.trim()}` : ''}.</p>
			{/if}
		</div>

		<div bind:this={trackerCard}>
			<div class="text-base-content/50 border-base-300 border-b pb-2 text-xs tracking-[0.2em] uppercase">
				Latest submission{tracked ? ` #${tracked.id}` : ''}
			</div>
			{#if tracked}
				{@const d = deltas.get(tracked.id)}
				<ul class="steps mt-4 w-full text-xs">
					<li class="step {step >= 1 ? 'step-primary' : ''}">Queued</li>
					<li class="step {step >= 2 ? 'step-primary' : ''}">Scoring</li>
					<li class="step {step >= 3 ? (tracked.status === 'failed' ? 'step-error' : 'step-primary') : ''}">Result</li>
				</ul>
				<div class="mt-4 text-center">
					{#if tracked.status === 'queued'}
						<p class="text-base-content/60 text-sm">{queueAhead ? `${queueAhead} ahead of you` : 'Up next'}</p>
					{:else if tracked.status === 'scoring'}
						<p class="text-primary flex items-center justify-center gap-2 text-sm">
							<span class="loading loading-dots loading-sm"></span> Running on the test set
						</p>
					{:else if tracked.status === 'failed'}
						<p class="text-error text-left text-sm">{tracked.error}</p>
					{:else if tracked.hidden}
						<p class="text-base-content/60 text-sm">Scored. Results come at the reveal.</p>
					{:else}
						<div class="text-5xl font-semibold"><AnimatedNumber value={tracked.accuracy ?? 0} /></div>
						<div class="num mt-2 text-sm">
							{#if d == null}
								<span class="text-primary">first score</span>
							{:else if d > 0}
								<span class="text-primary">+{(d * 100).toFixed(1)} new best</span>
							{:else}
								<span class="text-base-content/50">−{Math.abs(d * 100).toFixed(1)} vs best</span>
							{/if}
						</div>
						<button type="button" class="btn btn-ghost btn-sm mt-3" onclick={() => openStats(tracked.id)}>Class breakdown</button>
					{/if}
				</div>
			{:else}
				<p class="text-base-content/40 mt-3 text-sm">Nothing submitted yet.</p>
			{/if}
		</div>

		{#if mySubs.length}
			<div>
				<div class="text-base-content/50 border-base-300 border-b pb-2 text-xs tracking-[0.2em] uppercase">History</div>
				<ul class="mt-2 text-sm">
					{#each mySubs.slice(0, 8) as s (s.id)}
						{@const d = deltas.get(s.id)}
						<li class="border-base-300 border-b">
							<button
								type="button"
								class="hover:bg-base-content/[0.03] flex w-full cursor-pointer justify-between py-2 text-left disabled:cursor-default"
								disabled={s.status !== 'done' || s.hidden}
								onclick={() => openStats(s.id)}
							>
							<span class="num text-base-content/40">#{s.id}</span>
							<span class="num">
								{#if s.status === 'done'}
									{s.hidden ? 'hidden' : pct(s.accuracy)}
									{#if d != null && d > 0}<span class="text-primary">↑</span>{/if}
								{:else}
									<span class="text-base-content/40">{s.status}</span>
								{/if}
							</span>
							</button>
						</li>
					{/each}
				</ul>
			</div>
		{/if}
	</aside>
</main>

<style>
	.charged {
		box-shadow:
			0 0 0 1px var(--color-primary),
			0 0 32px -6px var(--color-primary);
	}
</style>
