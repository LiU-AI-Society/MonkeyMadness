<script lang="ts">
	import { fade } from 'svelte/transition';
	import { asset, resolve } from '$app/paths';
	import { pct } from '$lib/live.svelte';
	import { closeStats, prettyLabel, statsModal } from '$lib/stats.svelte';
	import type { ClassStat, Submission } from '$lib/types';

	let dialog = $state<HTMLDialogElement>();
	let data = $state<{ submission: Submission; perClass: ClassStat[] | null } | null>(null);
	let failed = $state(false);
	let index = $state(0);

	$effect(() => {
		const id = statsModal.id;
		if (id === null) {
			dialog?.close();
			return;
		}
		data = null;
		failed = false;
		index = 0;
		dialog?.showModal();
		const ctrl = new AbortController();
		fetch(resolve('/api/submissions/[id]', { id: String(id) }), { signal: ctrl.signal })
			.then((r) => (r.ok ? r.json() : Promise.reject()))
			.then((d) => (data = d))
			.catch(() => !ctrl.signal.aborted && (failed = true));
		return () => ctrl.abort();
	});

	const monkeyImage = (label: string) => asset(`/monkeys/${label}.jpg`);
	const sorted = $derived(data?.perClass ? [...data.perClass].sort((a, b) => b.recall - a.recall) : []);
	const current = $derived(sorted[index]);
	const step = (d: number) => sorted.length && (index = (index + d + sorted.length) % sorted.length);

	function onKeydown(e: KeyboardEvent) {
		if (statsModal.id === null || !sorted.length || (e.target as HTMLElement).tagName === 'SELECT') return;
		if (e.key === 'ArrowRight') step(1);
		if (e.key === 'ArrowLeft') step(-1);
	}
</script>

<svelte:window onkeydown={onKeydown} />

<dialog bind:this={dialog} class="modal" onclose={closeStats}>
	<div class="modal-box border-base-300 max-w-3xl overflow-hidden border p-0">
		<header class="border-base-300 grid grid-cols-[1fr_auto_1fr] items-center gap-6 border-b px-6 py-4">
			<div class="min-w-0">
				<div class="text-base-content/50 text-xs tracking-[0.2em] uppercase">Submission #{statsModal.id}</div>
				<h2 class="mt-1 truncate text-xl font-semibold tracking-tight">{data?.submission.team ?? ' '}</h2>
			</div>
			{#if sorted.length}
				<select class="select w-64" bind:value={index} aria-label="Class">
					{#each sorted as c, i (c.label)}
						<option value={i}>{prettyLabel(c.label)} · {pct(c.recall, 0)}</option>
					{/each}
				</select>
			{:else}
				<span></span>
			{/if}
			{#if data && !data.submission.hidden && data.submission.status === 'done'}
				<div class="flex justify-end gap-8 text-right">
					<div>
						<div class="text-base-content/50 text-xs tracking-[0.2em] uppercase">Accuracy</div>
						<div class="num mt-1 text-xl">{pct(data.submission.accuracy)}</div>
					</div>
					<div>
						<div class="text-base-content/50 text-xs tracking-[0.2em] uppercase">F1</div>
						<div class="num text-base-content/70 mt-1 text-xl">{pct(data.submission.f1)}</div>
					</div>
				</div>
			{/if}
		</header>

		{#if current}
			<div class="bg-base-300 relative aspect-video">
				{#key current.label}
					<img
						transition:fade={{ duration: 250 }}
						src={monkeyImage(current.label)}
						alt={prettyLabel(current.label)}
						class="absolute inset-0 size-full object-cover"
					/>
				{/key}
				<div class="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent"></div>
				<div class="absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-black/70 to-transparent"></div>
				<div class="pointer-events-none absolute top-0 left-0 p-6 text-3xl font-semibold tracking-tight text-white">
					{prettyLabel(current.label)}
				</div>

				<button class="nav left-4" aria-label="Previous class" onclick={() => step(-1)}>‹</button>
				<button class="nav right-4" aria-label="Next class" onclick={() => step(1)}>›</button>

				<div class="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-6 p-6 text-white">
					<div>
						<div class="flex items-baseline gap-3">
							<span class="num text-6xl font-semibold">{pct(current.recall, 0)}</span>
							<span class="text-white/60">correctly identified</span>
						</div>
						<div class="mt-4 h-1 w-72 overflow-hidden rounded-full bg-white/15">
							<div class="bg-primary h-full rounded-full transition-[width] duration-500" style="width:{current.recall * 100}%"></div>
						</div>
					</div>
					<dl class="grid shrink-0 grid-cols-[auto_auto] gap-x-6 gap-y-2 text-sm [&>dd]:text-right">
						<dt class="text-white/50">Precision</dt>
						<dd class="num">{pct(current.precision, 0)}</dd>
						<dt class="text-white/50">F1</dt>
						<dd class="num">{pct(current.f1, 0)}</dd>
						{#if current.confused_with}
							<dt class="text-white/50">Mistaken for</dt>
							<dd>
								<button class="pointer-events-auto inline-flex cursor-pointer items-center gap-2 hover:underline" onclick={() => (index = sorted.findIndex((c) => c.label === current.confused_with))}>
									<img src={monkeyImage(current.confused_with)} alt="" class="size-5 rounded-full object-cover" />
									{prettyLabel(current.confused_with)}
									<span class="num text-white/50">×{current.confused_count}</span>
								</button>
							</dd>
						{/if}
					</dl>
				</div>
			</div>
		{:else}
			<div class="px-6 py-16 text-center">
				{#if failed}
					<p class="text-error">Could not load the stats.</p>
				{:else if !data}
					<span class="loading loading-dots"></span>
				{:else if data.submission.hidden}
					<p class="text-base-content/60">The leaderboard is frozen. Stats come with the reveal.</p>
				{:else if data.submission.status !== 'done'}
					<p class="text-base-content/60">Not scored yet ({data.submission.status}).</p>
				{:else}
					<p class="text-base-content/60">No per-class stats were recorded for this submission.</p>
				{/if}
			</div>
		{/if}
	</div>
	<form method="dialog" class="modal-backdrop"><button aria-label="Close">close</button></form>
</dialog>

<style>
	.nav {
		position: absolute;
		top: 50%;
		translate: 0 -50%;
		display: grid;
		place-items: center;
		width: 2.75rem;
		height: 2.75rem;
		border-radius: 999px;
		background: rgb(0 0 0 / 0.45);
		color: white;
		font-size: 1.75rem;
		line-height: 1;
		cursor: pointer;
		z-index: 10;
		opacity: 0.7;
		transition: opacity 0.2s, background 0.2s;
	}
	.nav:hover {
		opacity: 1;
		background: rgb(0 0 0 / 0.7);
	}
</style>
