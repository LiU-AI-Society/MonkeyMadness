<script lang="ts">
	import { onMount } from 'svelte';
	import { flip } from 'svelte/animate';
	import { fly, fade } from 'svelte/transition';
	import { cubicOut } from 'svelte/easing';
	import confetti from 'canvas-confetti';
	import { resolve } from '$app/paths';
	import { Clock, improvementDeltas, live, pct } from '$lib/live.svelte';
	import AnimatedNumber from '$lib/components/AnimatedNumber.svelte';
	import Countdown from '$lib/components/Countdown.svelte';
	import QrCode from '$lib/components/QrCode.svelte';
	import Logo from '$lib/components/Logo.svelte';
	import { openStats } from '$lib/stats.svelte';

	const MAX_ROWS = 10;
	let root = $state<HTMLElement>();
	const clock = new Clock(live);

	let submitUrl = $state('');
	let takeover = $state<{ team: string; accuracy: number } | null>(null);
	let flashes = $state<Record<string, number>>({});
	let moves = $state<Record<string, number>>({});

	onMount(() => {
		submitUrl = new URL(resolve('/submit'), location.href).href;
		// In fullscreen (the projector), scale all rem sizes with the viewport.
		const html = document.documentElement;
		const onFullscreen = () => (html.style.fontSize = document.fullscreenElement ? 'clamp(14px, calc(0.5vw + 0.7vh), 30px)' : '');
		document.addEventListener('fullscreenchange', onFullscreen);

		const offEvent = live.onEvent((e) => {
			if (e.kind !== 'scored') return;
			if (e.delta === null || e.delta > 0) flash(e.team);
			if (e.newLeader && e.accuracy !== null) celebrate(e.team, e.accuracy);
		});
		return () => {
			document.removeEventListener('fullscreenchange', onFullscreen);
			html.style.fontSize = '';
			offEvent();
		};
	});

	function flash(team: string) {
		const key = team.toLowerCase();
		flashes[key] = Date.now();
		setTimeout(() => delete flashes[key], 2500);
	}

	function celebrate(team: string, accuracy: number) {
		takeover = { team, accuracy };
		const colors = ['#3080ff', '#90c5ff', '#fafafa'];
		const base = { particleCount: 70, spread: 60, startVelocity: 55, ticks: 250, colors, scalar: 0.9 };
		confetti({ ...base, angle: 60, origin: { x: 0, y: 0.8 } });
		confetti({ ...base, angle: 120, origin: { x: 1, y: 0.8 } });
		setTimeout(() => (takeover = null), 5500);
	}

	const snap = $derived(live.snapshot);
	const board = $derived(snap?.leaderboard ?? []);

	// Remember each team's last rank so rows can show how far they moved for a while.
	let lastRanks = new Map<string, number>();
	$effect(() => {
		const next = new Map(board.map((r) => [r.team.toLowerCase(), r.rank]));
		for (const [team, rank] of next) {
			const before = lastRanks.get(team);
			if (before !== undefined && before !== rank) {
				moves[team] = before - rank;
				setTimeout(() => delete moves[team], 8000);
			}
		}
		lastRanks = next;
	});

	const deltas = $derived(improvementDeltas(snap?.submissions ?? []));
	const evaluating = $derived((snap?.submissions ?? []).filter((s) => s.status === 'queued' || s.status === 'scoring'));
	const results = $derived(
		(snap?.submissions ?? [])
			.filter((s) => s.status === 'done' || s.status === 'failed')
			.sort((a, b) => (b.scoredAt ?? '').localeCompare(a.scoredAt ?? ''))
			.slice(0, 8)
	);
	const rankLabel = (rank: number) => String(rank).padStart(2, '0');
	const clockTime = (iso: string | null) =>
		iso ? new Date(iso).toLocaleTimeString('sv-SE', { hour: '2-digit', minute: '2-digit' }) : '';
</script>

<svelte:head><title>Scoreboard · MonkeyMadness</title></svelte:head>

<div bind:this={root} class="bg-base-100 flex h-full flex-col gap-8 overflow-hidden px-10 py-8">
	<!-- Header -->
	<header class="grid grid-cols-[1.8fr_1fr] items-center gap-12">
		<div class="flex items-center gap-5">
			<Logo class="text-base-content h-12 w-auto" />
			<div class="border-base-300 border-l pl-5">
				<h1 class="text-4xl font-semibold tracking-tight">MonkeyMadness</h1>
				<p class="text-base-content/50 mt-1 flex items-center gap-3 text-xs tracking-[0.25em] uppercase">
					Live leaderboard
					{#if snap?.frozen}<span class="text-primary">· Frozen</span>{/if}
					{#if !live.connected}<span class="text-error">· Reconnecting</span>{/if}
				</p>
			</div>
		</div>
		<!-- Same columns as the body: clock sits on the right column's left edge, QR on its right edge. -->
		<div class="flex items-center justify-between gap-5">
			<Countdown endTime={snap?.endTime ?? null} now={clock.now} />
			<div class="flex items-center gap-5">
			<div class="text-right">
				<div class="text-base-content/50 text-xs tracking-[0.25em] uppercase">Submit at</div>
				<div class="num mt-1 text-lg">{submitUrl.replace(/^https?:\/\//, '').replace(/\/$/, '')}</div>
			</div>
			{#if submitUrl}<QrCode text={submitUrl} size={84} />{/if}
			</div>
		</div>
	</header>

		<button
		class="btn btn-ghost btn-square btn-sm text-base-content/30 hover:text-base-content fixed right-4 bottom-4"
		title="Fullscreen"
		aria-label="Fullscreen"
		onclick={() => root?.requestFullscreen()}
	>
		<svg class="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" /></svg>
	</button>

	<div class="grid min-h-0 flex-1 grid-cols-[1.8fr_1fr] gap-12">
		<!-- Standings -->
		<section class="flex min-h-0 flex-col">
			<div class="label-row grid-cols-[4rem_1fr_9rem_6rem_4rem]">
				<span>Rank</span><span>Team</span><span class="text-right">Accuracy</span><span class="text-right">F1</span><span class="text-right">Runs</span>
			</div>
			<ol>
				{#each board.slice(0, MAX_ROWS) as row (row.team.toLowerCase())}
					{@const key = row.team.toLowerCase()}
					{@const move = moves[key]}
					<li
						animate:flip={{ duration: 700, easing: cubicOut }}
						in:fade={{ duration: 400 }}
						class="border-base-300 relative grid grid-cols-[4rem_1fr_9rem_6rem_4rem] items-center border-b py-4 transition-colors duration-700"
						class:bg-primary-flash={flashes[key]}
					>
						<button class="hover:bg-base-content/[0.03] absolute inset-0 z-10 cursor-pointer" aria-label="Class stats for {row.team}" onclick={() => openStats(row.best.id)}></button>
						<span class="num text-xl {row.rank === 1 ? 'text-primary' : 'text-base-content/40'}">{rankLabel(row.rank)}</span>
						<div class="min-w-0">
							<div class="flex items-baseline gap-3">
								<span class="truncate font-medium {row.rank === 1 ? 'text-3xl' : 'text-2xl'}">{row.team}</span>
								{#if move}
									<span transition:fade class="num text-sm {move > 0 ? 'text-primary' : 'text-base-content/40'}">
										{move > 0 ? '↑' : '↓'}{Math.abs(move)}
									</span>
								{/if}
							</div>
						</div>
						<span class="text-right font-semibold {row.rank === 1 ? 'text-4xl' : 'text-3xl'}"><AnimatedNumber value={row.accuracy} /></span>
						<span class="text-base-content/50 text-right text-lg"><AnimatedNumber value={row.f1} /></span>
						<span class="num text-base-content/40 text-right">{row.submissions}</span>
						<!-- accuracy as a hairline under the row -->
						<span class="absolute bottom-[-1px] left-0 h-px transition-[width] duration-1000 {row.rank === 1 ? 'bg-primary' : 'bg-base-content/40'}" style="width:{row.accuracy * 100}%"></span>
					</li>
				{:else}
					<li class="text-base-content/40 py-24 text-center text-xl">Waiting for the first model</li>
				{/each}
			</ol>
			{#if board.length > MAX_ROWS}
				<p class="text-base-content/40 mt-4 text-sm">+ {board.length - MAX_ROWS} more teams</p>
			{/if}
		</section>

		<div class="flex min-h-0 flex-col gap-10">
			<!-- Queue -->
			<section>
				<div class="label-row flex justify-between">
					<span>Evaluating</span><span class="num">{evaluating.length || ''}</span>
				</div>
				<ul>
					{#each evaluating as s (s.id)}
						<li
							animate:flip={{ duration: 400 }}
							in:fly={{ y: -8, duration: 300 }}
							out:fade={{ duration: 250 }}
							class="border-base-300 relative flex items-center justify-between overflow-hidden border-b py-3"
						>
							<span class="truncate text-lg">{s.team}</span>
							{#if s.status === 'scoring'}
								<span class="text-primary flex items-center gap-2 text-sm">
									<span class="bg-primary size-1.5 animate-pulse rounded-full"></span> Scoring
								</span>
								<span class="scan"></span>
							{:else}
								<span class="text-base-content/40 text-sm">Queued</span>
							{/if}
						</li>
					{:else}
						<li class="text-base-content/30 py-3">Queue is empty</li>
					{/each}
				</ul>
			</section>

			<!-- Recent -->
			<section class="min-h-0 flex-1 overflow-hidden">
				<div class="label-row">Recent</div>
				<ul>
					{#each results as s (s.id)}
						{@const d = deltas.get(s.id)}
						<li
							animate:flip={{ duration: 500 }}
							in:fly={{ x: 16, duration: 450, easing: cubicOut }}
							class="border-base-300 relative grid grid-cols-[3rem_1fr_auto] items-baseline gap-3 border-b py-3"
						>
							{#if s.status === 'done' && !s.hidden}
								<button class="hover:bg-base-content/[0.03] absolute inset-0 z-10 cursor-pointer" aria-label="Class stats for {s.team} #{s.id}" onclick={() => openStats(s.id)}></button>
							{/if}
							<span class="num text-base-content/30 text-sm">{clockTime(s.scoredAt)}</span>
							<div class="min-w-0">
								<div class="truncate">{s.team}</div>
							</div>
							<div class="num text-right">
								{#if s.status === 'failed'}
									<span class="text-error text-sm">Failed</span>
								{:else if s.hidden}
									<span class="text-base-content/40 text-sm">Hidden</span>
								{:else}
									<span class="text-lg">{pct(s.accuracy)}</span>
									<span class="ml-2 inline-block w-14 text-sm {d != null && d > 0 ? 'text-primary' : 'text-base-content/30'}">
										{#if d == null}new{:else}{d > 0 ? '+' : '−'}{Math.abs(d * 100).toFixed(1)}{/if}
									</span>
								{/if}
							</div>
						</li>
					{/each}
				</ul>
			</section>
		</div>
	</div>
</div>

{#if takeover}
	<div transition:fade={{ duration: 400 }} class="bg-base-100/95 fixed inset-0 z-50 grid place-items-center">
		<div in:fly={{ y: 24, duration: 700, easing: cubicOut }} class="text-center">
			<div class="text-primary text-sm tracking-[0.5em] uppercase">New leader</div>
			<div class="mt-6 text-8xl font-semibold tracking-tight">{takeover.team}</div>
			<div class="mt-8 text-6xl font-semibold"><AnimatedNumber value={takeover.accuracy} duration={1800} /></div>
		</div>
	</div>
{/if}

<style>
	.label-row {
		display: grid;
		padding-bottom: 0.75rem;
		border-bottom: 1px solid var(--color-base-300);
		color: color-mix(in oklab, var(--color-base-content) 45%, transparent);
		font-size: 0.75rem;
		letter-spacing: 0.2em;
		text-transform: uppercase;
	}
	.label-row.flex {
		display: flex;
	}
	.bg-primary-flash {
		background-color: color-mix(in oklab, var(--color-primary) 12%, transparent);
	}
	.scan {
		position: absolute;
		bottom: -1px;
		left: 0;
		height: 1px;
		width: 30%;
		background: var(--color-primary);
		animation: scan 1.2s ease-in-out infinite;
	}
	@keyframes scan {
		from {
			transform: translateX(-100%);
		}
		to {
			transform: translateX(340%);
		}
	}
</style>
