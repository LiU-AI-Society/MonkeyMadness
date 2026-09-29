<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import favicon from '$lib/assets/favicon.svg';
	import { live, pct } from '$lib/live.svelte';
	import Logo from '$lib/components/Logo.svelte';
	import StatsModal from '$lib/components/StatsModal.svelte';

	let { children } = $props();
	let open = $state(false);

	onMount(() => live.connect());

	const links = [
		{ href: resolve('/'), label: 'Scoreboard', hint: 'Live standings' },
		{ href: resolve('/submit'), label: 'Submit', hint: 'Upload your .onnx model' }
	];
	const leader = $derived(live.snapshot?.leaderboard[0] ?? null);
	const active = (href: string) => page.url.pathname === href;
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link
		rel="stylesheet"
		href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700;800&family=Geist+Mono:wght@500;700&display=swap"
	/>
</svelte:head>

<div class="drawer">
	<input id="nav-drawer" type="checkbox" class="drawer-toggle" bind:checked={open} />

	<div class="drawer-content flex h-screen flex-col">
		<nav class="navbar border-base-300 bg-base-100 min-h-14 shrink-0 gap-2 border-b px-4">
			<label for="nav-drawer" class="btn btn-ghost btn-square btn-sm" aria-label="Open menu">
				<svg class="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path stroke-linecap="round" d="M4 6h16M4 12h16M4 18h16" />
				</svg>
			</label>
			<a href={resolve('/')} class="flex items-center gap-3 hover:no-underline">
				<Logo class="text-base-content h-6 w-auto" />
				<span class="font-semibold tracking-tight">MonkeyMadness</span>
			</a>
			<div class="flex-1"></div>
			{#if leader}
				<span class="text-base-content/50 text-sm">
					Leading <span class="text-base-content ml-1">{leader.team}</span>
					<span class="num ml-1">{pct(leader.accuracy)}</span>
				</span>
			{/if}
			<span class="text-base-content/50 ml-4 flex items-center gap-2 text-sm">
				<span class="size-1.5 rounded-full {live.connected ? 'bg-primary' : 'bg-error animate-pulse'}"></span>
				{live.connected ? 'Live' : 'Reconnecting'}
			</span>
		</nav>
		<div class="min-h-0 flex-1 overflow-auto">
			{@render children()}
		</div>
	</div>

	<div class="drawer-side z-50">
		<label for="nav-drawer" aria-label="Close menu" class="drawer-overlay"></label>
		<aside class="bg-base-100 border-base-300 flex min-h-full w-80 flex-col border-r">
			<div class="border-base-300 flex h-14 items-center justify-between border-b px-4">
				<a href={resolve('/')} class="flex items-center gap-3 hover:no-underline" onclick={() => (open = false)}><Logo class="h-6 w-auto" /><span class="font-semibold tracking-tight">MonkeyMadness</span></a>
				<label for="nav-drawer" class="btn btn-ghost btn-square btn-sm" aria-label="Close menu">✕</label>
			</div>
			<ul class="menu w-full gap-1 p-3">
				{#each links as link (link.href)}
					<li>
						<a href={link.href} class="py-3 {active(link.href) ? 'menu-active' : ''}" onclick={() => (open = false)}>
							<span class="flex flex-col">
								<span class="font-medium">{link.label}</span>
								<span class="text-xs opacity-50">{link.hint}</span>
							</span>
						</a>
					</li>
				{/each}
			</ul>

			{#if live.snapshot?.leaderboard.length}
				<div class="px-6 pt-4">
					<h3 class="text-base-content/50 border-base-300 mb-2 border-b pb-2 text-xs tracking-[0.2em] uppercase">Top 5</h3>
					<ol class="space-y-1.5 text-sm">
						{#each live.snapshot.leaderboard.slice(0, 5) as row (row.team)}
							<li class="flex justify-between gap-2">
								<span class="truncate"><span class="text-base-content/50 num">{row.rank}.</span> {row.team}</span>
								<span class="num">{pct(row.accuracy)}</span>
							</li>
						{/each}
					</ol>
				</div>
			{/if}
			<div class="flex-1"></div>
			<p class="text-base-content/40 p-6 text-xs">
				<a class="link" href="https://www.liuais.com/" target="_blank" rel="noreferrer">liuais.com</a>
				<span class="mx-1">·</span>
				<a class="link" href={resolve('/admin')} onclick={() => (open = false)}>admin</a>
			</p>
		</aside>
	</div>
</div>

<StatsModal />
