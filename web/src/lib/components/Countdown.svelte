<script lang="ts">
	let { endTime, now }: { endTime: string | null; now: number } = $props();

	const remaining = $derived(endTime ? Math.max(0, Date.parse(endTime) - now) : null);
	const parts = $derived.by(() => {
		if (remaining === null) return null;
		const s = Math.floor(remaining / 1000);
		return { h: Math.floor(s / 3600), m: Math.floor((s % 3600) / 60), s: s % 60 };
	});
	const urgent = $derived(remaining !== null && remaining > 0 && remaining < 10 * 60 * 1000);
	const pad = (n: number) => String(n).padStart(2, '0');
</script>

{#if parts}
	<div class="flex flex-col items-start leading-none" class:urgent>
		<span class="text-base-content/50 mb-2 text-xs tracking-[0.25em] uppercase">
			{remaining === 0 ? 'Time is up' : 'Time left'}
		</span>
		<span class="countdown num text-5xl font-medium">
			{#if parts.h > 0}<span style="--value:{parts.h}" aria-live="polite">{parts.h}</span>:{/if}
			<span style="--value:{parts.m}; --digits: 2">{pad(parts.m)}</span>:
			<span style="--value:{parts.s}; --digits: 2">{pad(parts.s)}</span>
		</span>
	</div>
{/if}

<style>
	.urgent {
		color: var(--color-error);
		animation: pulse 1s ease-in-out infinite;
	}
	@keyframes pulse {
		50% {
			opacity: 0.55;
		}
	}
</style>
