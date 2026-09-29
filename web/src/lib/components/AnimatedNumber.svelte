<script lang="ts">
	import { Tween } from 'svelte/motion';
	import { cubicOut } from 'svelte/easing';

	let { value, digits = 1, percent = true, duration = 900 }: { value: number; digits?: number; percent?: boolean; duration?: number } = $props();

	// svelte-ignore state_referenced_locally
	const tween = Tween.of(() => value, { duration, easing: cubicOut });
	const shown = $derived(percent ? `${(tween.current * 100).toFixed(digits)}%` : tween.current.toFixed(digits));
</script>

<span class="num">{shown}</span>
