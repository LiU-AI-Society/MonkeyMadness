<script lang="ts">
	import QRCode from 'qrcode';

	let { text, size = 120 }: { text: string; size?: number } = $props();

	let svg = $state('');
	$effect(() => {
		QRCode.toString(text, { type: 'svg', margin: 0, color: { dark: '#fafafa', light: '#0000' } }).then((s) => (svg = s));
	});
</script>

<div style="width:{size}px;height:{size}px" aria-label="QR code for {text}">{@html svg}</div>
