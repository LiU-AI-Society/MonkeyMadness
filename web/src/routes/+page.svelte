<script lang="ts">
	import { resolve } from '$app/paths';

	const SLIDES_URL =
		'https://docs.google.com/presentation/d/1VC7fK8MI8rnKlRPbKY3ylMFtQMWmfSDD/edit?usp=sharing&ouid=111602748456141665171&rtpof=true&sd=true';
	const VIDEO_URL = 'https://www.youtube.com/watch?v=bmWaPy7jvhQ';
	const COLAB_URL = 'https://colab.research.google.com';
	const REPO_URL = 'https://github.com/LiU-AI-Society/MonkeyMadness';

	let copied = $state(false);
	async function copyRepo() {
		try {
			await navigator.clipboard.writeText(REPO_URL);
			copied = true;
			setTimeout(() => (copied = false), 1800);
		} catch {
			/* clipboard blocked: the URL is selectable text anyway */
		}
	}
</script>

<svelte:head><title>MonkeyMadness · Get started</title></svelte:head>

<main class="mx-auto max-w-3xl px-8 py-16">
	<header>
		<div class="text-base-content/50 text-xs tracking-[0.25em] uppercase">LiU AI Society hackathon</div>
		<h1 class="mt-2 text-5xl font-semibold tracking-tight">MonkeyMadness</h1>
		<p class="text-base-content/60 mt-4 max-w-xl text-lg">
			Train a neural network that tells ten monkey species apart, then upload it and watch it climb the leaderboard.
		</p>
		<a class="btn btn-outline btn-sm mt-6" href={SLIDES_URL} target="_blank" rel="noreferrer">Intro presentation ↗</a>
	</header>

	<ol class="mt-14">
		<li class="step-row">
			<span class="step-num">01</span>
			<div>
				<h2 class="text-xl font-medium">Watch the setup video</h2>
				<p class="text-base-content/60 mt-1">A short walkthrough of opening the notebook in Colab and turning on the GPU.</p>
				<a class="btn btn-outline btn-sm mt-4" href={VIDEO_URL} target="_blank" rel="noreferrer">Watch on YouTube ↗</a>
			</div>
		</li>

		<li class="step-row">
			<span class="step-num">02</span>
			<div class="min-w-0 flex-1">
				<h2 class="text-xl font-medium">Open the notebook in Google Colab</h2>
				<p class="text-base-content/60 mt-1">
					Go to Colab, press <span class="text-base-content">Upload notebook</span>, pick
					<span class="text-base-content">GitHub</span> and paste:
				</p>
				<div class="border-base-300 bg-base-200 mt-4 flex items-center gap-2 rounded-lg border py-1.5 pr-1.5 pl-4">
					<code class="num min-w-0 flex-1 truncate text-sm select-all">{REPO_URL}</code>
					<button class="btn btn-sm {copied ? 'btn-primary' : ''}" onclick={copyRepo}>{copied ? 'Copied' : 'Copy'}</button>
				</div>
				<p class="text-base-content/60 mt-3">Then open <span class="num text-base-content">MonkeyMadness.ipynb</span>.</p>
				<a class="btn btn-outline btn-sm mt-4" href={COLAB_URL} target="_blank" rel="noreferrer">Open Google Colab ↗</a>
			</div>
		</li>

		<li class="step-row">
			<span class="step-num">03</span>
			<div>
				<h2 class="text-xl font-medium">Turn on the GPU</h2>
				<p class="text-base-content/60 mt-1">
					<span class="text-base-content">Runtime → Change runtime type</span> → Hardware accelerator:
					<span class="text-base-content">T4 GPU</span>. Training is much faster.
				</p>
			</div>
		</li>

		<li class="step-row">
			<span class="step-num">04</span>
			<div>
				<h2 class="text-xl font-medium">Train, export, submit</h2>
				<p class="text-base-content/60 mt-1">
					Follow the notebook. It saves your best model as an <span class="num text-base-content">.onnx</span> file. Download it and drag
					it into the submit page; it gets scored on a hidden test set within seconds.
				</p>
				<div class="mt-4 flex gap-3">
					<a class="btn btn-primary btn-sm" href={resolve('/submit')}>Submit a model</a>
					<a class="btn btn-outline btn-sm" href={resolve('/leaderboard')}>See the leaderboard</a>
				</div>
			</div>
		</li>
	</ol>
</main>

<style>
	.step-row {
		display: flex;
		gap: 2rem;
		padding: 2rem 0;
		border-top: 1px solid var(--color-base-300);
	}
	.step-num {
		width: 2.5rem;
		flex-shrink: 0;
		padding-top: 0.2rem;
		font-family: var(--font-mono);
		color: color-mix(in oklab, var(--color-base-content) 35%, transparent);
	}
</style>
