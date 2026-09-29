<script lang="ts">
	import { enhance } from '$app/forms';
	import { pct } from '$lib/live.svelte';

	let { data, form } = $props();

	const tzOffset = new Date().getTimezoneOffset();
	const toLocalInput = (iso: string | null) => {
		if (!iso) return '';
		const d = new Date(iso);
		return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
	};
	const submissions = $derived(data.admin ? [...data.snapshot.submissions].reverse() : []);
</script>

<svelte:head><title>MonkeyMadness · Admin</title></svelte:head>

<main class="mx-auto max-w-5xl space-y-6 px-8 py-8">
	<h1 class="text-3xl font-bold tracking-tight">Admin</h1>
	{#if form?.error}<div role="alert" class="alert alert-error alert-soft">{form.error}</div>{/if}

	{#if !data.admin}
		{#if !data.configured}
			<div class="alert alert-warning alert-soft">Set ADMIN_PASSWORD in the environment to enable the admin page.</div>
		{:else}
			<form method="POST" action="?/login" use:enhance class="join">
				<input class="input join-item" type="password" name="password" placeholder="Password" required />
				<button class="btn btn-primary join-item">Log in</button>
			</form>
		{/if}
	{:else}
		<div class="grid gap-4 md:grid-cols-2">
			<form method="POST" action="?/setEnd" use:enhance class="card border-base-300 border">
				<div class="card-body">
					<h2 class="card-title">End time</h2>
					<input class="input w-full" type="datetime-local" name="end" value={toLocalInput(data.snapshot.endTime)} />
					<input type="hidden" name="tzOffset" value={tzOffset} />
					<div class="card-actions"><button class="btn btn-primary">Save</button></div>
				</div>
			</form>
			<div class="card border-base-300 border">
				<div class="card-body">
					<h2 class="card-title">Freeze</h2>
					<p class="text-base-content/60 text-sm">While frozen, new scores are hidden on every public page until you reveal.</p>
					<form method="POST" action={data.snapshot.frozen ? '?/reveal' : '?/freeze'} use:enhance class="card-actions">
						<button class="btn {data.snapshot.frozen ? 'btn-primary' : ''}">
							{data.snapshot.frozen ? 'Reveal scores' : 'Freeze leaderboard'}
						</button>
					</form>
				</div>
			</div>
		</div>

		<section class="space-y-2">
			<h2 class="text-xl font-semibold">Submissions ({submissions.length})</h2>
			<div class="border-base-300 rounded-box overflow-x-auto border">
				<table class="table-sm table">
					<thead><tr><th>#</th><th>Team</th><th>Status</th><th>Acc</th><th>Created</th><th></th></tr></thead>
					<tbody>
						{#each submissions as s (s.id)}
							<tr>
								<td class="num">{s.id}</td><td>{s.team}</td><td>{s.status}</td>
								<td class="num">{pct(s.accuracy)}</td>
								<td class="num">{new Date(s.createdAt).toLocaleTimeString()}</td>
								<td>
									<form method="POST" action="?/delete" use:enhance>
										<input type="hidden" name="id" value={s.id} /><button class="btn btn-ghost btn-xs">Delete</button>
									</form>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</section>

		<form method="POST" action="?/reset" use:enhance class="join">
			<input class="input join-item" name="confirm" placeholder="Type RESET" />
			<button class="btn btn-error join-item">Delete all submissions</button>
		</form>
		<form method="POST" action="?/logout" use:enhance><button class="btn btn-ghost btn-sm">Log out</button></form>
	{/if}
</main>
