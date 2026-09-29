import { bus, snapshot } from '$lib/server/live';
import { isAdmin } from '$lib/server/auth';
import type { LiveEvent } from '$lib/types';
import type { RequestHandler } from './$types';

// Server-Sent Events: every state change pushes a full snapshot (small at hackathon
// scale) plus an optional one-off event for the big screen to celebrate.
export const GET: RequestHandler = ({ cookies, request }) => {
	const admin = isAdmin(cookies);
	const encoder = new TextEncoder();
	let cleanup = () => {};

	const stream = new ReadableStream({
		start(controller) {
			const send = (event: string, data: unknown) => {
				try {
					controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
				} catch {
					cleanup();
				}
			};
			// Coalesce bursts of changes into one snapshot per tick.
			let pending: ReturnType<typeof setTimeout> | null = null;
			const onChange = (event: LiveEvent | null) => {
				if (event) send('live', event);
				pending ??= setTimeout(() => {
					pending = null;
					send('snapshot', snapshot({ admin }));
				}, 50);
			};
			const heartbeat = setInterval(() => controller.enqueue(encoder.encode(': ping\n\n')), 15000);

			cleanup = () => {
				bus.off('change', onChange);
				clearInterval(heartbeat);
				if (pending) clearTimeout(pending);
			};
			bus.on('change', onChange);
			request.signal.addEventListener('abort', cleanup);
			send('snapshot', snapshot({ admin }));
		},
		cancel() {
			cleanup();
		}
	});

	return new Response(stream, {
		headers: {
			'content-type': 'text/event-stream',
			'cache-control': 'no-cache, no-transform',
			connection: 'keep-alive',
			// Stop nginx from buffering the stream behind a reverse proxy.
			'x-accel-buffering': 'no'
		}
	});
};
