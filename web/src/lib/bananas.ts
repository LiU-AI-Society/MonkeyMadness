// Little bananas that fly along curves between points (pure Web Animations, no deps).

const BANANA_SVG = `<svg viewBox="0 0 24 24" width="100%" height="100%" aria-hidden="true">
<path d="M3.5 13.5c2.6 5.6 11.3 7 16.6-1.4.5-.8-.1-1.7-.9-1.4-5.3 1.9-10.4 1.3-13.6-2.1-.6-.6-1.7-.3-1.8.5-.3 1.6-.6 3-.3 4.4z" fill="#facc15"/>
<path d="M4.3 12.2c3 3.8 8.4 5 14.4 1.4" stroke="#ca8a04" stroke-width="1" fill="none" stroke-linecap="round" opacity=".7"/>
<path d="M19.4 10.9l1.4-1.6" stroke="#713f12" stroke-width="1.6" stroke-linecap="round"/>
</svg>`;

type Point = { x: number; y: number };

const reducedMotion = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

function center(el: Element): Point {
	const r = el.getBoundingClientRect();
	return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
}

function randomPointIn(el: Element, inset = 0.2): Point {
	const r = el.getBoundingClientRect();
	return {
		x: r.left + r.width * (inset + Math.random() * (1 - 2 * inset)),
		y: r.top + r.height * (inset + Math.random() * (1 - 2 * inset))
	};
}

let layer: HTMLDivElement | null = null;
let inFlight = 0;

function getLayer() {
	if (!layer) {
		layer = document.createElement('div');
		layer.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:60;overflow:hidden';
		document.body.appendChild(layer);
	}
	return layer;
}

/** One banana from a to b along a bowed curve. Resolves when it lands. */
function fly(a: Point, b: Point, { duration = 1000, delay = 0, bow = 260, lift = 100 } = {}): Promise<void> {
	const el = document.createElement('div');
	const size = 24 + Math.random() * 12;
	el.style.cssText = `position:absolute;left:0;top:0;width:${size}px;height:${size}px;margin:${-size / 2}px 0 0 ${-size / 2}px;opacity:0`;
	el.innerHTML = BANANA_SVG;
	getLayer().appendChild(el);
	inFlight++;

	// Quadratic bezier with a randomly bowed control point, sampled into keyframes.
	const c = { x: (a.x + b.x) / 2 + (Math.random() - 0.5) * bow, y: Math.min(a.y, b.y) - lift * (0.5 + Math.random()) };
	const spin = (Math.random() < 0.5 ? -1 : 1) * (60 + Math.random() * 120);
	const frames: Keyframe[] = [];
	const steps = 16;
	for (let s = 0; s <= steps; s++) {
		const t = s / steps;
		const x = (1 - t) ** 2 * a.x + 2 * (1 - t) * t * c.x + t ** 2 * b.x;
		const y = (1 - t) ** 2 * a.y + 2 * (1 - t) * t * c.y + t ** 2 * b.y;
		const grow = t < 0.15 ? t / 0.15 : t > 0.85 ? (1 - t) / 0.15 : 1;
		frames.push({ transform: `translate(${x}px, ${y}px) rotate(${spin * t}deg) scale(${0.4 + grow * 0.6})`, opacity: Math.min(1, grow * 1.5) });
	}
	const anim = el.animate(frames, { duration, delay, easing: 'cubic-bezier(.45,.05,.55,.95)', fill: 'both' });
	return anim.finished
		.catch(() => {})
		.then(() => {
			el.remove();
			if (--inFlight === 0 && layer) {
				layer.remove();
				layer = null;
			}
		});
}

/** A burst of bananas from one element to another. Resolves when the last one lands. */
export function flowBananas(from: Element, to: Element, { count = 14, spread = 1200 } = {}): Promise<void> {
	if (reducedMotion()) return Promise.resolve();
	const a = center(from);
	const b = center(to);
	const flights = Array.from({ length: count }, (_, i) =>
		fly(a, b, { duration: 1400 + Math.random() * 500, delay: (i / count) * spread })
	);
	return Promise.all(flights).then(() => {});
}

/**
 * A continuous stream from a moving point (e.g. the dragged file under the cursor) into an element.
 * Returns a stop function; bananas already in the air finish their flight.
 */
export function streamBananas(from: () => Point, to: Element, { interval = 110 } = {}): () => void {
	if (reducedMotion()) return () => {};
	const timer = setInterval(() => {
		const a = from();
		const b = randomPointIn(to);
		const dist = Math.hypot(b.x - a.x, b.y - a.y);
		if (dist < 40) return; // cursor is already inside the target
		const jitter = { x: a.x + (Math.random() - 0.5) * 16, y: a.y + (Math.random() - 0.5) * 16 };
		fly(jitter, b, { duration: 900 + Math.min(dist, 900) * 1.2, bow: Math.min(dist, 400) * 0.5, lift: Math.min(dist, 400) * 0.25 });
	}, interval);
	return () => clearInterval(timer);
}
