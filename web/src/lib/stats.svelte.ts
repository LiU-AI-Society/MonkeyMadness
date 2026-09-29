/** Which submission's per-class stats modal is open (one shared modal lives in the root layout). */
export const statsModal = $state<{ id: number | null }>({ id: null });

export const openStats = (id: number) => (statsModal.id = id);
export const closeStats = () => (statsModal.id = null);

/** "black_headed_night_monkey" → "Black headed night monkey" */
export const prettyLabel = (label: string) => {
	const s = label.replace(/_/g, ' ');
	return s.charAt(0).toUpperCase() + s.slice(1);
};
