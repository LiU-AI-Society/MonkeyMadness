export type Status = 'queued' | 'scoring' | 'done' | 'failed';

export type Submission = {
	id: number;
	team: string;
	status: Status;
	accuracy: number | null;
	precision: number | null;
	recall: number | null;
	f1: number | null;
	error: string | null;
	createdAt: string;
	scoredAt: string | null;
	/** Scored after the leaderboard was frozen: the score is withheld until reveal. */
	hidden: boolean;
};

export type ClassStat = {
	label: string;
	precision: number;
	recall: number;
	f1: number;
	support: number;
	confused_with: string | null;
	confused_count: number;
};

export type TeamRow = {
	rank: number;
	team: string;
	accuracy: number;
	f1: number;
	precision: number;
	recall: number;
	submissions: number;
	best: Submission;
};

export type Snapshot = {
	serverTime: string;
	endTime: string | null;
	frozen: boolean;
	cooldownSeconds: number;
	submissions: Submission[];
	leaderboard: TeamRow[];
};

/** One-off moments the big screen celebrates (toasts, lead-change takeover). */
export type LiveEvent =
	| { kind: 'submitted'; team: string }
	| {
			kind: 'scored';
			team: string;
			accuracy: number | null;
			delta: number | null;
			rank: number | null;
			newLeader: boolean;
	  }
	| { kind: 'failed'; team: string }
	| { kind: 'reveal' };
