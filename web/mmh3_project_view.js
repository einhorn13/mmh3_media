// Pure projections shared by the dialog and offline UI checks.
export function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return "Timing unavailable";
    const minutes = Math.floor(seconds / 60);
    return `${minutes}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

export function takesForSegment(candidates, target) {
    return candidates.filter(c => (c.status === "accepted" ? c.accepted_segment_id : c.target_segment_id || "") === target);
}

export function timelineLayout(slots, pixelsPerSecond) {
    return slots.map(s => ({...s,
        width: Number.isFinite(s.duration_seconds) && s.duration_seconds > 0
            ? Math.max(2, s.duration_seconds * pixelsPerSecond) : 120,
        timingKnown: Number.isFinite(s.duration_seconds) && s.duration_seconds > 0}));
}
