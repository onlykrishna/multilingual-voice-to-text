"""
Analytics Generator — produce a structured JSON report + timeline SVG
from voice identification results.
"""
import json
import logging
import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class AnalyticsGenerator:
    """Compile detection results into a comprehensive report and SVG timeline."""

    # ── Report ────────────────────────────────────────────────────────────

    def generate_report(
        self,
        enrollment_data: Dict[str, Any],
        conversation_duration: float,
        all_segments: List[Dict[str, Any]],
        matched_segments: List[Dict[str, Any]],
        transcriptions: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Build the full voice identification JSON report.

        Args:
            enrollment_data:       Speaker profile dict from VoiceEnrollment.
            conversation_duration: Total conversation length in seconds.
            all_segments:          All diarized segments (with similarity key).
            matched_segments:      Only the target-speaker segments.
            transcriptions:        Optional {segment_idx: text} map from Whisper.

        Returns:
            Report dict matching the spec in the enhancement prompt.
        """
        transcriptions = transcriptions or {}

        # --- target speaker stats ---
        total_time  = sum(s["duration"] for s in matched_segments)
        n_occ       = len(matched_segments)
        avg_turn    = total_time / n_occ if n_occ else 0.0
        durations   = [s["duration"] for s in matched_segments]
        avg_conf    = (
            sum(s.get("similarity", 0) for s in matched_segments) / n_occ * 100
            if n_occ else 0.0
        )

        # --- other speakers stats ---
        other: Dict[str, Dict] = {}
        for seg in all_segments:
            if seg.get("is_match"):
                continue
            sp = seg.get("speaker", "UNKNOWN")
            if sp not in other:
                other[sp] = {"speaking_time": 0.0, "segment_count": 0}
            other[sp]["speaking_time"]  += seg["duration"]
            other[sp]["segment_count"]  += 1

        # --- timeline (all segments labelled) ---
        timeline = []
        for seg in sorted(all_segments, key=lambda x: x["start"]):
            timeline.append({
                "start":      round(seg["start"], 2),
                "end":        round(seg["end"],   2),
                "speaker":    "TARGET" if seg.get("is_match") else seg.get("speaker", "OTHER"),
                "confidence": round(seg.get("similarity", 0) * 100, 1) if seg.get("is_match") else None,
            })

        report = {
            "generated_at": datetime.datetime.now().isoformat(),
            "enrollment": {
                "speaker_id":          enrollment_data["speaker_id"],
                "name":                enrollment_data["metadata"].get("name"),
                "sample_duration":     enrollment_data["metadata"]["sample_duration"],
                "quality_score":       enrollment_data["metadata"]["quality_score"],
                "embedding_dimensions": enrollment_data["metadata"]["embedding_size"],
                "timestamp":           enrollment_data["metadata"]["created_at"],
            },
            "conversation": {
                "total_duration":          round(conversation_duration, 2),
                "total_speakers_detected": len(other) + (1 if matched_segments else 0),
                "target_speaker_present":  n_occ > 0,
                "overall_confidence":      round(avg_conf, 1),
            },
            "target_speaker_analysis": {
                "total_speaking_time":       round(total_time, 2),
                "speaking_time_percentage":  round(total_time / conversation_duration * 100, 1) if conversation_duration else 0,
                "occurrence_count":          n_occ,
                "average_turn_duration":     round(avg_turn, 2),
                "longest_turn_duration":     round(max(durations, default=0), 2),
                "shortest_turn_duration":    round(min(durations, default=0), 2),
            },
            "speaking_segments": [
                {
                    "segment_id":    i + 1,
                    "start_time":    round(seg["start"], 2),
                    "end_time":      round(seg["end"],   2),
                    "duration":      round(seg["duration"], 2),
                    "confidence":    round(seg.get("similarity", 0) * 100, 1),
                    "transcription": transcriptions.get(i, ""),
                }
                for i, seg in enumerate(matched_segments)
            ],
            "other_speakers": [
                {
                    "speaker_label":  sp,
                    "speaking_time":  round(d["speaking_time"], 2),
                    "percentage":     round(d["speaking_time"] / conversation_duration * 100, 1) if conversation_duration else 0,
                    "segment_count":  d["segment_count"],
                }
                for sp, d in sorted(other.items(), key=lambda x: -x[1]["speaking_time"])
            ],
            "timeline": timeline,
        }
        return report

    def save_report(self, report: Dict[str, Any], output_path: str) -> str:
        """Persist report as indented JSON and return the path."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved → {output_path}")
        return output_path

    # ── SVG Timeline ──────────────────────────────────────────────────────

    def generate_svg_timeline(
        self,
        timeline: List[Dict[str, Any]],
        total_duration: float,
        width: int = 900,
        bar_h: int = 32,
        gap: int = 10,
    ) -> str:
        """
        Generate an SVG speaker timeline as a string.

        Args:
            timeline:       List of {start, end, speaker} dicts.
            total_duration: Conversation length in seconds.
            width:          SVG width in pixels.
            bar_h:          Height of each speaker bar.
            gap:            Gap between bars.

        Returns:
            SVG string (ready to embed in HTML or save as .svg).
        """
        if total_duration <= 0:
            return "<svg></svg>"

        # Gather unique speakers (TARGET first)
        speakers_set = set(seg["speaker"] for seg in timeline)
        speakers = ["TARGET"] + sorted(s for s in speakers_set if s != "TARGET")

        COLORS = {
            "TARGET": "#8b5cf6",
        }
        OTHER_PALETTE = [
            "#06b6d4", "#f59e0b", "#10b981", "#ef4444",
            "#3b82f6", "#ec4899", "#a3e635", "#fb923c",
        ]
        for i, sp in enumerate(s for s in speakers if s != "TARGET"):
            COLORS[sp] = OTHER_PALETTE[i % len(OTHER_PALETTE)]

        label_w = 100
        plot_w  = width - label_w - 20
        total_h = len(speakers) * (bar_h + gap) + 60   # extra for axis

        px_per_s = plot_w / total_duration

        rows = []
        for row_idx, sp in enumerate(speakers):
            y = row_idx * (bar_h + gap) + 10
            color = COLORS.get(sp, "#888")
            label = "🎯 TARGET" if sp == "TARGET" else sp

            # Speaker label
            rows.append(
                f'<text x="{label_w - 8}" y="{y + bar_h/2 + 5}" '
                f'text-anchor="end" font-size="12" fill="#cbd5e1" '
                f'font-family="Inter, sans-serif">{label}</text>'
            )

            # Background bar
            rows.append(
                f'<rect x="{label_w}" y="{y}" width="{plot_w}" height="{bar_h}" '
                f'rx="4" fill="#1e1e2a"/>'
            )

            # Segments
            total_sp_time = 0.0
            for seg in timeline:
                if seg["speaker"] != sp:
                    continue
                sx = label_w + seg["start"] * px_per_s
                sw = max(2, (seg["end"] - seg["start"]) * px_per_s)
                total_sp_time += seg["end"] - seg["start"]
                tip = f"{seg['start']:.1f}s–{seg['end']:.1f}s"
                if seg.get("confidence") is not None:
                    tip += f" ({seg['confidence']:.0f}%)"
                rows.append(
                    f'<rect x="{sx:.1f}" y="{y}" width="{sw:.1f}" height="{bar_h}" '
                    f'rx="3" fill="{color}" opacity="0.9">'
                    f'<title>{tip}</title></rect>'
                )

            # Duration label on the right
            pct = total_sp_time / total_duration * 100 if total_duration else 0
            rows.append(
                f'<text x="{label_w + plot_w + 8}" y="{y + bar_h/2 + 5}" '
                f'font-size="10" fill="#94a3b8" font-family="Inter, sans-serif">'
                f'{total_sp_time:.0f}s ({pct:.0f}%)</text>'
            )

        # Time axis
        axis_y = len(speakers) * (bar_h + gap) + 15
        rows.append(
            f'<line x1="{label_w}" y1="{axis_y}" x2="{label_w + plot_w}" '
            f'y1="{axis_y}" y2="{axis_y}" stroke="#475569" stroke-width="1"/>'
        )
        tick_interval = max(10, round(total_duration / 10 / 10) * 10)
        t = 0
        while t <= total_duration:
            tx = label_w + t * px_per_s
            rows.append(
                f'<text x="{tx:.1f}" y="{axis_y + 14}" text-anchor="middle" '
                f'font-size="10" fill="#64748b" font-family="Inter, sans-serif">{t}s</text>'
            )
            rows.append(
                f'<line x1="{tx:.1f}" y1="{axis_y}" x2="{tx:.1f}" y2="{axis_y + 4}" '
                f'stroke="#475569" stroke-width="1"/>'
            )
            t += tick_interval

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{total_h}" '
            f'style="background:#0f172a;border-radius:12px;">'
            + "\n".join(rows)
            + "</svg>"
        )
        return svg
