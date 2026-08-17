# Este archivo:

# calcula los intervalos sin palabras reconocidas;
# considera también el tiempo anterior al primer segmento;
# considera el tiempo posterior al último segmento;
# evita reportar micropausas menores a dos segundos;
# calcula tiempo y turnos por SPEAKER_00, SPEAKER_01, etc.;
# no modifica la transcripción corrida.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_MINIMUM_GAP_SECONDS = 2.0
DEFAULT_INTERVAL_MERGE_TOLERANCE_SECONDS = 0.05
DEFAULT_SPEAKER_TURN_TOLERANCE_SECONDS = 0.75


def _read_value(item: Any, *names: str, default: Any = None) -> Any:
    """Lee un valor desde un objeto SQLAlchemy/Pydantic o desde un diccionario."""
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: float) -> float:
    return round(float(value), 3)


def _normalise_segments(
    segments: Iterable[Any],
    *,
    interval_duration_seconds: float,
) -> list[dict[str, Any]]:
    duration = max(0.0, _safe_float(interval_duration_seconds))
    normalised: list[dict[str, Any]] = []

    for segment in segments:
        start = max(
            0.0,
            _safe_float(
                _read_value(segment, "start_seconds", "start"),
            ),
        )
        end = max(
            0.0,
            _safe_float(
                _read_value(segment, "end_seconds", "end"),
            ),
        )

        if duration > 0:
            start = min(start, duration)
            end = min(end, duration)

        if end <= start:
            continue

        normalised.append(
            {
                "start": start,
                "end": end,
                "speaker": _read_value(
                    segment,
                    "speaker_label",
                    "speaker",
                ),
                "text": str(_read_value(segment, "text", default="") or "").strip(),
            }
        )

    return sorted(
        normalised,
        key=lambda item: (item["start"], item["end"]),
    )


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    tolerance_seconds: float,
) -> list[tuple[float, float]]:
    sorted_intervals = sorted(
        (
            (max(0.0, float(start)), max(0.0, float(end)))
            for start, end in intervals
            if float(end) > float(start)
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not sorted_intervals:
        return []

    merged: list[list[float]] = [
        [sorted_intervals[0][0], sorted_intervals[0][1]]
    ]

    for start, end in sorted_intervals[1:]:
        current = merged[-1]
        if start <= current[1] + tolerance_seconds:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def build_speech_gap_analysis(
    segments: Iterable[Any],
    *,
    interval_duration_seconds: float,
    minimum_gap_seconds: float = DEFAULT_MINIMUM_GAP_SECONDS,
    merge_tolerance_seconds: float = (
        DEFAULT_INTERVAL_MERGE_TOLERANCE_SECONDS
    ),
) -> dict[str, Any]:
    """
    Calcula intervalos sin palabras reconocidas usando los timestamps
    de los segmentos.

    Esta métrica NO afirma que exista silencio acústico. Durante un hueco
    puede haber ruido, música, voz demasiado baja o habla no reconocida.
    """
    duration = max(0.0, _safe_float(interval_duration_seconds))
    minimum_gap = max(0.0, _safe_float(minimum_gap_seconds))
    normalised = _normalise_segments(
        segments,
        interval_duration_seconds=duration,
    )
    speech_intervals = _merge_intervals(
        ((item["start"], item["end"]) for item in normalised),
        tolerance_seconds=max(0.0, merge_tolerance_seconds),
    )

    all_gaps: list[tuple[float, float]] = []
    cursor = 0.0

    for start, end in speech_intervals:
        if start > cursor:
            all_gaps.append((cursor, start))
        cursor = max(cursor, end)

    if duration > cursor:
        all_gaps.append((cursor, duration))

    reported_gaps: list[dict[str, Any]] = []
    reported_gap_seconds = 0.0
    all_uncovered_seconds = 0.0
    ignored_short_gap_seconds = 0.0

    for start, end in all_gaps:
        gap_duration = max(0.0, end - start)
        all_uncovered_seconds += gap_duration

        if gap_duration + 1e-9 < minimum_gap:
            ignored_short_gap_seconds += gap_duration
            continue

        if duration > 0 and start <= merge_tolerance_seconds and end >= (
            duration - merge_tolerance_seconds
        ):
            classification = "WHOLE_INTERVAL"
        elif start <= merge_tolerance_seconds:
            classification = "INITIAL"
        elif duration > 0 and end >= duration - merge_tolerance_seconds:
            classification = "FINAL"
        else:
            classification = "INTERNAL"

        reported_gap_seconds += gap_duration
        reported_gaps.append(
            {
                "start_seconds": _round(start),
                "end_seconds": _round(end),
                "duration_seconds": _round(gap_duration),
                "classification": classification,
            }
        )

    transcribed_speech_seconds = sum(
        end - start for start, end in speech_intervals
    )
    longest_gap = max(
        (
            item["duration_seconds"]
            for item in reported_gaps
        ),
        default=0.0,
    )

    initial_gap = next(
        (
            item["duration_seconds"]
            for item in reported_gaps
            if item["classification"] in {"INITIAL", "WHOLE_INTERVAL"}
        ),
        0.0,
    )
    final_gap = next(
        (
            item["duration_seconds"]
            for item in reversed(reported_gaps)
            if item["classification"] in {"FINAL", "WHOLE_INTERVAL"}
        ),
        0.0,
    )

    percentage = (
        (reported_gap_seconds / duration) * 100
        if duration > 0
        else 0.0
    )
    all_uncovered_percentage = (
        (all_uncovered_seconds / duration) * 100
        if duration > 0
        else 0.0
    )

    return {
        "definition": (
            "Intervalos en los que el modelo no generó palabras ni "
            "segmentos transcritos."
        ),
        "interpretation_warning": (
            "No equivale necesariamente a silencio acústico: puede existir "
            "ruido, música, voz baja o habla no reconocida."
        ),
        "minimum_gap_seconds": _round(minimum_gap),
        "interval_duration_seconds": _round(duration),
        "transcribed_speech_seconds": _round(transcribed_speech_seconds),
        "all_uncovered_seconds": _round(all_uncovered_seconds),
        "all_uncovered_percentage": _round(all_uncovered_percentage),
        "total_gap_seconds": _round(reported_gap_seconds),
        "gap_percentage": _round(percentage),
        "gap_count": len(reported_gaps),
        "longest_gap_seconds": _round(longest_gap),
        "initial_gap_seconds": _round(initial_gap),
        "final_gap_seconds": _round(final_gap),
        "ignored_short_gap_seconds": _round(ignored_short_gap_seconds),
        "intervals": reported_gaps,
    }



def _seconds_to_clock(value: float) -> str:
    total_milliseconds = max(0, int(round(float(value) * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _normalise_words(
    words: Iterable[Any],
    *,
    interval_duration_seconds: float,
) -> list[dict[str, Any]]:
    duration = max(0.0, _safe_float(interval_duration_seconds))
    normalised: list[dict[str, Any]] = []

    for word in words:
        start = max(
            0.0,
            _safe_float(
                _read_value(word, "start_seconds", "start"),
            ),
        )
        end = max(
            0.0,
            _safe_float(
                _read_value(word, "end_seconds", "end"),
            ),
        )

        if duration > 0:
            start = min(start, duration)
            end = min(end, duration)

        if end <= start:
            continue

        token = str(
            _read_value(word, "word", "text", default="") or ""
        ).strip()
        if not token:
            continue

        normalised.append(
            {
                "start": start,
                "end": end,
                "word": token,
                "speaker": _read_value(
                    word,
                    "speaker_label",
                    "speaker_id",
                    "speaker",
                ),
            }
        )

    return sorted(
        normalised,
        key=lambda item: (item["start"], item["end"]),
    )


def _dead_time_level(duration_seconds: float) -> str:
    if duration_seconds >= 60.0:
        return "EXTENDED"
    if duration_seconds >= 15.0:
        return "PROLONGED"
    return "RELEVANT"


def build_word_gap_analysis(
    words: Iterable[Any],
    *,
    interval_duration_seconds: float,
    minimum_gap_seconds: float = 5.0,
    merge_tolerance_seconds: float = 0.08,
) -> dict[str, Any]:
    """
    Calcula los intervalos sin palabras reconocidas usando timestamps
    individuales por palabra.

    Este es el análisis principal para el reporte de tiempos muertos. No
    afirma silencio acústico absoluto: puede existir ruido, música o voz que
    el modelo no reconoció.
    """
    duration = max(0.0, _safe_float(interval_duration_seconds))
    minimum_gap = max(0.0, _safe_float(minimum_gap_seconds))
    normalised = _normalise_words(
        words,
        interval_duration_seconds=duration,
    )

    if not normalised:
        return {
            "available": False,
            "analysis_source": "word_timestamps",
            "definition": (
                "Intervalos sin palabras reconocidas calculados con "
                "timestamps por palabra."
            ),
            "interpretation_warning": (
                "No equivale necesariamente a silencio acústico: puede "
                "existir ruido, música, voz baja o habla no reconocida."
            ),
            "minimum_gap_seconds": _round(minimum_gap),
            "interval_duration_seconds": _round(duration),
            "word_count": 0,
            "total_dead_time_seconds": 0.0,
            "dead_time_percentage": 0.0,
            "dead_time_count": 0,
            "longest_dead_time_seconds": 0.0,
            "ignored_short_gap_seconds": 0.0,
            "intervals": [],
            "unavailable_reason": (
                "El proveedor no devolvió timestamps por palabra o el "
                "trabajo fue generado antes de habilitar esta mejora."
            ),
        }

    word_intervals = _merge_intervals(
        ((item["start"], item["end"]) for item in normalised),
        tolerance_seconds=max(0.0, merge_tolerance_seconds),
    )

    all_gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in word_intervals:
        if start > cursor:
            all_gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration > cursor:
        all_gaps.append((cursor, duration))

    reported: list[dict[str, Any]] = []
    ignored_short_gap_seconds = 0.0
    all_uncovered_seconds = 0.0

    for start, end in all_gaps:
        gap_duration = max(0.0, end - start)
        all_uncovered_seconds += gap_duration

        if gap_duration + 1e-9 < minimum_gap:
            ignored_short_gap_seconds += gap_duration
            continue

        if duration > 0 and start <= merge_tolerance_seconds and end >= (
            duration - merge_tolerance_seconds
        ):
            position = "WHOLE_INTERVAL"
        elif start <= merge_tolerance_seconds:
            position = "INITIAL"
        elif duration > 0 and end >= duration - merge_tolerance_seconds:
            position = "FINAL"
        else:
            position = "INTERNAL"

        reported.append(
            {
                "start_seconds": _round(start),
                "end_seconds": _round(end),
                "duration_seconds": _round(gap_duration),
                "start_time": _seconds_to_clock(start),
                "end_time": _seconds_to_clock(end),
                "duration_time": _seconds_to_clock(gap_duration),
                "position": position,
                "level": _dead_time_level(gap_duration),
            }
        )

    total_dead_time = sum(
        item["duration_seconds"] for item in reported
    )
    percentage = (
        total_dead_time / duration * 100.0
        if duration > 0
        else 0.0
    )

    level_counts = {
        "RELEVANT": sum(
            1 for item in reported if item["level"] == "RELEVANT"
        ),
        "PROLONGED": sum(
            1 for item in reported if item["level"] == "PROLONGED"
        ),
        "EXTENDED": sum(
            1 for item in reported if item["level"] == "EXTENDED"
        ),
    }

    return {
        "available": True,
        "analysis_source": "word_timestamps",
        "definition": (
            "Intervalos sin palabras reconocidas calculados entre el final "
            "de una palabra y el inicio de la siguiente."
        ),
        "interpretation_warning": (
            "No equivale necesariamente a silencio acústico: puede existir "
            "ruido, música, voz baja o habla no reconocida."
        ),
        "minimum_gap_seconds": _round(minimum_gap),
        "interval_duration_seconds": _round(duration),
        "word_count": len(normalised),
        "all_uncovered_seconds": _round(all_uncovered_seconds),
        "total_dead_time_seconds": _round(total_dead_time),
        "dead_time_percentage": _round(percentage),
        "dead_time_count": len(reported),
        "longest_dead_time_seconds": _round(
            max(
                (item["duration_seconds"] for item in reported),
                default=0.0,
            )
        ),
        "ignored_short_gap_seconds": _round(
            ignored_short_gap_seconds
        ),
        "level_counts": level_counts,
        "levels": {
            "RELEVANT": "Desde el umbral configurado hasta menos de 15 s.",
            "PROLONGED": "Desde 15 s hasta menos de 60 s.",
            "EXTENDED": "60 s o más.",
        },
        "intervals": reported,
    }


def interval_iou(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> float:
    first_start = _safe_float(first.get("start_seconds"))
    first_end = _safe_float(first.get("end_seconds"))
    second_start = _safe_float(second.get("start_seconds"))
    second_end = _safe_float(second.get("end_seconds"))

    intersection = max(
        0.0,
        min(first_end, second_end) - max(first_start, second_start),
    )
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union if union > 0 else 0.0


def long_gap_intervals(
    analysis: Mapping[str, Any],
    *,
    minimum_duration_seconds: float,
) -> list[dict[str, Any]]:
    threshold = max(0.0, _safe_float(minimum_duration_seconds))
    intervals = analysis.get("intervals") or []
    return [
        dict(item)
        for item in intervals
        if isinstance(item, Mapping)
        and _safe_float(item.get("duration_seconds")) >= threshold
    ]


def long_gap_analyses_agree(
    first_analysis: Mapping[str, Any],
    second_analysis: Mapping[str, Any],
    *,
    minimum_duration_seconds: float,
    minimum_iou: float = 0.65,
) -> bool:
    first = long_gap_intervals(
        first_analysis,
        minimum_duration_seconds=minimum_duration_seconds,
    )
    second = long_gap_intervals(
        second_analysis,
        minimum_duration_seconds=minimum_duration_seconds,
    )

    if not first and not second:
        return True
    if not first or not second:
        return False

    required_iou = max(0.0, min(1.0, _safe_float(minimum_iou, 0.65)))

    def every_interval_matches(
        source: list[dict[str, Any]],
        target: list[dict[str, Any]],
    ) -> bool:
        return all(
            any(interval_iou(item, candidate) >= required_iou for candidate in target)
            for item in source
        )

    return every_interval_matches(first, second) and every_interval_matches(
        second,
        first,
    )


def result_quality_snapshot(
    result: Mapping[str, Any],
    *,
    interval_duration_seconds: float,
    minimum_gap_seconds: float,
    suspicious_gap_seconds: float,
) -> dict[str, Any]:
    words = list(result.get("words") or [])
    segments = list(result.get("segments") or [])
    duration = max(0.0, _safe_float(interval_duration_seconds))

    gap_analysis = build_word_gap_analysis(
        words,
        interval_duration_seconds=duration,
        minimum_gap_seconds=minimum_gap_seconds,
    )

    text = str(result.get("text") or "").strip()
    text_word_count = len(text.split())
    word_count = len(words)
    minutes = duration / 60.0
    word_rate = word_count / minutes if minutes > 0 else 0.0

    suspicious_intervals = long_gap_intervals(
        gap_analysis,
        minimum_duration_seconds=suspicious_gap_seconds,
    )

    return {
        "word_count": word_count,
        "text_word_count": text_word_count,
        "word_rate_per_minute": _round(word_rate),
        "longest_gap_seconds": _safe_float(
            gap_analysis.get("longest_dead_time_seconds")
        ),
        "suspicious_long_gap_count": len(suspicious_intervals),
        "suspicious_long_gaps": suspicious_intervals,
        "gap_analysis": gap_analysis,
        "diarization_requested": bool(
            result.get("diarization_requested")
        ),
    }


def suppress_unconfirmed_long_gaps(
    analysis: Mapping[str, Any],
    *,
    unconfirmed_intervals: Iterable[Mapping[str, Any]],
    minimum_duration_seconds: float,
    reason: str,
    minimum_iou: float = 0.25,
) -> dict[str, Any]:
    """
    Retira únicamente los huecos largos que coinciden con ventanas
    no confirmadas. Los demás intervalos del trabajo se conservan.
    """
    output = dict(analysis)
    threshold = max(0.0, _safe_float(minimum_duration_seconds))
    required_iou = max(0.0, min(1.0, _safe_float(minimum_iou, 0.25)))

    candidates = [
        dict(item)
        for item in unconfirmed_intervals
        if isinstance(item, Mapping)
    ]
    original = [
        dict(item)
        for item in output.get("intervals") or []
        if isinstance(item, Mapping)
    ]

    visible: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []

    for item in original:
        is_long = _safe_float(item.get("duration_seconds")) >= threshold
        matches_unconfirmed = any(
            interval_iou(item, candidate) >= required_iou
            for candidate in candidates
        )
        if is_long and matches_unconfirmed:
            suppressed.append(item)
        else:
            visible.append(item)

    total = sum(_safe_float(item.get("duration_seconds")) for item in visible)
    duration = _safe_float(output.get("interval_duration_seconds"))
    output["intervals"] = visible
    output["suppressed_intervals"] = suppressed
    output["suppressed_interval_count"] = len(suppressed)
    output["suppression_reason"] = reason if suppressed else None
    output["total_dead_time_seconds"] = _round(total)
    output["dead_time_percentage"] = _round(
        total / duration * 100.0 if duration > 0 else 0.0
    )
    output["dead_time_count"] = len(visible)
    output["longest_dead_time_seconds"] = _round(
        max(
            (_safe_float(item.get("duration_seconds")) for item in visible),
            default=0.0,
        )
    )
    output["level_counts"] = {
        "RELEVANT": sum(
            1 for item in visible if item.get("level") == "RELEVANT"
        ),
        "PROLONGED": sum(
            1 for item in visible if item.get("level") == "PROLONGED"
        ),
        "EXTENDED": sum(
            1 for item in visible if item.get("level") == "EXTENDED"
        ),
    }
    output["reliable"] = not suppressed
    return output


def build_speaker_analysis(
    segments: Iterable[Any],
    *,
    interval_duration_seconds: float,
    turn_merge_tolerance_seconds: float = (
        DEFAULT_SPEAKER_TURN_TOLERANCE_SECONDS
    ),
) -> dict[str, Any]:
    """
    Resume el tiempo de intervención por etiqueta de speaker.

    Las etiquetas SPEAKER_00, SPEAKER_01, etc. distinguen voces, pero no
    identifican por sí solas al docente o al estudiante.
    """
    duration = max(0.0, _safe_float(interval_duration_seconds))
    normalised = _normalise_segments(
        segments,
        interval_duration_seconds=duration,
    )

    intervals_by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    segment_count_by_speaker: dict[str, int] = defaultdict(int)
    unlabeled_segment_count = 0
    unlabeled_seconds = 0.0

    for item in normalised:
        label = str(item["speaker"] or "").strip()
        if not label:
            unlabeled_segment_count += 1
            unlabeled_seconds += item["end"] - item["start"]
            continue

        intervals_by_speaker[label].append(
            (item["start"], item["end"])
        )
        segment_count_by_speaker[label] += 1

    speakers: list[dict[str, Any]] = []

    for label, intervals in intervals_by_speaker.items():
        turns = _merge_intervals(
            intervals,
            tolerance_seconds=max(
                0.0,
                turn_merge_tolerance_seconds,
            ),
        )
        turn_durations = [end - start for start, end in turns]
        speaking_seconds = sum(turn_durations)
        percentage = (
            (speaking_seconds / duration) * 100
            if duration > 0
            else 0.0
        )

        speakers.append(
            {
                "speaker_label": label,
                "speaking_seconds": _round(speaking_seconds),
                "speaking_percentage_of_interval": _round(percentage),
                "segment_count": segment_count_by_speaker[label],
                "turn_count": len(turns),
                "average_turn_seconds": _round(
                    speaking_seconds / len(turns)
                    if turns
                    else 0.0
                ),
                "longest_turn_seconds": _round(
                    max(turn_durations, default=0.0)
                ),
                "first_intervention_seconds": _round(
                    turns[0][0] if turns else 0.0
                ),
                "last_intervention_seconds": _round(
                    turns[-1][1] if turns else 0.0
                ),
            }
        )

    speakers.sort(
        key=lambda item: (
            -item["speaking_seconds"],
            item["speaker_label"],
        )
    )

    return {
        "definition": (
            "Resumen descriptivo de intervención por etiqueta de voz "
            "devuelta por la diarización."
        ),
        "identity_warning": (
            "Una etiqueta como SPEAKER_00 no identifica automáticamente "
            "al docente; debe asignarse o verificarse aparte."
        ),
        "diarization_data_available": bool(speakers),
        "detected_speakers": len(speakers),
        "unlabeled_segment_count": unlabeled_segment_count,
        "unlabeled_segment_seconds": _round(unlabeled_seconds),
        "speakers": speakers,
    }


def build_transcript_analysis(
    segments: Iterable[Any],
    *,
    interval_duration_seconds: float,
    minimum_gap_seconds: float = DEFAULT_MINIMUM_GAP_SECONDS,
) -> dict[str, Any]:
    materialised_segments = list(segments)
    return {
        "speech_gap_analysis": build_speech_gap_analysis(
            materialised_segments,
            interval_duration_seconds=interval_duration_seconds,
            minimum_gap_seconds=minimum_gap_seconds,
        ),
        "speaker_analysis": build_speaker_analysis(
            materialised_segments,
            interval_duration_seconds=interval_duration_seconds,
        ),
    }

