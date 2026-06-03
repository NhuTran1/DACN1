"""Extract and rank expiration-date candidates from OCR text."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from ai_engine.ocr.regex_patterns import (
    DATE_PATTERNS,
    EXPIRATION_KEYWORDS,
    MONTHS,
    PRODUCTION_KEYWORDS,
    DatePattern,
)


KEYWORD_WINDOW = 48


@dataclass(frozen=True)
class KeywordMatch:
    keyword: str
    start: int
    end: int


def normalize_year(year: int) -> int | None:
    """Expand OCR-friendly two-digit years into the 2020-2099 range."""
    if 20 <= year <= 99:
        return 2000 + year
    if 1000 <= year <= 9999:
        return year
    return None


def _keyword_matches(text: str, keywords: Iterable[str]) -> list[KeywordMatch]:
    matches: list[KeywordMatch] = []
    for keyword in keywords:
        pattern = rf"(?<![A-Z0-9]){re.escape(keyword)}(?![A-Z0-9])"
        matches.extend(
            KeywordMatch(keyword, match.start(), match.end())
            for match in re.finditer(pattern, text)
        )
    return matches


def _nearest_preceding_keyword(
    start: int,
    keywords: list[KeywordMatch],
) -> tuple[str | None, int | None]:
    preceding_keywords = [keyword for keyword in keywords if keyword.end <= start]
    if not preceding_keywords:
        return None, None
    nearest = max(preceding_keywords, key=lambda keyword: keyword.end)
    return nearest.keyword, start - nearest.end


def _normalize_match(pattern: DatePattern, match: re.Match[str]) -> tuple[str, bool] | None:
    groups = match.groupdict()
    year = normalize_year(int(groups["year"]))
    if year is None:
        return None

    month_name = groups.get("month_name")
    month = MONTHS[month_name] if month_name else int(groups["month"])
    day_was_assumed = pattern.precision == "month"

    try:
        day = calendar.monthrange(year, month)[1] if day_was_assumed else int(groups["day"])
        normalized = date(year, month, day)
    except ValueError:
        return None
    return normalized.isoformat(), day_was_assumed


def _candidate_score(
    *,
    precision: str,
    expiration_distance: int | None,
    production_distance: int | None,
) -> float:
    score = 0.72 if precision == "day" else 0.62
    if expiration_distance is not None and expiration_distance <= KEYWORD_WINDOW:
        score += 0.24 * (1.0 - expiration_distance / (KEYWORD_WINDOW + 1))
    if production_distance is not None and production_distance <= KEYWORD_WINDOW:
        score -= 0.32 * (1.0 - production_distance / (KEYWORD_WINDOW + 1))
    return round(max(0.0, min(1.0, score)), 4)


def _overlaps(start: int, end: int, occupied_spans: list[tuple[int, int]]) -> bool:
    return any(start < occupied_end and occupied_start < end for occupied_start, occupied_end in occupied_spans)


def extract_date_candidates(text: str) -> list[dict]:
    """Extract valid non-overlapping date candidates and attach context scores."""
    normalized_text = (text or "").upper()
    expiration_matches = _keyword_matches(normalized_text, EXPIRATION_KEYWORDS)
    production_matches = _keyword_matches(normalized_text, PRODUCTION_KEYWORDS)
    occupied_spans: list[tuple[int, int]] = []
    candidates: list[dict] = []

    for pattern in DATE_PATTERNS:
        for match in pattern.regex.finditer(normalized_text):
            if _overlaps(match.start(), match.end(), occupied_spans):
                continue
            normalized_match = _normalize_match(pattern, match)
            if normalized_match is None:
                continue

            normalized_date, day_was_assumed = normalized_match
            # Only preceding labels are useful context. This prevents an EXP
            # label for a later date from incorrectly boosting an earlier date.
            expiration_keyword, expiration_distance = _nearest_preceding_keyword(
                match.start(), expiration_matches
            )
            production_keyword, production_distance = _nearest_preceding_keyword(
                match.start(), production_matches
            )
            candidates.append(
                {
                    "raw": match.group(0),
                    "normalized_date": normalized_date,
                    "detected_format": pattern.name,
                    "confidence": _candidate_score(
                        precision=pattern.precision,
                        expiration_distance=expiration_distance,
                        production_distance=production_distance,
                    ),
                    "expiration_keyword": expiration_keyword,
                    "expiration_keyword_distance": expiration_distance,
                    "production_keyword": production_keyword,
                    "production_keyword_distance": production_distance,
                    "day_was_assumed": day_was_assumed,
                    "recovered_from_noise": pattern.recovered_from_noise,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
            occupied_spans.append((match.start(), match.end()))

    return sorted(candidates, key=lambda candidate: candidate["start"])


def _selection_rank(candidate: dict) -> tuple:
    expiration_distance = candidate["expiration_keyword_distance"]
    production_distance = candidate["production_keyword_distance"]
    near_expiration = expiration_distance is not None and expiration_distance <= KEYWORD_WINDOW
    near_production = production_distance is not None and production_distance <= KEYWORD_WINDOW
    closer_to_expiration = (
        near_expiration
        and (production_distance is None or expiration_distance <= production_distance)
    )
    return (
        closer_to_expiration,
        near_expiration,
        not near_production,
        candidate["confidence"],
        not candidate["day_was_assumed"],
        -candidate["start"],
    )


def parse_date_text(text: str) -> dict:
    """Parse OCR text and select the most likely expiration date."""
    raw_text = text or ""
    candidates = extract_date_candidates(raw_text)
    warnings: list[str] = []

    if not candidates:
        warnings.append("No valid expiration-date candidate was found in OCR text.")
        return {
            "success": False,
            "raw_text": raw_text,
            "candidates": [],
            "selected_date": None,
            "selected_raw": None,
            "detected_format": None,
            "confidence": 0.0,
            "warnings": warnings,
            "normalized_date": None,
            "is_valid": False,
        }

    has_useful_context = any(
        (
            candidate["expiration_keyword_distance"] is not None
            and candidate["expiration_keyword_distance"] <= KEYWORD_WINDOW
        )
        or (
            candidate["production_keyword_distance"] is not None
            and candidate["production_keyword_distance"] <= KEYWORD_WINDOW
        )
        for candidate in candidates
    )
    if len(candidates) > 1 and not has_useful_context:
        selected = max(candidates, key=lambda candidate: candidate["normalized_date"])
        warnings.append("No context keyword found; selected the latest date as fallback.")
    else:
        selected = max(candidates, key=_selection_rank)
        if len(candidates) > 1:
            warnings.append("Multiple date candidates found; selected the strongest expiration-date match.")

    if selected["day_was_assumed"]:
        warnings.append("Selected month-only date was normalized to the last day of the month.")
    if selected["recovered_from_noise"]:
        warnings.append("Selected date was recovered from noisy OCR text.")

    return {
        "success": True,
        "raw_text": raw_text,
        "candidates": candidates,
        "selected_date": selected["normalized_date"],
        "selected_raw": selected["raw"],
        "detected_format": selected["detected_format"],
        "confidence": selected["confidence"],
        "warnings": warnings,
        "normalized_date": selected["normalized_date"],
        "is_valid": True,
    }


def parse_expiration_date(text: str) -> dict:
    """Parse OCR text and select the most likely expiration date."""
    return parse_date_text(text)
