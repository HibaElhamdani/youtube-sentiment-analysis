"""Preprocessing utilities for YouTube comment sentiment analysis."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from config import PROCESSED_DATA_PATH, RAW_DATA_PATH


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#\w+", re.IGNORECASE)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_MULTISPACE_RE = re.compile(r"\s+")
_ARABIC_LETTERS_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN_LETTERS_RE = re.compile(r"[a-zA-Z]")
_EMOJI_RE = re.compile(
	"["
	"\U0001F300-\U0001F5FF"
	"\U0001F600-\U0001F64F"
	"\U0001F680-\U0001F6FF"
	"\U0001F700-\U0001F77F"
	"\U0001F780-\U0001F7FF"
	"\U0001F800-\U0001F8FF"
	"\U0001F900-\U0001F9FF"
	"\U0001FA00-\U0001FA6F"
	"\U0001FA70-\U0001FAFF"
	"\u2600-\u26FF"
	"\u2700-\u27BF"
	"]+",
	flags=re.UNICODE,
)


# A conservative stopword list (Darija + Arabic + French + English) to reduce noise.
# Avoid including short sentiment words here.
_STOPWORDS = {
	"و",
	"في",
	"على",
	"من",
	"الى",
	"إلى",
	"عن",
	"ما",
	"لا",
	"نعم",
	"هو",
	"هي",
	"هم",
	"ها",
	"هذا",
	"هذه",
	"ذلك",
	"تلك",
	"واش",
	"شنو",
	"شحال",
	"فين",
	"كيفاش",
	"علاش",
	"b",
	"c",
	"d",
	"de",
	"des",
	"du",
	"et",
	"le",
	"la",
	"les",
	"un",
	"une",
	"the",
	"and",
	"to",
	"of",
	"for",
	"in",
}


# A small lexicon to prevent dropping short but meaningful sentiment comments.
_SENTIMENT_LEXICON = {
	"خايب",
	"حامض",
	"يخ",
	"زويين",
	"زوين",
	"مزيان",
	"مزيانة",
	"bzzaf",
	"bzaaf",
	"7mar",
	"m9awed",
	"mqawda",
	"مقرف",
	"روعة",
	"واعر",
	"وااعرة",
	"khayb",
	"hamd",
}


# Heuristic vocab for Darija vs MSA. This is intentionally small and explainable.
_DARIJA_MARKERS = {
	"بزاف",
	"واش",
	"عافاك",
	"حيت",
	"كاين",
	"غادي",
	"خايب",
	"زوين",
	"مزيان",
	"مكاين",
	"بغيت",
	"راك",
	"دابا",
	"شوية",
	"هاد",
	"ديال",
	"nt",
	"nta",
	"bghit",
	"ghadi",
	"bzaaf",
	"7it",
	"3lach",
}
_MSA_MARKERS = {
	"يجب",
	"ينبغي",
	"ذلك",
	"هذه",
	"لكن",
	"لذلك",
	"بسبب",
	"الذي",
	"التي",
	"حيث",
	"بالتالي",
	"على",
	"إلى",
}


@dataclass
class PreprocessConfig:
	min_tokens: int = 2
	# Lower threshold keeps mixed Darija/MSA comments (option 2).
	darija_ratio_threshold: float = 0.4
	keep_short_if_sentiment: bool = True
	drop_emoji_only: bool = True


def _normalize_arabic(text: str) -> str:
	# Basic normalization to reduce variants without aggressive stemming.
	text = re.sub(r"[\u064B-\u065F\u0670]", "", text)  # diacritics
	text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
	text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
	text = text.replace("ة", "ه")
	return text


def _strip_noise(text: str) -> str:
	text = _URL_RE.sub(" ", text)
	text = _MENTION_RE.sub(" ", text)
	text = _HASHTAG_RE.sub(" ", text)
	text = _TIME_RE.sub(" ", text)
	text = _EMOJI_RE.sub(" ", text)
	return text


def _basic_clean(text: str) -> str:
	text = text.replace("\u200f", " ").replace("\u200e", " ")
	text = _strip_noise(text)
	text = _normalize_arabic(text)
	text = text.lower()
	text = re.sub(r"[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", " ", text)
	text = _MULTISPACE_RE.sub(" ", text).strip()
	return text


def _tokenize(text: str) -> List[str]:
	if not text:
		return []
	return [tok for tok in text.split() if tok]


def _remove_stopwords(tokens: Iterable[str]) -> List[str]:
	return [tok for tok in tokens if tok not in _STOPWORDS]


def _has_letters(text: str) -> bool:
	return bool(_ARABIC_LETTERS_RE.search(text) or _LATIN_LETTERS_RE.search(text))


def _is_emoji_only(text: str) -> bool:
	if not text:
		return True
	return not _has_letters(_strip_noise(text))


def _is_darija(tokens: List[str], cfg: PreprocessConfig) -> bool:
	if not tokens:
		return False
	darija_hits = sum(1 for t in tokens if t in _DARIJA_MARKERS)
	msa_hits = sum(1 for t in tokens if t in _MSA_MARKERS)
	total_hits = darija_hits + msa_hits
	if darija_hits == 0 and msa_hits == 0:
		# Keep if comment has mixed latin + arabic, typical Darija writing.
		has_arabic = any(_ARABIC_LETTERS_RE.search(t) for t in tokens)
		has_latin = any(_LATIN_LETTERS_RE.search(t) for t in tokens)
		has_arabizi_digits = any(re.search(r"[2379]", t) for t in tokens)
		return (has_arabic and has_latin) or has_arabizi_digits
	if total_hits == 0:
		return False
	return (darija_hits / total_hits) >= cfg.darija_ratio_threshold


def _has_short_sentiment(tokens: List[str]) -> bool:
	return any(tok in _SENTIMENT_LEXICON for tok in tokens)


def preprocess_comment(text: str, cfg: PreprocessConfig) -> Optional[Dict[str, object]]:
	if not isinstance(text, str):
		return None

	if cfg.drop_emoji_only and _is_emoji_only(text):
		return None

	cleaned = _basic_clean(text)
	if not cleaned or not _has_letters(cleaned):
		return None

	tokens = _remove_stopwords(_tokenize(cleaned))
	if not tokens:
		return None

	if cfg.keep_short_if_sentiment and len(tokens) < cfg.min_tokens:
		if not _has_short_sentiment(tokens):
			return None
	elif len(tokens) < cfg.min_tokens:
		return None

	if not _is_darija(tokens, cfg):
		return None

	return {
		"text_clean": cleaned,
		"tokens": tokens,
	}


def _load_json(path: str) -> List[Dict[str, object]]:
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def _save_json(path: str, data: List[Dict[str, object]]) -> None:
	with open(path, "w", encoding="utf-8") as handle:
		json.dump(data, handle, ensure_ascii=False, indent=2)


def run_preprocessing(
	raw_path: str = RAW_DATA_PATH,
	output_path: str = PROCESSED_DATA_PATH,
	cfg: Optional[PreprocessConfig] = None,
) -> Dict[str, int]:
	if cfg is None:
		cfg = PreprocessConfig()

	raw_items = _load_json(raw_path)
	processed: List[Dict[str, object]] = []
	counters = Counter()

	for item in raw_items:
		text = item.get("text", "")
		result = preprocess_comment(text, cfg)
		if result is None:
			counters["dropped"] += 1
			continue

		enriched = {
			"comment_id": item.get("comment_id"),
			"text": text,
			"text_clean": result["text_clean"],
			"tokens": result["tokens"],
			"video_id": item.get("video_id"),
			"author": item.get("author"),
			"date": item.get("date"),
			"likes": item.get("likes"),
			"channel": item.get("channel"),
		}
		processed.append(enriched)
		counters["kept"] += 1

	_save_json(output_path, processed)
	counters["total"] = len(raw_items)
	return dict(counters)


if __name__ == "__main__":
	stats = run_preprocessing()
	print(f"Total: {stats.get('total', 0)}")
	print(f"Kept: {stats.get('kept', 0)}")
	print(f"Dropped: {stats.get('dropped', 0)}")
