"""Preprocessing utilities for YouTube + Instagram comment sentiment analysis."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import emoji
except ImportError:
    emoji = None

from config import PROCESSED_DATA_PATH, RAW_DATA_PATH

# Chemin du fichier Instagram
INSTAGRAM_DATA_PATH = "data/raw/inst.json"

# ═══════════════════════════════════════════════════════
# REGEX DE NETTOYAGE
# ═══════════════════════════════════════════════════════

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#\w+", re.IGNORECASE)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_MULTISPACE_RE = re.compile(r"\s+")
_ARABIC_LETTERS_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN_LETTERS_RE = re.compile(r"[a-zA-Z]")
_DIGIT_RE = re.compile(r"\d")

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
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "\U00000023\U000020E3"
    "\U0000002A\U000020E3"
    "\U00000030-\U00000039\U000020E3"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)

_PUNCTUATION_RE = re.compile(
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~"
    r"\u060C\u061B\u061F\u0640\u066A\u066B\u066C\u066D"
    r"\u00AB\u00BB\u2018\u2019\u201C\u201D"
    r"\u2026\u2013\u2014"
    r"]+"
)


def _remove_all_emojis(text: str) -> str:
    if not text:
        return text
    if emoji is not None:
        return emoji.replace_emoji(text, replace=" ")
    return _EMOJI_RE.sub(" ", text)


# ═══════════════════════════════════════════════════════
# PATTERNS MORPHOLOGIQUES DARIJA
# ═══════════════════════════════════════════════════════

_DARIJA_VERB_PREFIX_RE = re.compile(
    r"^(كي|كا|كن|كت|غا|ما|تا|نا|يا)"
    r"[\u0600-\u06FF]{2,}"
)

_DARIJA_NEGATION_RE = re.compile(
    r"^ما[\u0600-\u06FF]{2,}(ش|شي)$"
)

_DARIJA_LATIN_NEGATION_RE = re.compile(
    r"^ma[a-z]{2,}(?:ch|sh)$",
    re.IGNORECASE
)

_DARIJA_LATIN_VERB_PREFIX_RE = re.compile(
    r"^(?:ka|ki|kan|kat|tan|tat|gha|ghan|na|ta|ya|ba|da)[a-z]{2,}$",
    re.IGNORECASE
)

_DARIJA_SUFFIX_RE = re.compile(
    r"[\u0600-\u06FF]{3,}(ني|تي|تو|ناه|هاش|نيش|ليه|ليها|ليهم|تهم|ناها|وها|وه|هم|كم|نا)$"
)

_DARIJA_CONTRACTION_RE = re.compile(
    r"^(دال|دل|فال|فل|بال|بل|ديال|فيه|بيه|عليه)[\u0600-\u06FF]*$"
)

_DARIJA_PLURAL_RE = re.compile(
    r"[\u0600-\u06FF]{3,}(ين|ات|وا|يو|يين)$"
)

_DARIJA_ELONGATION_RE = re.compile(
    r"^[a-z]*([aeiou])\1{2,}[a-z]*$",
    re.IGNORECASE
)


def _has_darija_morphology(tokens: List[str]) -> int:
    count = 0
    for tok in tokens:
        tok_lower = tok.lower()
        if _DARIJA_VERB_PREFIX_RE.match(tok):
            count += 1
        elif _DARIJA_NEGATION_RE.match(tok):
            count += 1
        elif _DARIJA_SUFFIX_RE.match(tok):
            count += 1
        elif _DARIJA_CONTRACTION_RE.match(tok):
            count += 1
        elif _DARIJA_PLURAL_RE.match(tok):
            count += 1
        elif _DARIJA_LATIN_NEGATION_RE.match(tok_lower):
            count += 1
        elif _DARIJA_LATIN_VERB_PREFIX_RE.match(tok_lower):
            count += 1
        elif _DARIJA_ELONGATION_RE.match(tok_lower):
            count += 1
    return count


# ═══════════════════════════════════════════════════════
# STOPWORDS
# ═══════════════════════════════════════════════════════

_STOPWORDS = frozenset({
    "و", "في", "على", "من", "الى", "إلى", "عن", "نعم",
    "هو", "هي", "هم", "ها", "هذا", "هذه", "ذلك", "تلك",
    "واش", "شنو", "ان", "أن", "إن",
    "a", "b", "c", "d", "de", "des", "du", "et", "le", "la", "les", 
    "un", "une", "est", "sont", "avec", "pour", "pas", "que", "qui",
    "the", "and", "to", "of", "for", "in", "is", "are", "it", "this",
    "that", "you", "i", "a", "an", "be", "have", "has", "was", "were",
    "هاد", "هادي", "هادو", "ديال", "لي", "اللي",
    "كان", "يكون", "غادي", "كاين", "راه",
    "عند", "كل", "بحال", "أو", "او", "يعني", "كيف",
    "باش", "حتى", "بلي", "شي", "داك", "ديك", "هداك",
    "فيه", "فيها", "معا", "عليه", "عليها",
    "بيه", "بيها", "ليه", "ليها",
    "منين", "كيفما", "فاش", "ملي",
    "غي", "غير", "تا", "حتا",
    "را", "يالاه", "اوا", "ايوا",
    "آش", "شكون", "كون", "الا", "إلا",
})


# ═══════════════════════════════════════════════════════
# DARIJA MARKERS
# ═══════════════════════════════════════════════════════

_DARIJA_MARKERS = frozenset({
    "بزاف", "واش", "عافاك", "حيت", "كاين", "غادي",
    "خايب", "زوين", "مزيان", "مكاين", "بغيت", "راك",
    "دابا", "شوية", "هاد", "ديال", "مسكين", "واعر",
    "كندير", "كنشوف", "كنقول", "كيقول", "كيدير",
    "كنموت", "بغا", "كلشي", "عندو", "عندها",
    "ماكاين", "خاصك", "خاصني", "بصح", "صافي", "زعما",
    "عاد", "يالاه", "ماشي", "كيفاش", "هادشي", "راه",
    "كاع", "والو", "يقدر", "خصك", "خصني",
    "مابغيتش", "كنبغي", "عجبني", "كرهت", "ضحكت", "بكيت",
    "خفت", "فرحت", "زعفت",
    "عرفتي", "عرفت", "عرفنا", "عرفو",
    "ندير", "نديرو", "دير", "ديري", "دارت", "دار",
    "بقات", "بقا", "بقيت", "بقينا",
    "تلعبو", "تلعب", "كيلعب", "كيلعبو",
    "كيخليني", "كيخلي", "خلاني", "خلات",
    "نشوف", "نشوفو", "شفت", "شفتي", "شافو",
    "نقول", "نقولو", "قلت", "قلتي", "قالو",
    "نمشي", "نمشيو", "مشيت", "مشا", "مشات",
    "نوقف", "وقفت", "وقف",
    "نكتب", "كتبت", "كتب",
    "نفهم", "فهمت", "فهمتي", "فهمنا",
    "نسمع", "سمعت", "سمعتي",
    "نخدم", "خدمت", "خدام", "خدامة",
    "جبت", "جيت", "جاب", "جابت",
    "بدا", "بديت", "بدينا",
    "درت", "دارت", "دارو", "درنا",
    "طلع", "طلعت", "طلعو",
    "دوز", "دوزت", "دوزها",
    "هديك", "هدا", "هدوك", "هادوك", "هادو",
    "ديالي", "ديالك", "ديالو", "ديالها",
    "ديالنا", "ديالكم", "ديالهم",
    "وصافي", "سافي", "واخا", "خلاص",
    "الاه", "بصاح", "صحيح",
    "يالله", "اجي", "سير", "جي",
    "ماعرفتش", "ماكانش", "مافهمتش",
    "ماعجبنيش", "ماقدرش", "ماعنديش",
    "مابقاش", "ماجاش", "ماكايناش",
    "مسمعتش", "مشفتش", "ماشفتش",
    "لايك", "فيديو", "واعرة",
    "شنو", "علاش", "فين", "كيفاش",
    "خويا", "ختي", "صاحبي", "صاحبتي",
    "وليدات", "دراري", "بنات", "ولاد",
    "فلوس", "خدمة", "قراية",
    "زنقة", "حومة", "بلاد", "بلادنا",
    "مغرب", "مغربي", "مغربية",
    "نتا", "نتي", "حنا", "هوما", "نتوما",
    "تبارك", "تبارك الله", "الله", "لله",
    "الفردة", "لفردة", "لوال", "لول",
    "المحسادة", "لمحسادة", "محسادة",
    "المهداوي", "لمهداوي", "مهداوي",
    "سخونيات", "سخونية",
    "فرشها", "فراش",
    "مسمنة", "مسمن",
    "دايرة", "داير",
    "خاوي", "خاوية",
    "ميكروب",
    "دبز", "دابز",
    "كيتعايرو",  # Nouveau mot d'Instagram
    "الدعاوي", "الزوينين",  # Nouveau
    # Arabizi
    "nt", "nta", "nti", "ntoma", "hna", "homa",
    "bghit", "bghiti", "bghina", "bghitk", "bghitkom", "bghitkoum",
    "ghadi", "ghada", "ghanmchi", "ghanndir",
    "bzaaf", "bzaf", "bzzaf",
    "7it", "hit",
    "3lach", "3la", "3lih", "3liha",
    "wach", "wash", "wesh",
    "chno", "chnou", "chnu",
    "fin", "feen", "fayn",
    "kifach", "kifash",
    "daba", "dab", "db",
    "bgh", "bghi", "bghina", "brina",
    "jib", "jibi", "jibha", "jibhom",
    "ahsan", "a7san", "7san",
    "b4ina", "brina", "bghina",
    "had", "hada", "hadi", "hadchi",
    "bda", "bdit", "bdina", "bdaw",
    "ash", "ach",
    "ken", "kan", "kant", "kano",
    "zouiin", "zwin", "zwiin", "zouiiin",
    "jit", "jiti", "jaw", "jina",
    "bakri", "bekri", "bakrii", "bakriii",
    "akhiran", "akhiiran", "akhiiiiran",
    "hchouma", "7chouma", "hchoumaa",
    "banliya", "banlia", "banlya",
    "deratni", "dartni", "dertni",
    "ferassi", "frassi", "frasi",
    "fiha",
    "khir", "kheir", "5ir",
    "malha", "mal7a",
    "darbo", "darbou", "darboh",
    "msamna", "msemna", "msmnna",
    "tlah", "tla7",
    "dayr", "dayer", "dayra", "dayrin",
    "lferda", "ferda", "lfirda",
    "lowl", "lwel", "louwel", "lowal",
    "dwzha", "dwzhaa", "dwezha", "dwzhaaa",
    "chkon", "chkoun",
    "skhoniyat", "skhouniyat", "s5oniyat",
    "dima", "diima", "diiima",
    "maghrib", "lmaghrib", "lmghrib", "mghrb",
    "lagitou", "lgito", "lgitoh",
    "lmehssada", "lm7ssada", "lmhssada", "mehssada",
    "lmehdaoui", "mehdaoui", "mehdawi", "lmhdawi",
    "zmatkoum", "zmatkom", "zmatkum",
    "frechha", "frachha",
    "gydato", "gidato", "guidato", "gidatou",
    "dabz", "dabez", "dabzz", "dabzza",
    "microb", "mikrob", "mkrob",
    "khawi", "khawya", "5awi",
    "kandir", "katdir", "kaydir",
    "kanchof", "katchof", "kaychof",
    "kan3rf", "kat3rf", "kay3rf",
    "kanbghi", "katbghi", "kaybghi",
    "kanmchi", "katmchi", "kaymchi",
    "mabghitch", "mafhmtch", "ma3rftch",
    "makynch", "makaynch",
    "mchitch",
    "mskine", "mskina", "mskin",
    "wlad", "wld", "wlidi",
    "kolchi", "kolshi", "kulshi",
    "mzyan", "mzyana", "mezyan", "meziana",
    "zwina", "zwin", "zween",
    "khoya", "khouya", "kho",
    "sahbi", "sa7bi", "s7abi",
    "dyal", "dial",
    "fhad",
    "bach", "bash",
    "bla", "bila",
    "ghi", "ghir", "gher",
    "ta", "taa",
    "rah",
    "sir", "siri",
    "aji",
    "safi", "saafi", "safii",
    "wa3r", "wa3ra", "waer", "waera",
    "khayb", "khayba", "5ayb",
    "walakin", "walakine",
    "machi", "mashi",
    "saraha", "sara7a", "sraha",
    "hamda", "hamdola", "hamdolah",
    "hamdoulah", "hamdoullah", "hamdoula", "hamdoulillah",
    "tbarkllah", "tbark", "tbarek",
    "machallah", "mashallah",
    "nchalah", "inchallah", "nshallah",
    "chhal", "ch7al",
    "chkoun", "shkoun",
    "3jbni", "ajebni", "3ajbni",
    "dir", "diri", "diro",
    "chouf", "chof",
    "smiya", "smiti",
    "zwaj",
    "khdma", "khedma",
    "flous", "flouss",
    "drari",
    "walo", "walou", "walloo",
    "wakha", "wakhha",
    "yak", "yakk", "yakkk",
    "ewa", "awa", "awaa", "ewaa",
    "ila", "ilaa", "ilaaa",
    "7ta", "hta", "7tta", "httaa",
    "bessa7", "bsa7", "bessa77",
    "3afak", "3afaak", "3afakk",
    "chokran", "choukran", "chokraan",
    "smeh", "sme7", "smehli", "sme7li",
    "blati", "blaati",
    "baraka", "barakaa", "barakaaa",
    "yalah", "yallah", "yallaah",
    "hania", "haniaa",
    "3lash", "3lachh",
    "finek", "finekk",
    "winek", "winekk",
    "chnahiya", "chnahia", "chnahiyaa",
    "labas", "labaas", "labass",
    "bikhir", "bi5ir",
    "hamdullah", "hamdollah", "7amdollah",
    "nchofo", "nchofou", "nchofouk",
    "tji", "tjii",
    "nmchi", "nmchii",
    "nrj3", "nrje3",
    "khouya", "khoya",
    "khti", "khtii",
    "wldi", "wldii",
    "bnti", "bntii",
    "rajli", "rajlii",
    "mrati", "mratii",
})


# ═══════════════════════════════════════════════════════
# GENERIC PHRASES
# ═══════════════════════════════════════════════════════

_GENERIC_PHRASES = frozenset({
    "ما شاء الله", "ماشاء الله",
    "اللهم بارك", "اللهم بارك فيك",
    "الله يبارك", "الله يبارك فيك",
    "بارك الله فيك", "بارك الله",
    "الحمد لله", "سبحان الله",
    "لا اله الا الله", "لا إله إلا الله",
    "اللهم صل على محمد", "اللهم صل وسلم",
    "ما شاء الله تبارك الرحمن",
    "سبحان الله وبحمده", "سبحان الله العظيم",
    "لا حول ولا قوة الا بالله",
    "لا حول ولا قوة إلا بالله",
    "استغفر الله العظيم", "استغفر الله",
    "الله اكبر", "الله أكبر",
})

_GENERIC_TOKENS = frozenset({
    "اللهم", "سبحان", "استغفر", "الحمد",
})


# ═══════════════════════════════════════════════════════
# SENTIMENT LEXICON
# ═══════════════════════════════════════════════════════

_SENTIMENT_LEXICON = frozenset({
    "خايب", "خايبة", "حامض", "يخ", "مقرف", "كارثة", "كارثه", 
    "فاشل", "فاشلة", "تفو", "قبيح", "قبيحة", "نقز", 
    "حشومة", "حشومه", "عيب", "حمار", "حمارة", "بغل",
    "زبل", "قمامة", "قمامه", "خنز", "كريه", "كريهة",
    "مغبن", "ساخط", "ضعيف", "ضعيفة", "صفر",
    "نازل", "نازلة", "خاسر", "خاسرة", "فاسد", "فاسدة",
    "مريض", "مريضة", "غبي", "غبية", "بليد", "بليدة", "جاهل", "جاهلة",
    "مكروه", "منافق", "كذاب", "كذابة", "حقير", "حقيرة",
    "وسخ", "وسخة", "قذر", "قذرة", "نجس",
    "ميكروب", "خاوي", "خاوية", "دابز",
    "tfo", "9bi7", "fashel", "fashla",
    "hchouma", "7chouma",
    "5ayb", "khayb", "khayba",
    "7mar", "7mara", "hmar", "hmara",
    "m9rf", "mqrf",
    "microb", "mikrob",
    "khawi", "khawya",
    "dabz", "dabez",
    "زوين", "زوينة", "زويين", "زويينة",
    "مزيان", "مزيانة", "مزيانين",
    "روعة", "رائع", "رائعة",
    "واعر", "واعرة", "واعرين",
    "طوب", "نضيف", "نضيفة",
    "بومبا", "ممتاز", "ممتازة",
    "عظيم", "عظيمة", "جميل", "جميلة",
    "حلو", "حلوة", "خطير", "خطيرة",
    "فابور", "قنبلة", "قنبله",
    "ديما", "فخر", "فخور",
    "شكرا", "شكراً", "تحية", "تحيه",
    "نجم", "نجمة", "بطل", "بطلة",
    "اسطورة", "اسطوره", "أسطورة",
    "حب", "حبيت", "عشق", "عشقت",
    "فرحان", "فرحانة", "سعيد", "سعيدة",
    "مبروك", "مبروكة", "تهنئة",
    "أحسن", "احسن", "خير",
    "top", "tooop", "toooop",
    "bravo", "bravooo",
    "3jbni", "3ajbni", "ajebni",
    "zwin", "zwina", "zween", "zweena",
    "zouiin", "zouiiin",
    "wa3r", "wa3ra", "waer", "waera",
    "mzyan", "mzyana", "mezyan",
    "tbarkllah", "tbark", "tbarek",
    "machallah", "mashallah",
    "nice", "cool", "great", "amazing", "awesome",
    "love", "loved", "best",
    "ahsan", "a7san",
    "khir", "kheir", "5ir",
})


# ═══════════════════════════════════════════════════════
# MSA MARKERS
# ═══════════════════════════════════════════════════════

_MSA_MARKERS = frozenset({
    "يجب", "ينبغي", "لكن", "لذلك", "بسبب",
    "الذي", "التي", "الذين", "اللاتي", "اللواتي",
    "حيث", "بالتالي", "كما", "وقد", "تم", "لدى",
    "أيضاً", "أيضا", "ايضا", "علاوة", "فضلاً", "فضلا",
    "بينما", "رغم", "مما", "إذ", "وفق", "وفقاً",
    "نحو", "خلال", "ضمن", "تجاه", "بشأن",
    "نظراً", "نظرا", "إثر", "اثر",
    "عقب", "سوى", "دون", "قبل", "بعد", "عبر", "ضد",
    "يتوجب", "يستلزم", "يتطلب", "يقتضي",
    "يُعد", "يعد", "تُعد", "تعد",
    "يُعتبر", "يعتبر", "تُعتبر", "تعتبر",
    "أكد", "أكدت", "يؤكد", "تؤكد",
    "أشار", "أشارت", "يشير", "تشير",
    "أوضح", "أوضحت", "يوضح", "توضح",
    "صرح", "صرحت", "يصرح", "تصرح",
    "المواطنون", "المواطنين", "الحكومة", "الدولة",
    "المجتمع", "السياسة", "الاقتصاد", "التنمية",
    "المؤسسات", "الإدارة", "القانون", "الدستور",
})


_RELIGIOUS_MARKERS_STRICT = frozenset({
    "اللهم", "رسول", "النبي", "المسلمين", "الدعاء",
    "صحبه", "أجمعين", "وسلم", "نبينا", "آله",
    "استغفر", "وبحمده",
    "صلى الله عليه وسلم", "عليه السلام",
    "رضي الله عنه", "رضي الله عنها",
})


_SPAM_PATTERNS = frozenset({
    "subscribe", "channel", "please", "follow", "check",
    "click", "link", "bio", "visit", "website",
    "free", "win", "gift", "money", "earn",
    "giveaway", "promotion", "discount", "offer",
    "bonjour", "merci", "salut", "comment", "pourquoi",
    "parce", "vraiment", "tellement", "jamais", "toujours",
    "aujourd", "demain", "hier",
})


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

@dataclass
class PreprocessConfig:
    min_tokens: int = 1
    darija_ratio_threshold: float = 0.4
    min_darija_hits: int = 1
    allow_mixed_script: bool = True
    allow_arabizi: bool = True
    allow_arabic_no_markers: bool = True
    arabic_no_markers_max_tokens: int = 18
    allow_latin_no_markers: bool = False
    latin_no_markers_max_tokens: int = 5
    keep_short_if_sentiment: bool = True
    drop_emoji_only: bool = True
    drop_religious_msa: bool = True
    drop_generic_phrases: bool = True
    drop_spam: bool = True


# ═══════════════════════════════════════════════════════
# FONCTIONS DE NETTOYAGE
# ═══════════════════════════════════════════════════════

def _normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("\u0640", "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = text.replace("ة", "ه")
    return text


def _normalize_elongation(text: str) -> str:
    text = re.sub(r"([\u0600-\u06FF])\1{2,}", r"\1\1", text)
    text = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)
    return text


def _strip_noise(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    text = _TIME_RE.sub(" ", text)
    text = _remove_all_emojis(text)
    return text


def _remove_punctuation(text: str) -> str:
    return _PUNCTUATION_RE.sub(" ", text)


def _basic_clean(text: str) -> str:
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = _strip_noise(text)
    text = _normalize_arabic(text)
    text = _normalize_elongation(text)
    text = text.lower()
    text = _remove_punctuation(text)
    text = re.sub(r"[^a-z0-9\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", " ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


# ═══════════════════════════════════════════════════════
# TOKENISATION
# ═══════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = text.split()
    filtered: List[str] = []
    for tok in tokens:
        if not tok:
            continue
        if _DIGIT_RE.search(tok):
            if _ARABIC_LETTERS_RE.search(tok) or _LATIN_LETTERS_RE.search(tok):
                filtered.append(tok)
            continue
        if _ARABIC_LETTERS_RE.search(tok) or _LATIN_LETTERS_RE.search(tok):
            filtered.append(tok)
    return filtered


def _remove_stopwords(tokens: Iterable[str]) -> List[str]:
    return [tok for tok in tokens if tok not in _STOPWORDS]


# ═══════════════════════════════════════════════════════
# FONCTIONS DE DÉTECTION
# ═══════════════════════════════════════════════════════

def _has_letters(text: str) -> bool:
    return bool(_ARABIC_LETTERS_RE.search(text) or _LATIN_LETTERS_RE.search(text))


def _is_emoji_only(text: str) -> bool:
    if not text:
        return True
    cleaned = _strip_noise(text)
    return not _has_letters(cleaned)


def _darija_stats(tokens: List[str]) -> Dict[str, object]:
    if not tokens:
        return {
            "darija_hits": 0, "darija_morphology": 0, "msa_hits": 0,
            "spam_hits": 0, "has_arabic": False, "has_latin": False,
            "has_arabizi_digits": False,
        }
    darija_hits = 0
    msa_hits = 0
    spam_hits = 0
    for t in tokens:
        t_lower = t.lower()
        if t in _DARIJA_MARKERS or t_lower in _DARIJA_MARKERS:
            darija_hits += 1
        if t in _MSA_MARKERS:
            msa_hits += 1
        if t_lower in _SPAM_PATTERNS:
            spam_hits += 1
    darija_morphology = _has_darija_morphology(tokens)
    has_arabic = any(_ARABIC_LETTERS_RE.search(t) for t in tokens)
    has_latin = any(_LATIN_LETTERS_RE.search(t) for t in tokens)
    has_arabizi_digits = any(re.search(r"[23579]", t) for t in tokens)
    return {
        "darija_hits": darija_hits, "darija_morphology": darija_morphology,
        "msa_hits": msa_hits, "spam_hits": spam_hits,
        "has_arabic": has_arabic, "has_latin": has_latin,
        "has_arabizi_digits": has_arabizi_digits,
    }


def _is_darija(tokens: List[str], cfg: PreprocessConfig) -> bool:
    if not tokens:
        return False
    stats = _darija_stats(tokens)
    darija_hits = stats["darija_hits"]
    darija_morphology = stats["darija_morphology"]
    msa_hits = stats["msa_hits"]
    spam_hits = stats["spam_hits"]
    total_darija = darija_hits + darija_morphology
    
    if total_darija > 0:
        if msa_hits > 0:
            total = total_darija + msa_hits
            ratio = total_darija / total
            if ratio < cfg.darija_ratio_threshold:
                return False
        return True
    if msa_hits > 0 and total_darija == 0:
        return False
    if stats["has_arabizi_digits"]:
        return True
    if stats["has_arabic"] and stats["has_latin"]:
        return True
    if stats["has_arabic"] and not stats["has_latin"]:
        if len(tokens) > cfg.arabic_no_markers_max_tokens:
            return False
        return True
    if stats["has_latin"] and not stats["has_arabic"]:
        if cfg.drop_spam and spam_hits > 0:
            spam_ratio = spam_hits / len(tokens)
            if spam_ratio > 0.3:
                return False
        for tok in tokens:
            tok_lower = tok.lower()
            if tok_lower in _DARIJA_MARKERS:
                return True
        for tok in tokens:
            tok_lower = tok.lower()
            if _DARIJA_LATIN_NEGATION_RE.match(tok_lower):
                return True
            if _DARIJA_LATIN_VERB_PREFIX_RE.match(tok_lower):
                return True
        if cfg.allow_latin_no_markers and len(tokens) <= cfg.latin_no_markers_max_tokens:
            return True
        return False
    return False


def _has_short_sentiment(tokens: List[str]) -> bool:
    for tok in tokens:
        tok_lower = tok.lower()
        if tok in _SENTIMENT_LEXICON or tok_lower in _SENTIMENT_LEXICON:
            return True
    return False


def _is_generic_phrase(cleaned: str, tokens: List[str]) -> bool:
    if cleaned in _GENERIC_PHRASES:
        return True
    if not tokens:
        return True
    if len(tokens) <= 4:
        generic_count = sum(1 for tok in tokens if tok in _GENERIC_TOKENS)
        if generic_count == len(tokens):
            return True
    return False


def _is_religious_msa(tokens: List[str], cleaned: str) -> bool:
    if cleaned in _GENERIC_PHRASES:
        return True
    if not tokens:
        return True
    non_stop_tokens = [t for t in tokens if t not in _STOPWORDS]
    if not non_stop_tokens:
        return True
    religious_count = sum(1 for t in non_stop_tokens if t in _RELIGIOUS_MARKERS_STRICT)
    if len(non_stop_tokens) > 0:
        religious_ratio = religious_count / len(non_stop_tokens)
        return religious_ratio > 0.5
    return True


def _is_spam(tokens: List[str], cfg: PreprocessConfig) -> bool:
    if not cfg.drop_spam:
        return False
    if not tokens:
        return False
    spam_count = sum(1 for t in tokens if t.lower() in _SPAM_PATTERNS)
    if len(tokens) > 0:
        spam_ratio = spam_count / len(tokens)
        return spam_ratio > 0.5
    return False


# ═══════════════════════════════════════════════════════
# PIPELINE PRINCIPALE
# ═══════════════════════════════════════════════════════

def _preprocess_comment_internal(
    text: str, cfg: PreprocessConfig
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    if not isinstance(text, str):
        return None, "not_text"
    text = text.strip()
    if not text:
        return None, "empty"
    if cfg.drop_emoji_only and _is_emoji_only(text):
        return None, "emoji_only"
    cleaned = _basic_clean(text)
    if not cleaned or not _has_letters(cleaned):
        return None, "empty_after_clean"
    full_tokens = _tokenize(cleaned)
    if not full_tokens:
        return None, "empty_after_tokenize"
    if _is_spam(full_tokens, cfg):
        return None, "spam"
    is_darija_text = _is_darija(full_tokens, cfg)
    if not is_darija_text:
        if cfg.drop_religious_msa and _is_religious_msa(full_tokens, cleaned):
            return None, "religious_msa"
        return None, "not_darija"
    if cfg.drop_generic_phrases and _is_generic_phrase(cleaned, full_tokens):
        return None, "generic_phrase"
    tokens = _remove_stopwords(full_tokens)
    if not tokens:
        return None, "only_stopwords"
    if len(tokens) < cfg.min_tokens:
        if cfg.keep_short_if_sentiment and _has_short_sentiment(tokens):
            pass
        else:
            return None, "too_short"
    stats = _darija_stats(full_tokens)
    lang_hint = "darija"
    if stats["has_arabic"] and stats["has_latin"]:
        lang_hint = "mixed"
    elif stats["has_arabizi_digits"] and not stats["has_arabic"]:
        lang_hint = "arabizi"
    elif not stats["has_arabic"] and stats["has_latin"]:
        lang_hint = "arabizi"
    return {
        "text_clean": cleaned,
        "tokens": tokens,
        "lang_hint": lang_hint,
    }, None


def preprocess_comment(text: str, cfg: Optional[PreprocessConfig] = None) -> Optional[Dict[str, object]]:
    if cfg is None:
        cfg = PreprocessConfig()
    result, _ = _preprocess_comment_internal(text, cfg)
    return result


# ═══════════════════════════════════════════════════════
# CHARGEMENT / SAUVEGARDE
# ═══════════════════════════════════════════════════════

def _load_json(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: str, data: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _load_instagram_data(path: str) -> List[Dict[str, object]]:
    """Charge et convertit les données Instagram au format standard."""
    if not os.path.exists(path):
        print(f"  ℹ️ Fichier Instagram non trouvé: {path}")
        return []
    
    try:
        with open(path, "r", encoding="utf-8") as handle:
            instagram_data = json.load(handle)
        
        converted = []
        for item in instagram_data:
            text = item.get("text", "")
            if not text or len(text.strip()) < 2:
                continue
            
            item_id = item.get("id", "")
            comment_id = f"ig_{item_id}" if not str(item_id).startswith("ig_") else item_id
            
            post_url = item.get("postUrl", "")
            shortcode = ""
            if "/p/" in post_url:
                shortcode = post_url.split("/p/")[1].split("/")[0]
            
            converted.append({
                "comment_id": comment_id,
                "text": text,
                "video_id": shortcode or post_url,
                "author": item.get("ownerUsername", "unknown"),
                "date": item.get("timestamp", ""),
                "likes": item.get("likesCount", 0),
                "channel": f"@{item.get('ownerUsername', 'instagram')}",
                "source": "instagram",
            })
            
            for reply in item.get("replies", []):
                reply_text = reply.get("text", "")
                if not reply_text or len(reply_text.strip()) < 2:
                    continue
                reply_id = reply.get("id", "")
                converted.append({
                    "comment_id": f"ig_{reply_id}",
                    "text": reply_text,
                    "video_id": shortcode or post_url,
                    "author": reply.get("ownerUsername", "unknown"),
                    "date": reply.get("timestamp", ""),
                    "likes": reply.get("likesCount", 0),
                    "channel": f"@{reply.get('ownerUsername', 'instagram')}",
                    "source": "instagram",
                })
        
        print(f"  ✅ {len(converted)} commentaires Instagram chargés")
        return converted
        
    except Exception as e:
        print(f"  ⚠️ Erreur chargement Instagram: {e}")
        return []


# ═══════════════════════════════════════════════════════
# EXÉCUTION PRINCIPALE
# ═══════════════════════════════════════════════════════

def run_preprocessing(
    raw_path: str = RAW_DATA_PATH,
    instagram_path: str = "data/raw/inst.json",
    output_path: str = PROCESSED_DATA_PATH,
    dropped_path: str = "data/processed/comments_dropped.json",
    cfg: Optional[PreprocessConfig] = None,
) -> Dict[str, int]:
    """Exécute le pipeline de prétraitement (YouTube + Instagram)."""

    if cfg is None:
        cfg = PreprocessConfig()

    print("\n📂 Chargement des données...")
    print(f"  📺 YouTube: {raw_path}")
    
    raw_items = []
    if os.path.exists(raw_path):
        raw_items = _load_json(raw_path)
        print(f"  ✅ {len(raw_items)} commentaires YouTube chargés")
    else:
        print(f"  ⚠️ Fichier YouTube non trouvé: {raw_path}")
    
    for item in raw_items:
        if "source" not in item:
            item["source"] = "youtube"
    
    print(f"  📸 Instagram: {instagram_path}")
    instagram_items = _load_instagram_data(instagram_path)
    
    all_items = raw_items + instagram_items
    
    seen_ids = set()
    unique_items = []
    for item in all_items:
        cid = item.get("comment_id")
        if cid and cid not in seen_ids:
            unique_items.append(item)
            seen_ids.add(cid)
    all_items = unique_items
    
    youtube_count = sum(1 for i in all_items if i.get("source") == "youtube")
    instagram_count = sum(1 for i in all_items if i.get("source") == "instagram")
    
    print(f"\n📊 Total avant traitement:")
    print(f"  📺 YouTube  : {youtube_count}")
    print(f"  📸 Instagram: {instagram_count}")
    print(f"  📊 Total    : {len(all_items)}")
    
    print("\n🔄 Prétraitement en cours...")
    
    processed: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    counters: Counter = Counter()

    for item in all_items:
        text = item.get("text", "")
        result, reason = _preprocess_comment_internal(text, cfg)

        if result is None:
            counters["dropped"] += 1
            if reason:
                counters[f"dropped_{reason}"] += 1
            dropped.append({
                "comment_id": item.get("comment_id"),
                "text": text,
                "video_id": item.get("video_id"),
                "author": item.get("author"),
                "date": item.get("date"),
                "likes": item.get("likes"),
                "channel": item.get("channel"),
                "source": item.get("source", "unknown"),
                "drop_reason": reason,
            })
            continue

        enriched = {
            "comment_id": item.get("comment_id"),
            "text": text,
            "text_clean": result["text_clean"],
            "tokens": result["tokens"],
            "date": item.get("date"),
            "likes": item.get("likes"),
            "channel": item.get("channel"),
            "lang_hint": result["lang_hint"],
        }
        processed.append(enriched)
        counters["kept"] += 1

    unique_map: Dict[str, Dict[str, object]] = {}
    for item in processed:
        text_clean = item.get("text_clean")
        if isinstance(text_clean, str) and text_clean not in unique_map:
            unique_map[text_clean] = item
        else:
            counters["duplicates_removed"] += 1
    
    processed = list(unique_map.values())
    counters["final_count"] = len(processed)

    final_youtube = sum(1 for i in processed if i.get("source") == "youtube")
    final_instagram = sum(1 for i in processed if i.get("source") == "instagram")
    
    counters["final_youtube"] = final_youtube
    counters["final_instagram"] = final_instagram

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(dropped_path), exist_ok=True)
    
    _save_json(output_path, processed)
    _save_json(dropped_path, dropped)

    counters["total"] = len(all_items)
    return dict(counters)


# ═══════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    stats = run_preprocessing()
    
    print("\n" + "=" * 60)
    print(" RÉSULTATS DU PRÉTRAITEMENT")
    print("=" * 60)
    
    print(f"\n{'ENTRÉE':<40}")
    print(f"  {'Total commentaires bruts':<35} : {stats.get('total', 0):>8}")
    
    print(f"\n{' SORTIE':<40}")
    print(f"  {'Gardés (avant dédup)':<35} : {stats.get('kept', 0):>8}")
    print(f"  {'Doublons supprimés':<35} : {stats.get('duplicates_removed', 0):>8}")
    print(f"  {'Commentaires finaux':<35} : {stats.get('final_count', 0):>8}")
    
    print(f"\n{' PAR SOURCE':<40}")
    print(f"  {'📺 YouTube':<35} : {stats.get('final_youtube', 0):>8}")
    print(f"  {'📸 Instagram':<35} : {stats.get('final_instagram', 0):>8}")
    
    print(f"\n{'🗑️ SUPPRIMÉS':<40}")
    print(f"  {'Total supprimés':<35} : {stats.get('dropped', 0):>8}")
    
    print(f"\n{'📋 Détail des suppressions:':<40}")
    drop_reasons = {k: v for k, v in stats.items() if k.startswith("dropped_")}
    for key in sorted(drop_reasons.keys(), key=lambda x: -drop_reasons[x]):
        reason = key.replace("dropped_", "")
        print(f"    {reason:<32} : {drop_reasons[key]:>6}")
    
    print("\n" + "=" * 60)