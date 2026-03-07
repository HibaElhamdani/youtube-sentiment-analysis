import sys
import os
import json
import time
import re

# Add project root to sys.path to allow importing from config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from googleapiclient.discovery import build
from config import YOUTUBE_API_KEY, RAW_DATA_PATH

# =====================================
# CONFIGURATION
# =====================================

TOTAL_COMMENTS = 60000
DARiJA_PERCENT = 0.85
NEWS_PERCENT = 0.15

DARiJA_CHANNELS = [
    "Simo Sedraty",
    "Kawalis",
    "Marouane53",
    "Raw Soueelt"
]

NEWS_CHANNELS = [
    "Hespress",
    "Kifache TV",
    "i3lamtv"
]

MAX_VIDEOS_PER_CHANNEL = 150
MAX_COMMENTS_PER_VIDEO = 500

# =====================================
# LINGUISTIC MARKERS
# =====================================

DARIJA_MARKERS = {
    "واش","علاش","حيت","ديال","بزاف","راه",
    "عندو","عندها","كن","ماشي","هاد","كاين",
    "مكنش","غادي","دابا","خويا","اختي",
    "حنا","نتا","نتي","حيتاش","عافاك"
}

LATIN_DARIJA_MARKERS = [
    "7","9","3","5","2","gh","kh","ch"
]

MSA_STRONG_MARKERS = {
    "يجب","ينب��ي","حيث","لذلك","بالتالي",
    "وفق","علاوة","بينما","رغم",
    "المجتمع","السياسة","الاقتصاد",
    "المؤسسات","الإدارة","القانون",
    "الدستور","التنمية","الحكومة",
    "الرئيس","البرلمان","تصريح","بيان",
    "السلطات","الوزارة","الهيئة"
}

# =====================================
# TEXT CLEANING
# =====================================

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return text.strip()

# =====================================
# SMART DARIJA FILTER
# =====================================

def is_valid_comment(text: str) -> bool:
    if not text:
        return False

    text = clean_text(text)
    tokens = text.split()

    if not any(c.isalpha() for c in text):
        return False

    if len(tokens) < 2:
        return False

    if len(tokens) > 50:
        return False

    darija_score = 0
    msa_score = 0

    # Darija Arabic markers
    for w in DARIJA_MARKERS:
        if w in text:
            darija_score += 2

    # Darija Latin markers
    for marker in LATIN_DARIJA_MARKERS:
        if marker in text:
            darija_score += 1

    # MSA markers
    for w in MSA_STRONG_MARKERS:
        if w in text:
            msa_score += 2

    # ✅ Keep if Darija dominant or equal
    if darija_score >= msa_score:
        return True

    return False

# =====================================
# SCRAPER CLASS
# =====================================

class YouTubeScraper:
    def __init__(self):
        self.youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        self.all_comments = []
        self.seen_ids = set()
        self._load_existing_data()

    def _load_existing_data(self):
        if os.path.exists(RAW_DATA_PATH):
            with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
                self.all_comments = json.load(f)
                self.seen_ids = {c["comment_id"] for c in self.all_comments}
            print(f"📦 {len(self.seen_ids)} commentaires déjà chargés.")

    def get_channel_id(self, channel_name):
        request = self.youtube.search().list(
            q=channel_name,
            part="snippet",
            type="channel",
            maxResults=1
        )
        response = request.execute()
        if response["items"]:
            return response["items"][0]["id"]["channelId"]
        return None

    def get_channel_videos(self, channel_id):
        video_ids = []
        next_page_token = None

        while len(video_ids) < MAX_VIDEOS_PER_CHANNEL:
            request = self.youtube.search().list(
                channelId=channel_id,
                part="id",
                type="video",
                order="viewCount",
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response["items"]:
                video_ids.append(item["id"]["videoId"])

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return video_ids

    def get_comments(self, video_id):
        video_comments = []
        next_page_token = None

        while len(video_comments) < MAX_COMMENTS_PER_VIDEO:
            try:
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()

                for item in response["items"]:
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comment_id = item["id"]

                    if comment_id in self.seen_ids:
                        continue

                    text = snippet["textDisplay"]

                    if not is_valid_comment(text):
                        continue

                    comment_data = {
                        "comment_id": comment_id,
                        "text": text,
                        "video_id": video_id,
                        "author": snippet["authorDisplayName"],
                        "date": snippet["publishedAt"],
                        "likes": snippet["likeCount"],
                    }

                    video_comments.append(comment_data)
                    self.seen_ids.add(comment_id)

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

                time.sleep(0.3)

            except Exception as e:
                if "quotaExceeded" in str(e):
                    print("🛑 Quota dépassé. Sauvegarde...")
                    self.save_data()
                    sys.exit()
                break

        return video_comments

    def save_data(self):
        with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.all_comments, f, ensure_ascii=False, indent=2)

        print(f"💾 Sauvegarde: {len(self.all_comments)} commentaires")

    def scrape_balanced(self):
        target_darija = int(TOTAL_COMMENTS * DARiJA_PERCENT)
        target_news = int(TOTAL_COMMENTS * NEWS_PERCENT)

        print("🎯 Objectif Darija:", target_darija)
        print("🎯 Objectif News:", target_news)

        for channel_list, target in [
            (DARiJA_CHANNELS, target_darija),
            (NEWS_CHANNELS, target_news)
        ]:

            for channel in channel_list:
                print(f"\n🔍 Scraping {channel}")

                if len(self.all_comments) >= TOTAL_COMMENTS:
                    break

                cid = self.get_channel_id(channel)
                if not cid:
                    continue

                videos = self.get_channel_videos(cid)

                for vid in videos:
                    if len(self.all_comments) >= TOTAL_COMMENTS:
                        break

                    comments = self.get_comments(vid)

                    for c in comments:
                        c["channel"] = channel

                    self.all_comments.extend(comments)

                print(f"✅ {channel} terminé → total: {len(self.all_comments)}")
                self.save_data()

# =====================================
# RUN
# =====================================

if __name__ == "__main__":
    scraper = YouTubeScraper()
    scraper.scrape_balanced()