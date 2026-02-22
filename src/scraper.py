# src/scraper.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from googleapiclient.discovery import build
import json
import time
from config import YOUTUBE_API_KEY, RAW_DATA_PATH, MAX_COMMENTS


class YouTubeScraper:

    def __init__(self):
        self.youtube = build("youtube", "v3",
                             developerKey=YOUTUBE_API_KEY)

    def get_channel_id(self, channel_name):
        request = self.youtube.search().list(
            q=channel_name,
            part="snippet",
            type="channel",
            maxResults=1
        )
        response = request.execute()

        if response["items"]:
            channel_id = response["items"][0]["id"]["channelId"]
            title = response["items"][0]["snippet"]["title"]
            print(f"✅ Chaîne trouvée : {title} ({channel_id})")
            return channel_id
        else:
            print(f"❌ Chaîne '{channel_name}' non trouvée")
            return None

    def get_channel_videos(self, channel_id, date_start, date_end,
                           max_videos=200):
        video_ids = []
        next_page_token = None

        while len(video_ids) < max_videos:
            request = self.youtube.search().list(
                channelId=channel_id,
                part="id",
                type="video",
                order="date",
                publishedAfter=date_start,
                publishedBefore=date_end,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response["items"]:
                video_ids.append(item["id"]["videoId"])

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        print(f"  📹 {len(video_ids)} vidéos trouvées")
        return video_ids

    def get_comments(self, video_id, max_comments=500, seen_ids=None):
        comments = []
        next_page_token = None
        seen_ids = seen_ids or set()

        while len(comments) < max_comments:
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
                    if comment_id in seen_ids:
                        continue
                    comments.append({
                        "comment_id": comment_id,
                        "text": snippet["textDisplay"],
                        "video_id": video_id,
                        "author": snippet["authorDisplayName"],
                        "date": snippet["publishedAt"],
                        "likes": snippet["likeCount"],
                    })
                    seen_ids.add(comment_id)

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

                time.sleep(0.5)

            except Exception as e:
                print(f"    ⚠️ Erreur : {e}")
                break

        return comments

    def scrape_channel(self, channel_name, date_start, date_end,
                       max_comments_per_channel=10000, max_videos=200,
                       seen_ids=None):
        print(f"\n{'='*50}")
        print(f"🔍 Chaîne : {channel_name}")
        print(f"{'='*50}")

        channel_id = self.get_channel_id(channel_name)
        if not channel_id:
            return []

        video_ids = self.get_channel_videos(
            channel_id, date_start, date_end, max_videos=max_videos
        )

        all_comments = []

        for i, video_id in enumerate(video_ids):
            print(f"  📹 Vidéo {i+1}/{len(video_ids)} : {video_id}")

            comments = self.get_comments(video_id, seen_ids=seen_ids)

            for comment in comments:
                comment["channel"] = channel_name

            all_comments.extend(comments)

            print(f"    📝 {len(comments)} commentaires"
                  f" (total: {len(all_comments)})")

            if len(all_comments) >= max_comments_per_channel:
                print(f"  ✅ Objectif atteint : {len(all_comments)}")
                break

            time.sleep(0.5)

        batch_path = f"data/raw/batch_{channel_name.replace(' ', '_')}.json"
        self.save(all_comments, batch_path)

        return all_comments

    def _load_existing(self, path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
            return {c.get("comment_id"): c for c in items if c.get("comment_id")}
        except Exception as e:
            print(f"⚠️ Impossible de lire {path}: {e}")
            return {}

    def scrape_all_channels(self):
        channels = [
            "Hespress",
            "Simo Sedraty",
            "Kifache TV",
            "i3lamtv",
            "Kawalis",
            "Marouane53",
            "Raw Soueelt",
        ]

        # Période : janvier 2025 → février 2026
        date_start = "2025-01-01T00:00:00Z"
        date_end = "2026-02-28T23:59:59Z"

        comments_per_channel = MAX_COMMENTS // len(channels)

        # Reprendre avec déduplication si fichier existe
        existing = self._load_existing(RAW_DATA_PATH)
        seen_ids = set(existing.keys())
        all_comments = list(existing.values())

        for channel in channels:
            comments = self.scrape_channel(
                channel_name=channel,
                date_start=date_start,
                date_end=date_end,
                max_comments_per_channel=comments_per_channel,
                seen_ids=seen_ids,
            )
            all_comments.extend(comments)

            print(f"\n📊 TOTAL jusqu'ici : {len(all_comments)} commentaires")

        # Supprimer les doublons
        unique = {c["comment_id"]: c for c in all_comments if c.get("comment_id")}
        all_comments = list(unique.values())

        # Sauvegarder
        self.save(all_comments, RAW_DATA_PATH)

        print(f"\n{'='*50}")
        print(f"✅ EXTRACTION TERMINÉE")
        print(f"📊 Total : {len(all_comments)} commentaires")
        print(f"💾 Sauvegardé dans {RAW_DATA_PATH}")
        print(f"{'='*50}")

        return all_comments

    def save(self, comments, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)
        print(f"  💾 Sauvegardé dans {path}")


if __name__ == "__main__":
    scraper = YouTubeScraper()
    scraper.scrape_all_channels()