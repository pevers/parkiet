#!/usr/bin/env python3
"""
Parel Radio Podcast Scraper

This script fetches all episodes from the Parel Radio podcast by scraping
pages 1-40 from https://www.nporadio1.nl/podcasts/parel-radio
It extracts episode titles and MP3 URLs from the __NEXT_DATA__ script tag.
"""

import requests
import json
import time
import os
import re
from bs4 import BeautifulSoup
from urllib.parse import quote
from typing import Dict, List, Optional
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ParelRadioScraper:
    def __init__(self, output_dir: str = "../data/hoorspelen/Parel Radio"):
        self.base_url = "https://www.nporadio1.nl/podcasts/parel-radio"
        self.output_dir = output_dir
        self.session = requests.Session()

        # Set user agent to be polite
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.episodes: List[Dict] = []

    def fetch_page(self, page_num: int) -> Optional[str]:
        """Fetch a single page and return the HTML content."""
        url = f"{self.base_url}?page={page_num}"
        logger.info(f"Fetching page {page_num}: {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch page {page_num}: {e}")
            return None

    def extract_next_data(self, html_content: str) -> Optional[dict]:
        """Extract and parse the __NEXT_DATA__ script tag."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the __NEXT_DATA__ script tag
        script_tag = soup.find(
            "script", {"id": "__NEXT_DATA__", "type": "application/json"}
        )

        if not script_tag:
            logger.error("Could not find __NEXT_DATA__ script tag")
            return None

        try:
            return json.loads(script_tag.string)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from __NEXT_DATA__: {e}")
            return None

    def extract_episodes(self, next_data: dict) -> List[Dict]:
        """Extract episode information from the parsed JSON data."""
        episodes = []

        try:
            episodes_data = next_data["props"]["pageProps"]["episodes"]

            for episode in episodes_data:
                episode_info = {
                    "id": episode.get("id"),
                    "name": episode.get("name"),
                    "description": episode.get("description"),
                    "published_at": episode.get("publishedAt"),
                    "image_url": episode.get("imageUrl"),
                    "url": episode.get("url"),
                    "mp3_url": None,
                }

                # Extract MP3 URL from player parameters
                if "player" in episode and "parameters" in episode["player"]:
                    for param in episode["player"]["parameters"]:
                        if param.get("name") == "progressive":
                            episode_info["mp3_url"] = param.get("value")
                            break

                episodes.append(episode_info)
                logger.info(f"Extracted episode: {episode_info['name']}")

        except KeyError as e:
            logger.error(f"Failed to extract episodes from JSON data: {e}")

        return episodes

    def save_episode_data(self, episodes: List[Dict], filename: str = "episodes.json"):
        """Save episode data to JSON file."""
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(episodes, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(episodes)} episodes to {filepath}")

    def scrape_all_pages(
        self, start_page: int = 1, end_page: int = 40, delay: float = 2.0
    ):
        """Scrape all pages from start_page to end_page."""
        all_episodes = []

        for page_num in range(start_page, end_page + 1):
            # Fetch the page
            html_content = self.fetch_page(page_num)
            if not html_content:
                logger.warning(f"Skipping page {page_num} due to fetch error")
                continue

            # Extract JSON data
            next_data = self.extract_next_data(html_content)
            if not next_data:
                logger.warning(f"Skipping page {page_num} due to JSON extraction error")
                continue

            # Extract episodes
            episodes = self.extract_episodes(next_data)
            all_episodes.extend(episodes)

            # Check if we've reached the last page
            try:
                pagination = next_data["props"]["pageProps"]["pagination"]
                current_page = pagination.get("currentPage", page_num)
                max_page = pagination.get("maxPage", 40)

                logger.info(
                    f"Page {current_page}/{max_page}, found {len(episodes)} episodes"
                )

                if current_page >= max_page:
                    logger.info("Reached the last page, stopping")
                    break

            except KeyError:
                logger.warning("Could not determine pagination info")

            # Be respectful with delays
            if page_num < end_page:
                logger.info(f"Waiting {delay} seconds before next request...")
                time.sleep(delay)

        # Remove duplicates based on episode ID
        unique_episodes = []
        seen_ids = set()

        for episode in all_episodes:
            if episode["id"] not in seen_ids:
                unique_episodes.append(episode)
                seen_ids.add(episode["id"])

        logger.info(f"Found {len(unique_episodes)} unique episodes total")

        # Save the data in multiple formats
        self.save_episode_data(unique_episodes, "all_episodes.json")

        return unique_episodes

    def create_safe_filename(self, episode_name: str, episode_id: str) -> str:
        """Create a safe filename for the episode."""
        # Remove unsafe characters and limit length
        safe_name = re.sub(r"[^\w\s-]", "", episode_name).strip()
        safe_name = re.sub(r"[-\s]+", "-", safe_name)

        # Limit filename length and add episode ID to ensure uniqueness
        if len(safe_name) > 100:
            safe_name = safe_name[:100]

        return f"{episode_id}_{safe_name}.mp3"

    def download_episode(self, episode: Dict) -> bool:
        """Download a single episode MP3 file."""
        mp3_url = episode.get("mp3_url")
        if not mp3_url:
            logger.warning(f"No MP3 URL for episode: {episode.get('name', 'Unknown')}")
            return False

        filename = self.create_safe_filename(episode["name"], episode["id"])
        filepath = os.path.join(self.output_dir, filename)

        # Skip if file already exists
        if os.path.exists(filepath):
            logger.info(f"File already exists, skipping: {filename}")
            return True

        try:
            logger.info(f"Downloading: {episode['name']}")
            logger.info(f"URL: {mp3_url}")

            response = self.session.get(mp3_url, timeout=60, stream=True)
            response.raise_for_status()

            # Get file size for progress tracking
            total_size = int(response.headers.get("content-length", 0))

            with open(filepath, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.info(
                                f"Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)"
                            )

            logger.info(f"Successfully downloaded: {filename}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to download {episode['name']}: {e}")
            # Clean up partial download
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {episode['name']}: {e}")
            # Clean up partial download
            if os.path.exists(filepath):
                os.remove(filepath)
            return False

    def download_episodes(
        self, episodes_file: str = "all_episodes.json", delay: float = 1.0
    ) -> int:
        """Download all episodes from the JSON file."""
        episodes_path = os.path.join(self.output_dir, episodes_file)

        if not os.path.exists(episodes_path):
            logger.error(f"Episodes file not found: {episodes_path}")
            return 0

        try:
            with open(episodes_path, "r", encoding="utf-8") as f:
                episodes = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse episodes file: {e}")
            return 0

        episodes_with_mp3 = [ep for ep in episodes if ep.get("mp3_url")]
        total_episodes = len(episodes_with_mp3)

        logger.info(f"Found {total_episodes} episodes with MP3 URLs to download")

        successful_downloads = 0
        failed_downloads = 0

        for i, episode in enumerate(episodes_with_mp3, 1):
            logger.info(f"Processing episode {i}/{total_episodes}")

            if self.download_episode(episode):
                successful_downloads += 1
            else:
                failed_downloads += 1

            # Be respectful with delays between downloads
            if i < total_episodes and delay > 0:
                logger.info(f"Waiting {delay} seconds before next download...")
                time.sleep(delay)

        logger.info(
            f"Download summary: {successful_downloads} successful, {failed_downloads} failed"
        )
        return successful_downloads

    def load_episodes_from_file(
        self, episodes_file: str = "all_episodes.json"
    ) -> List[Dict]:
        """Load episodes from the JSON file."""
        episodes_path = os.path.join(self.output_dir, episodes_file)

        if not os.path.exists(episodes_path):
            logger.error(f"Episodes file not found: {episodes_path}")
            return []

        try:
            with open(episodes_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse episodes file: {e}")
            return []


def main():
    """Main function to run the scraper."""
    parser = argparse.ArgumentParser(description="Parel Radio Podcast Scraper")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download episodes from existing all_episodes.json file",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Starting page number for scraping (default: 1)",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=40,
        help="Ending page number for scraping (default: 40)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between requests in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=1.0,
        help="Delay between downloads in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/hoorspelen/Parel Radio",
        help="Output directory for data and downloads",
    )

    args = parser.parse_args()

    scraper = ParelRadioScraper(output_dir=args.output_dir)

    if args.download_only:
        logger.info("Starting download of existing episodes...")
        downloaded_count = scraper.download_episodes(delay=args.download_delay)
        logger.info(f"Download completed! Downloaded {downloaded_count} episodes.")
    else:
        logger.info("Starting Parel Radio podcast scraping...")
        episodes = scraper.scrape_all_pages(
            start_page=args.start_page, end_page=args.end_page, delay=args.delay
        )

        logger.info(f"Scraping completed! Found {len(episodes)} episodes.")
        logger.info(f"Data saved to: {scraper.output_dir}")

        # Print some statistics
        episodes_with_mp3 = [ep for ep in episodes if ep.get("mp3_url")]
        logger.info(f"Episodes with MP3 URLs: {len(episodes_with_mp3)}")

        # Ask if user wants to download episodes
        try:
            response = (
                input("\nDo you want to download all episodes now? (y/n): ")
                .strip()
                .lower()
            )
            if response in ["y", "yes"]:
                logger.info("Starting download of episodes...")
                downloaded_count = scraper.download_episodes(delay=args.download_delay)
                logger.info(
                    f"Download completed! Downloaded {downloaded_count} episodes."
                )
        except KeyboardInterrupt:
            logger.info("\nDownload skipped by user.")


if __name__ == "__main__":
    main()
