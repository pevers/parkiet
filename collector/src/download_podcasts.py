import os
import requests
import feedparser
import hashlib
import time
from azure.storage.blob import BlobServiceClient

data = [
    "De Wadden, een andere oorlog",
    "Keuringsdienst van Waarde",
    "Groeten uit Grolloo",
    # "Maarten van Rossem en Tom Jessen",   # Vervelende stem
    "Onaantastbaar",
    "Geschiedenis Inside",
    "Boekestijn en De Wijk",
    "De Barbie Tapes",
    "De Ongelooflijke Podcast",
    "Daphne's Hulptroepen",
    "de Volkskrant Elke Dag",
    "De zaak ontleed",
    "AI Report",
    "NRC Vandaag",
    "De Oranjezomer",
    "Moordzaken",
    "Operatie Onthechting",
    "In De Waaier",
    "Filosofie is makkelijker als je denkt",
    "Napleiten",
    "NOS Met het Oog op Morgen",
    "Aaf en Lies lossen het wel weer op",
    "De Vuurwerkramp",
    "Omdenken Podcast",
    "De Zaak X",
    "Het Fortuin Carlier",
    "Over Leven",
    "De Jodenjager",
    "De Spindoctors",
    "Verraad in de geschiedenis",
    "Weer een dag",
    "Mr. Big",
    "De Jortcast",
    "Zolang het leuk is",
    "De Dag",
    "Niet In Mijn Buik",
    "De Stemming van Vullings en De Rooy",
    "Het Uur",
    "The Rest Is History",
    "FD Dagkoers",
    "AD Voetbal podcast",
    "Eén Grote Familie",
    "De Grote Podcastlas",
    "Nooit meer slapen",
    "F1 Aan Tafel",
    "NRC Onbehaarde Apen",
    "De Boordradio",
    "The Diary Of A CEO with Steven Bartlett",
    "Zo, Opgelost",
    "Zo simpel is het niet – Stellinga & Schinkel over economie",
    "Echt Gebeurd",
    "Kick-off met Valentijn Driessen",
    "POM - Een podcast over media, cultuur, technologie en ondernemen",
    "Geen tijd over",
    "Proces X",
    "Veldheren",
    "The Mel Robbins Podcast",
    "Moorden in de jaren...",
    "Betrouwbare Bronnen",
    "Argos Actueel",
    "Bureau Sport Podcast",
    "Pro Forma",
    "Vandaag Inside",
    "Het Haagse Achterhuis",
    "Liefste Lies",
    "Europa Draait Door",
    "Jacob, dood in Qatar",
    "Marc-Marie & Isa Vinden Iets",
    "De Universiteit van Nederland Podcast",
    "De Publieke Tribune",
    "Chantal & Tina",
    "NOS Formule 1-Podcast",
    "In de greep van Gaslighting",
    "Ondertussen in de kosmos",
    "Bouta's laatste zet | Argos Series",
    "Zelfspodcast",
    "Dutch Dragons",
    "De Nederlandse bewaaksters van Auschwitz",
    "RADIO BOOS",
    "Bureau Warmoesstraat 2 | Lammert & Babs",
    "Bubbels!",
    "Bingomaffia",
    "Vandaag in de Geschiedenis",
    "AI, je nieuwe collega",
    "De Vijftigers",
    "De Correspondent",
    "Met Groenteman in de kast",
    "Wat nu? met Diederik Samsom & Mathijs Bouman",
    "MISCHA!",
    "Teun en Gijs vertellen alles",
    "De Nieuwe Wereld",
    "De Restaurantmoord",
    "Etenstijd!",
    "NRC Haagse Zaken",
    "De Oorlog Verklaard",
    "NOS Voetbalpodcast",
    "Zware Jongens",
    "Het Spoor Terug",
    "De Rode Lantaarn",
    "Waanzinnig Land met Johan Fretz",
    "Schaduwoorlog",
    "GRAPES Podcast",
    "De Grote Tech Show | BNR",
    "Bo & Pauline: Genoeg over ons",
    "The Ezra Klein Show",
    "De Snapchatmoord",
    "Follow the Money",
    "Eerst dit",
    "Kunstof",
    "Boskamp & Kleine Gijp",
    "OVTBureau Buitenland",
    "Blokhuis de Podcast",
    "Spijkers met Koppen",
    "FOUT",
    "In Het Wiel",
    "Culturele bagage",
    "Help, ik heb een puber!",
    "Ochtendnieuws | BNR",
    "Het Digitale Front",
    "Amerika Podcast",
    "Stoute schoenen",
    "KINDEREN!!!!",
    "We zijn toch niet gek?",
    "Vik & Gert",
    "Maffe Meesterwerken",
    "De Technoloog | BNR",
    "De Podcast Psycholoog",
    "De Mediaweek",
    "Ik heb levenslang | Pointer Podcasts",
    "Cold cases: Tegen het licht",
    "Niemandsland",
    "Tweewielers",
    "Brussen & Veelo Podcast",
    "Geschiedenis voor herbeginners",
    "Dit is de Bijbel",
    "De Wielrenners van Voskuil",
    "De Kist",
    "DE GROTE PLAAT",
    "Wat blijft",
    "Afhameren met Wouter de Winther",
    "Die podcast over routines",
    "Global News Podcast",
    "Obscure Figuren",
    "Live Slow Ride Fast Podcast",
    "De Zelfhulpvraag",
    "Voetbalpraat",
    "Via Via",
    "Over de liefde",
    "Ziggo Sport: Race Café",
    "Van Dis Ongefilterd",
    "Big Time",
    "Buitenhof",
    "De Communicado's",
    "De liefde van nu",
    "Voorheen Schaamteloos Randstedelijk (VSR)",
    "Geuze & Gorgels",
    "Jong Beleggen, de podcast",
    "Nieuwe bazen in de zorg",
    "Maak Afvallen Makkelijk",
    "Het Misdaadbureau",
    "Tina's TV Update",
    "Veroordeeld",
    "Radio Vrij Nederland",
    "Op mijn Netvlies",
    "Doorzetters | met Ruud Hendriks en Richard Bross",
    "Van Bekhovens Britten | BNR",
    "Bloedheet & Tranen - de overgang zonder onzin",
    "Lang verhaal kort",
    "Ongeboren Leiders",
    "Hoe overleef ik werken met Gen Z | BusinessWise",
    "VI ZSM",
    "Drie Kwartjes",
    "Oorlogsmist",
    "De Geweldloze Podcast - Over opvoeden en zo!",
]

PODCASTINDEX_API_KEY = os.getenv("PODCASTINDEX_API_KEY", "UQRW8UYWQEEYEWBKEBT9")
PODCASTINDEX_API_SECRET = os.getenv(
    "PODCASTINDEX_API_SECRET", "TrkW#fcfBwD^Mj#wRXmy^aFeCNVtrHf^mhbJFP8V"
)


def get_podcast_feed_url(title):
    url = "https://api.podcastindex.org/api/1.0/search/bytitle"
    headers = get_podcastindex_headers()
    params = {"q": title}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    time.sleep(1)  # Rate limiting: max 1 request per second
    results = response.json()
    feeds = results.get("feeds", [])
    if not feeds:
        print(f"No feeds found for {title}")
        return None
    return feeds[0]["url"]


def get_podcastindex_headers():
    now = int(time.time())
    auth_string = PODCASTINDEX_API_KEY + PODCASTINDEX_API_SECRET + str(now)
    auth_hash = hashlib.sha1(auth_string.encode("utf-8")).hexdigest()
    return {
        "User-Agent": "mokumai/1.0",
        "X-Auth-Date": str(now),
        "X-Auth-Key": PODCASTINDEX_API_KEY,
        "Authorization": auth_hash,
    }


def download_episodes(feed_url, title):
    feed = feedparser.parse(feed_url)

    # Azure Blob Storage setup
    AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_CONTAINER_NAME = "podcasts"
    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_CONNECTION_STRING
    )

    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
    for entry in feed.entries:
        audio_url = None
        for link in entry.get("links", []):
            if link.get("type", "").startswith("audio"):
                audio_url = link.get("href")
                break
        if not audio_url:
            continue
        episode_title = entry.get("title", "episode")
        safe_title = "".join(
            c for c in episode_title if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        blob_path = f"{title}/{safe_title}.mp3"
        blob_client = container_client.get_blob_client(blob_path)
        if blob_client.exists():
            print(f"Already in Azure: {blob_path}, skipping local download.")
            continue
        # Download locally if not in Azure
        local_dir = os.path.join("data", "podcasts", title)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f"{safe_title}.mp3")
        if os.path.exists(local_path):
            print(f"Already downloaded locally: {local_path}")
            continue
        print(f"Downloading {audio_url} to {local_path}")
        try:
            with requests.get(audio_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            time.sleep(1)  # Rate limiting: max 1 request per second
        except Exception as e:
            print(f"Failed to download {audio_url}: {e}")


def main():
    for title in data:
        print(f"Processing: {title}")
        feed_url = get_podcast_feed_url(title)
        if feed_url:
            download_episodes(feed_url, title)


if __name__ == "__main__":
    main()
