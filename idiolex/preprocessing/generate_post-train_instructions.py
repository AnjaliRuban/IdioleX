"""
generate_dialect_instructions.py — Build dialectal Arabic instruction data for GRPO training.

Downloads/loads 9 dialectal Arabic corpora, then uses an LLM (via litellm) to generate
AL-QASIDA-style instruction prompts in the target dialect.

Produces two types of instructions:
  1. MONOLINGUAL: Dialectal instruction wrapping a dialectal sentence
     (e.g. "أكمل الجملة التالية: ..." in Egyptian Arabic)
  2. TRANSLATION: Bitext-based instructions asking to translate to/from English
     (e.g. "ترجم الجملة دي للإنجليزي: ...")

Usage:
    # Set your litellm credentials
    export LITELLM_API_KEY=sk-...
    export LITELLM_API_BASE_URL=https://...

    # Generate instructions (downloads datasets automatically where possible)
    python generate_dialect_instructions.py \
        --output_dir data/dialect_instructions \
        --model gpt-4o-mini \
        --max_per_dataset 500

    # If you have manually downloaded datasets, specify their paths:
    python generate_dialect_instructions.py \
        --output_dir data/dialect_instructions \
        --madar_path data/MADAR/           # dir of MADAR.corpus.{City}.tsv files \
        --saudial_path data/SauDial.csv    # single CSV with EN/MSA/dialect columns \
        --masc_path data/MASC/             # dir of masc_*.tsv files \
        --joda_path data/JODA/

Requirements:
    pip install litellm datasets requests tqdm
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import time
import asyncio
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

import litellm
from litellm import acompletion


# ---------------------------------------------------------------------------
# AL-QASIDA-style instruction templates (English versions — LLM translates)
# These mirror the 8 templates from Table 7 of the AL-QASIDA paper.
# ---------------------------------------------------------------------------

# Maps dataset dialect labels to standardized country codes used in AL-QASIDA
DIALECT_TO_COUNTRY = {
    # MADAR city names → country codes
    "ALG": "dza", "ALX": "egy", "AMM": "jor", "ASW": "egy", "BAG": "irq",
    "BAS": "irq", "BEI": "lbn", "BEN": "lby", "CAI": "egy", "DAM": "syr",
    "DOH": "qat", "FES": "mar", "JED": "sau", "JER": "pse", "KHA": "sdn",
    "MOS": "irq", "MSA": "msa", "MUS": "omn", "RAB": "mar", "RIY": "sau",
    "SAL": "jor", "SAN": "yem", "SFX": "tun", "TRI": "lby", "TUN": "tun",
    "Algiers": "dza", "Cairo": "egy", "Damascus": "syr", "Fes": "mar",
    "Aleppo": "syr", "Alexandria": "egy", "Amman": "jor", "Aswan": "egy",
    "Baghdad": "irq", "Basra": "irq", "Beirut": "lbn", "Benghazi": "lby",
    "Doha": "qat", "Jeddah": "sau", "Mosul": "irq", "Muscat": "omn",
    "Rabat": "mar", "Salt": "jor", "Sanaa": "yem", "Sfax": "tun",
    "Tripoli": "lby", "Tunis": "tun",
    "Jerusalem": "pse", "Khartoum": "sdn", "Riyadh": "sau",
    # Country names (from PALM)
    "Egypt": "egy", "egypt": "egy",
    "Morocco": "mar", "morocco": "mar",
    "Algeria": "dza", "algeria": "dza",
    "Syria": "syr", "syria": "syr",
    "Palestine": "pse", "palestine": "pse",
    "Jordan": "jor", "jordan": "jor",
    "Sudan": "sdn", "sudan": "sdn",
    "Iraq": "irq", "iraq": "irq",
    "Lebanon": "lbn", "lebanon": "lbn",
    "Tunisia": "tun", "tunisia": "tun",
    "Libya": "lby", "libya": "lby",
    "Yemen": "yem", "yemen": "yem",
    "Saudi Arabia": "sau", "saudi arabia": "sau",
    "Qatar": "qat", "qatar": "qat",
    "Oman": "omn", "oman": "omn",
    "Kuwait": "kwt", "kuwait": "kwt",
    "Bahrain": "bhr", "bahrain": "bhr",
    "UAE": "are", "uae": "are",
    "Mauritania": "mrt", "mauritania": "mrt",
    "Somalia": "som", "somalia": "som",
    "Djibouti": "dji", "djibouti": "dji",
    "Comoros": "com", "comoros": "com",
    # Generic dialect labels
    "egyptian": "egy", "egy": "egy", "EGY": "egy",
    "levantine": "syr", "lev": "syr", "LEV": "syr",
    "gulf": "sau", "glf": "sau", "GLF": "sau",
    "maghrebi": "mar", "mgr": "mar", "MGR": "mar",
    "iraqi": "irq", "irq": "irq", "IRQ": "irq",
    "saudi": "sau", "sau": "sau", "SAU": "sau",
    "moroccan": "mar", "mar": "mar", "MAR": "mar",
    "algerian": "dza", "dza": "dza", "DZA": "dza",
    "syrian": "syr", "syr": "syr", "SYR": "syr",
    "palestinian": "pse", "pse": "pse", "PSE": "pse",
    "jordanian": "jor", "jor": "jor", "JOR": "jor",
    "sudanese": "sdn", "sdn": "sdn", "SDN": "sdn",
    "tunisian": "tun", "tun": "tun", "TUN": "tun",
    "libyan": "lby", "lby": "lby", "LBY": "lby",
    "lebanese": "lbn", "lbn": "lbn", "LBN": "lbn",
    "yemeni": "yem", "yem": "yem", "YEM": "yem",
    "darija": "mar",
}

COUNTRY_TO_DIALECT_NAME = {
    "egy": "Egyptian Arabic",
    "sau": "Saudi Arabic",
    "mar": "Moroccan Arabic (Darija)",
    "dza": "Algerian Arabic",
    "syr": "Syrian Arabic",
    "pse": "Palestinian Arabic",
    "jor": "Jordanian Arabic",
    "sdn": "Sudanese Arabic",
    "irq": "Iraqi Arabic",
    "lbn": "Lebanese Arabic",
    "tun": "Tunisian Arabic",
    "lby": "Libyan Arabic",
    "yem": "Yemeni Arabic",
    "qat": "Qatari Arabic",
    "omn": "Omani Arabic",
    "kwt": "Kuwaiti Arabic",
}


# ---------------------------------------------------------------------------
# Dataset loaders — each returns list of dicts:
#   {"text": str, "dialect": str, "source": str}
#   and optionally {"english": str} for bitext
# ---------------------------------------------------------------------------

def load_habibi(cache_dir: str, habibi_path: str | None = None) -> list[dict]:
    """HABIBI song lyrics corpus. Tries GitHub download first, falls back to local path."""
    samples = []
    csv_text = None

    if habibi_path and os.path.exists(habibi_path):
        print(f"[HABIBI] Loading from local path: {habibi_path}")
        with open(habibi_path, encoding="utf-8", errors="ignore") as f:
            csv_text = f.read()
    else:
        # Try multiple URLs (branch may be main or master, file may be in different dirs)
        urls = [
            "https://media.githubusercontent.com/media/ArabicNLP-UK/Habibi/refs/heads/main/habibi1.csv",
            "https://raw.githubusercontent.com/ArabicNLP-UK/Habibi/main/Habibi_CSV/habibi.csv",
            "https://raw.githubusercontent.com/ArabicNLP-UK/Habibi/master/Habibi_CSV/habibi.csv",
            "https://raw.githubusercontent.com/ArabicNLP-UK/Habibi/main/habibi.csv",
            "https://raw.githubusercontent.com/ArabicNLP-UK/Habibi/master/habibi.csv",
        ]
        for url in urls:
            try:
                print(f"[HABIBI] Trying {url}...")
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                csv_text = resp.text
                print(f"[HABIBI] Downloaded from {url}")
                break
            except Exception:
                continue

        if csv_text is None:
            print("[HABIBI] Could not download. Provide --habibi_path to a local CSV.")
            print("  Download from: https://github.com/ArabicNLP-UK/Habibi")
            return []

    reader = csv.DictReader(io.StringIO(csv_text))
    # Detect column names (may vary: verse/Verse/lyrics/text/etc.)
    fields = reader.fieldnames or []
    text_key = next((f for f in fields if f.lower() in ("verse", "lyrics", "text", "sentence")), None)
    country_key = next((f for f in fields if f.lower() in ("country", "nationality", "dialect", "songdialect")), None)

    if not text_key:
        print(f"[HABIBI] Cannot find text column in {fields}")
        return []

    for row in reader:
        text = (row.get(text_key) or "").strip()
        country = (row.get(country_key) or "").strip().lower() if country_key else ""
        if text and len(text) > 10:
            dialect = DIALECT_TO_COUNTRY.get(country, country)
            samples.append({
                "text": text,
                "dialect": dialect,
                "source": "habibi",
                "type": "monolingual",
            })
    print(f"[HABIBI] Loaded {len(samples)} verses")
    return samples


def load_edc(cache_dir: str) -> list[dict]:
    """Egyptian Dialect Corpus from GitHub."""
    print("[EDC] Downloading...")
    url = "https://raw.githubusercontent.com/TaghreedT/EDC/main/EDC.txt"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    samples = []
    for line in resp.text.strip().split("\n"):
        text = line.strip()
        if text and len(text) > 10:
            samples.append({
                "text": text,
                "dialect": "egy",
                "source": "edc",
                "type": "monolingual",
            })
    print(f"[EDC] Loaded {len(samples)} sentences")
    return samples


def load_atlaset(cache_dir: str) -> list[dict]:
    """Atlaset Moroccan Arabic from HuggingFace."""
    print("[Atlaset] Loading from HuggingFace...")
    from datasets import load_dataset
    try:
        ds = load_dataset("atlasia/Atlaset", split="train")
    except Exception as e:
        print(f"[Atlaset] Failed to load: {e}")
        return []

    samples = []
    for row in ds:
        # Try common field names
        text = row.get("text") or row.get("sentence") or row.get("content") or ""
        if isinstance(text, str) and len(text.strip()) > 10:
            sample = {
                "text": text.strip(),
                "dialect": "mar",
                "source": "atlaset",
                "type": "monolingual",
            }
            # Check for English/translation field
            en = row.get("english") or row.get("translation") or row.get("en") or ""
            if en and isinstance(en, str) and len(en.strip()) > 5:
                sample["english"] = en.strip()
                sample["type"] = "bitext"
            samples.append(sample)
    print(f"[Atlaset] Loaded {len(samples)} samples")
    return samples



def load_flores(cache_dir: str) -> list[dict]:
    """FLORES-200 Arabic dialect splits from HuggingFace."""
    print("[FLORES] Loading from HuggingFace...")
    from datasets import load_dataset

    flores_langs = {
        "arz_Arab": "egy",  # Egyptian Arabic
        "acm_Arab": "irq",  # Mesopotamian Arabic (Iraqi)
        "apc_Arab": "syr",  # North Levantine Arabic
        "ars_Arab": "sau",  # Najdi Arabic (Saudi)
        "ary_Arab": "mar",  # Moroccan Arabic
    }

    # Try multiple dataset names (API has changed over time)
    dataset_names = [
        "openlanguagedata/flores_plus",
        "facebook/flores",
        "Muennighoff/flores200",
    ]

    samples = []
    for ds_name in dataset_names:
        if samples:
            break
        print(f"[FLORES] Trying {ds_name}...")

        # Load English once
        try:
            ds_en = load_dataset(ds_name, "eng_Latn", split="devtest")
        except Exception:
            try:
                ds_en = load_dataset(ds_name, "eng", split="devtest")
            except Exception:
                try:
                    # Some versions use a single config with language column
                    ds_all = load_dataset(ds_name, split="devtest")
                    ds_en = ds_all.filter(lambda x: x.get("language") == "eng_Latn" or x.get("lang") == "eng")
                except Exception as e:
                    print(f"[FLORES]   {ds_name} failed: {e}")
                    continue

        en_sentences = [row.get("sentence", row.get("text", "")) for row in ds_en]

        for flores_code, dialect in flores_langs.items():
            try:
                ds = load_dataset(ds_name, flores_code, split="devtest")
                for i, ar_row in enumerate(ds):
                    text = ar_row.get("sentence", ar_row.get("text", ""))
                    en_text = en_sentences[i] if i < len(en_sentences) else ""
                    if text and len(text.strip()) > 10:
                        samples.append({
                            "text": text.strip(),
                            "english": en_text.strip(),
                            "dialect": dialect,
                            "source": "flores",
                            "type": "bitext",
                        })
            except Exception as e:
                print(f"[FLORES]   {flores_code} failed: {e}")

    print(f"[FLORES] Loaded {len(samples)} bitext pairs")
    return samples


def load_madar(madar_path: str | None) -> list[dict]:
    """MADAR-26 parallel corpus (requires manual download from CAMeL Lab).
    Directory of per-city TSV files named like MADAR.corpus.Cairo.tsv.
    Each file has one sentence per line in that city's dialect.
    English is NOT included in the public release (copyright)."""
    if not madar_path or not os.path.exists(madar_path):
        print("[MADAR] Path not provided or not found. Skipping.")
        print("  Download from: https://camel.abudhabi.nyu.edu/madar-parallel-corpus/")
        return []

    print(f"[MADAR] Loading from {madar_path}...")
    samples = []
    path = Path(madar_path)
    tsv_files = list(path.rglob("*.tsv")) if path.is_dir() else [path]

    # Build parallel sentences: group by line number across city files
    # so we can create bitext pairs between dialects
    city_sentences = {}  # city_name -> list of sentences (indexed by line)

    for tsv_file in tsv_files:
        # Extract city name from filename like MADAR.corpus.Cairo.tsv
        fname = tsv_file.stem  # e.g. "MADAR.corpus.Cairo"
        parts = fname.split(".")
        city_name = None
        for p in parts:
            if p in DIALECT_TO_COUNTRY:
                city_name = p
                break
        # Also try the last non-empty part
        if not city_name:
            for p in reversed(parts):
                if p and p not in ("MADAR", "corpus", "tsv", "train", "dev", "test"):
                    city_name = p
                    break

        if not city_name:
            print(f"  Skipping {tsv_file.name}: cannot determine city")
            continue

        dialect = DIALECT_TO_COUNTRY.get(city_name, "")
        if not dialect or dialect == "msa":
            # Still store MSA for bitext pairing
            pass

        sentences = []
        with open(tsv_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                # TSV may have multiple columns; take the last non-empty one as text
                parts = line.strip().split("\t")
                text = parts[-1].strip() if parts else ""
                sentences.append(text)

        city_sentences[city_name] = {
            "dialect": dialect if dialect and dialect != "msa" else "msa",
            "sentences": sentences,
        }

    # Find English and MSA sentences for bitext pairing
    en_sents = None
    msa_sents = None
    for city, data in city_sentences.items():
        if city.lower() in ("english", "english.index"):
            en_sents = data["sentences"]
        elif data["dialect"] == "msa" or city.lower() in ("msa", "msa.index"):
            msa_sents = data["sentences"]

    # Build samples from each city (skip MSA, English, French)
    for city_name, data in city_sentences.items():
        dialect = data["dialect"]
        if dialect == "msa" or city_name.lower() in ("english", "english.index", "french", "french.index"):
            continue

        for i, text in enumerate(data["sentences"]):
            if not text or len(text) < 5:
                continue

            sample = {
                "text": text,
                "dialect": dialect,
                "source": "madar",
                "city": city_name,
            }

            # Pair with English (preferred) or MSA for bitext
            if en_sents and i < len(en_sents) and en_sents[i]:
                sample["english"] = en_sents[i]
                sample["type"] = "bitext"
            elif msa_sents and i < len(msa_sents) and msa_sents[i]:
                sample["english"] = msa_sents[i]  # MSA as translation source
                sample["type"] = "bitext"
            else:
                sample["type"] = "monolingual"

            samples.append(sample)

    print(f"[MADAR] Loaded {len(samples)} samples from {len(city_sentences)} city files "
          f"({sum(1 for s in samples if s['type'] == 'bitext')} bitext, "
          f"{sum(1 for s in samples if s['type'] == 'monolingual')} monolingual)")
    return samples


def load_saudial(saudial_path: str | None) -> list[dict]:
    """SauDial Saudi Arabic Dialects Game Localization Dataset (Mendeley).
    CSV with parallel columns: English, MSA, and 4 Saudi dialects
    (Najdi, Hijazi, Janoubi, Eastern)."""
    if not saudial_path or not os.path.exists(saudial_path):
        print("[SauDial] Path not provided or not found. Skipping.")
        print("  Download from: https://data.mendeley.com/datasets/mzdwkb2t6d/2")
        return []

    print(f"[SauDial] Loading from {saudial_path}...")
    samples = []
    path = Path(saudial_path)

    csv_files = list(path.rglob("*.csv")) if path.is_dir() else [path]

    for csv_path in csv_files:
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            header = [h.strip().lower() for h in reader.fieldnames or []]

            # Columns: Dialect, ..., English Text, Modern Standard Arabic (MSA) Translation, 
            #          Dialect Translation, ...
            fields = reader.fieldnames or []
            en_key = next((h for h in fields if "english" in h.lower()), None)
            msa_key = next((h for h in fields if "msa" in h.lower() or "modern standard" in h.lower()), None)
            da_key = next((h for h in fields if "dialect" in h.lower() and "translation" in h.lower()), None)
            subdialect_key = next((h for h in fields if h.strip().lower() == "dialect"), None)

            if not da_key:
                print(f"  {csv_path.name}: no 'Dialect Translation' column in {fields}")
                continue

            for row in reader:
                da_text = (row.get(da_key) or "").strip()
                if not da_text or len(da_text) < 5:
                    continue

                en_text = (row.get(en_key) or "").strip() if en_key else ""
                msa_text = (row.get(msa_key) or "").strip() if msa_key else ""
                sub_dialect = (row.get(subdialect_key) or "").strip().lower() if subdialect_key else ""

                sample = {
                    "text": da_text,
                    "dialect": "sau",
                    "source": "saudial",
                    "sub_dialect": sub_dialect,
                }

                if en_text and len(en_text) > 3:
                    sample["english"] = en_text
                    sample["type"] = "bitext"
                elif msa_text and len(msa_text) > 3:
                    sample["english"] = msa_text
                    sample["type"] = "bitext"
                else:
                    sample["type"] = "monolingual"

                samples.append(sample)

    print(f"[SauDial] Loaded {len(samples)} samples "
          f"({sum(1 for s in samples if s['type'] == 'bitext')} bitext, "
          f"{sum(1 for s in samples if s['type'] == 'monolingual')} monolingual)")
    return samples


def load_masc(masc_path: str | None) -> list[dict]:
    """MASC multi-dialect Arabic speech corpus transcripts (manual download).
    Directory of split-based TSV files like masc_software.tsv.
    Expected TSV format: columns include dialect label and text."""
    if not masc_path or not os.path.exists(masc_path):
        print("[MASC] Path not provided or not found. Skipping.")
        print("  Download from: https://github.com/almoslmi/masc")
        return []

    print(f"[MASC] Loading from {masc_path}...")
    samples = []
    path = Path(masc_path)
    tsv_files = list(path.rglob("*.tsv")) + list(path.rglob("*.csv"))
    if path.suffix.lower() in (".tsv", ".csv"):
        tsv_files = [path]

    for tsv_file in tsv_files:
        with open(tsv_file, encoding="utf-8", errors="ignore") as f:
            # Try to detect if there's a header
            first_line = f.readline().strip()
            f.seek(0)

            # Check if first line looks like a header
            first_parts = first_line.split("\t")
            has_header = any(
                h.lower() in ("dialect", "text", "label", "sentence", "country", "id")
                for h in first_parts
            )

            if has_header:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    # Find text column (case-insensitive)
                    text = ""
                    for k in ("Text", "text", "Sentence", "sentence", "Lyrics", "lyrics"):
                        if k in row and row[k]:
                            text = row[k].strip()
                            break
                    # Find dialect/country column (case-insensitive)
                    dialect_label = ""
                    for k in ("Country", "country", "Dialect", "dialect", "Label", "label"):
                        if k in row and row[k]:
                            dialect_label = row[k].strip()
                            break

                    if text and len(text) > 10:
                        dialect = (DIALECT_TO_COUNTRY.get(dialect_label)
                                   or DIALECT_TO_COUNTRY.get(dialect_label.lower())
                                   or "unknown")
                        samples.append({
                            "text": text,
                            "dialect": dialect,
                            "source": "masc",
                            "type": "monolingual",
                        })
            else:
                # No header — assume TSV with dialect_label<TAB>text or just text
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        dialect_label = parts[0].strip().lower()
                        text = parts[-1].strip()
                    elif len(parts) == 1:
                        text = parts[0].strip()
                        dialect_label = ""
                    else:
                        continue

                    if text and len(text) > 10:
                        dialect = DIALECT_TO_COUNTRY.get(dialect_label, "unknown")
                        samples.append({
                            "text": text,
                            "dialect": dialect,
                            "source": "masc",
                            "type": "monolingual",
                        })

    print(f"[MASC] Loaded {len(samples)} transcripts from {len(tsv_files)} files")
    return samples


def load_joda(joda_path: str | None) -> list[dict]:
    """JODA Jordanian dialect corpus (Mendeley).
    CSV with columns: Source, Text, Type, Corrected Text, Diacritized Text."""
    if not joda_path or not os.path.exists(joda_path):
        print("[JODA] Path not provided or not found. Skipping.")
        print("  Download from: https://data.mendeley.com/datasets/ffrskd27f4/1")
        return []

    print(f"[JODA] Loading from {joda_path}...")
    samples = []
    path = Path(joda_path)
    csv_files = list(path.rglob("*.csv")) if path.is_dir() else [path]

    for csv_path in csv_files:
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []

            # Prefer "Corrected Text" over raw "Text"
            text_key = next((k for k in fields if "corrected" in k.lower()), None)
            if not text_key:
                text_key = next((k for k in fields if k.lower() == "text"), None)
            if not text_key:
                text_key = next((k for k in fields if "text" in k.lower()), None)

            if not text_key:
                print(f"  {csv_path.name}: no text column found in {fields}")
                continue

            for row in reader:
                text = (row.get(text_key) or "").strip()
                if text and len(text) > 10:
                    samples.append({
                        "text": text,
                        "dialect": "jor",
                        "source": "joda",
                        "type": "monolingual",
                    })

    print(f"[JODA] Loaded {len(samples)} sentences")
    return samples


def load_palm_dialect(cache_dir: str) -> list[dict]:
    """PALM dataset — dialect-only samples."""
    print("[PALM] Loading dialect samples from HuggingFace...")
    from datasets import load_dataset
    try:
        ds = load_dataset("UBC-NLP/palm", split="train")
    except Exception as e:
        print(f"[PALM] Failed to load: {e}")
        return []

    samples = []
    for row in ds:
        variety = row.get("language_variety", "MSA")
        if variety == "MSA":
            continue
        text = row.get("output", "").strip()
        instruction = row.get("instruction", "").strip()
        country = row.get("country", "").strip()
        # language_variety is just "DA", so map via country name
        dialect = DIALECT_TO_COUNTRY.get(country, DIALECT_TO_COUNTRY.get(country.lower(), "unknown"))
        if text and len(text) > 10:
            samples.append({
                "text": text,
                "dialect": dialect,
                "source": "palm",
                "type": "monolingual",
                "original_instruction": instruction,
            })
    print(f"[PALM] Loaded {len(samples)} dialect samples")
    return samples


# ---------------------------------------------------------------------------
# Static bitext instruction templates (no LLM needed)
# ---------------------------------------------------------------------------

BITEXT_PREFIXES_EN_TO_DA = [
    "ترجم للهجتك: ",
    "قول هاد الحكي بلهجتك: ",
    "كيف بتقول هاد بلهجتك: ",
    "حول للعامية: ",
    "اكتب هاد بالعامية: ",
]

BITEXT_PREFIXES_MSA_TO_DA = [
    "حول الجملة دي للهجتك: ",
    "قولها بالعامية: ",
    "اعد كتابتها باللهجة: ",
]


def _is_arabic(text: str) -> bool:
    """Check if text is primarily Arabic script."""
    arabic = sum(1 for ch in text if 0x0600 <= ord(ch) <= 0x06FF or 0x0750 <= ord(ch) <= 0x077F)
    alpha = sum(1 for ch in text if ch.isalpha())
    return arabic / max(alpha, 1) > 0.5


def create_bitext_instructions(samples: list[dict]) -> list[dict]:
    """Create translation instruction pairs directly from bitext — no LLM needed.
    Detects whether the source is English or MSA and picks appropriate prefixes."""
    results = []
    for s in samples:
        if s["type"] != "bitext" or "english" not in s:
            continue

        source_text = s["english"]
        # Detect if the "english" field is actually MSA (Arabic text)
        if _is_arabic(source_text):
            prefix = random.choice(BITEXT_PREFIXES_MSA_TO_DA)
        else:
            prefix = random.choice(BITEXT_PREFIXES_EN_TO_DA)

        instruction = prefix + source_text

        results.append({
            "prompt": [{"role": "user", "content": instruction}],
            "ground_truth": s["text"],  # original dialectal text
            "dialect": s["dialect"],
            "source": s["source"],
            "type": "bitext",
        })
    return results


# ---------------------------------------------------------------------------
# LLM-based instruction generation (monolingual only)
# ---------------------------------------------------------------------------

# AL-QASIDA template styles (English descriptions for the LLM to follow)
# Target dialects for evaluation — prioritize these in data generation
TARGET_DIALECTS = {"egy", "mar", "pse", "sau", "syr", "jor"}

# Dialect-specific question words to guide the LLM
DIALECT_QUESTION_PATTERNS = {
    "egy": ["إيه", "ايه", "يعني إيه", "ممكن أعرف", "إيه رأيك", "ايه معنى"],
    "mar": ["شنو", "كيفاش", "شنو كتعني", "واش", "فين", "علاش"],
    "pse": ["شو", "ايش", "كيف", "احكيلي", "شو يعني", "ايش معنى"],
    "sau": ["وش", "ايش", "وشو", "كيف", "ايش هي", "ايش معنى"],
    "syr": ["شو", "كيف", "شو هي", "فيك تقلي", "احكي", "بدي تعطيني"],
    "jor": ["شو", "كيف", "ايش", "احكيلي", "شو يعني", "وين"],
}

MONOLINGUAL_STYLES = [
    "an open-ended question about {topic} that this text answers — use question words like {qwords}",
    "a 'what is...' or 'what does... mean' question about {topic} that this text answers — use question words like {qwords}",
    "a 'tell me about...' or 'explain...' request about {topic} that elicits this text — use dialectal phrasing with words like {qwords}",
    "a 'how do you...' question about {topic} that this text answers — use question words like {qwords}",
    "a culturally curious question about {topic} that a native speaker might ask — use words like {qwords}",
    "a question asking for the meaning of something mentioned in the text — use question words like {qwords}",
]


# AL-QASIDA question topics
QUESTION_TOPICS = [
    "food, recipes, or traditional dishes from the region",
    "local proverbs, sayings, or their meanings",
    "cultural traditions, celebrations, or customs",
    "geography, cities, or landmarks",
    "history, historical figures, or events",
    "local dialect words or expressions and their meanings",
    "nature, animals, or agriculture in the region",
    "arts, music, or literature",
]
 
async def _generate_one(sample: dict, model: str, sem: asyncio.Semaphore) -> dict | None:
    """Generate a single dialectal instruction for one sample."""
    from litellm import acompletion
 
    api_key = os.environ.get("LITELLM_API_KEY")
    base_url = os.environ.get("LITELLM_API_BASE_URL")
 
    dialect_name = COUNTRY_TO_DIALECT_NAME.get(sample["dialect"], f"Arabic dialect ({sample['dialect']})")
    topic = random.choice(QUESTION_TOPICS)
    qwords = ", ".join(DIALECT_QUESTION_PATTERNS.get(sample["dialect"], ["\u0634\u0648", "\u0627\u064a\u0634", "\u0643\u064a\u0641"]))
 
    # Step 1: Generate a question
    question_prompt = (
        f"You are a native speaker of {dialect_name}.\n"
        f"Below is a text in {dialect_name}. Write a short question (1 sentence) in {dialect_name} "
        f"that someone might ask a chatbot, where this text would be a natural answer.\n\n"
        f"Rules:\n"
        f"- Write ONLY the question, nothing else.\n"
        f"- The question must be in {dialect_name}, NOT Modern Standard Arabic.\n"
        f"- Use dialectal question words like: {qwords}\n"
        f"- The question should be about {topic}.\n"
        f"- Do not copy or rephrase the text below.\n\n"
        f"Text:\n{sample['text']}"
    )
 
    async with sem:
        for attempt in range(3):
            try:
                # Generate question
                response = await acompletion(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    messages=[{"role": "user", "content": question_prompt}],
                    # temperature=0.8,
                    # max_tokens=256,
                    # timeout=6000,
                )
                instruction = response.choices[0].message.content.strip()
 
                if not instruction or len(instruction) < 5:
                    return None
                if instruction.startswith('"') and instruction.endswith('"'):
                    instruction = instruction[1:-1]
                if instruction.startswith("'") and instruction.endswith("'"):
                    instruction = instruction[1:-1]
 
                # Step 2: Generate a full answer incorporating the corpus text
                answer_prompt = (
                    f"You are a helpful chatbot that ONLY speaks {dialect_name}. "
                    f"A user asked you: {instruction}\n\n"
                    f"Write a helpful, detailed answer (3-6 sentences) in {dialect_name}. "
                    f"You MUST incorporate this phrase naturally in your answer: {sample['text']}\n\n"
                    f"Rules:\n"
                    f"- Write ONLY the answer, nothing else.\n"
                    f"- The answer must be entirely in {dialect_name}, NOT Modern Standard Arabic.\n"
                    f"- Be informative and conversational, like a native speaker chatting.\n"
                    f"- Do NOT start with 'Sure' or 'Of course' in any language."
                )
 
                answer_response = await acompletion(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    messages=[{"role": "user", "content": answer_prompt}],
                    # temperature=0.8,
                    # max_tokens=512,
                    # timeout=6000,
                )
                full_answer = answer_response.choices[0].message.content.strip()
 
                if not full_answer or len(full_answer) < 20:
                    # Fall back to corpus text if expansion fails
                    full_answer = sample["text"]
 
                return {
                    "prompt": [{"role": "user", "content": instruction}],
                    "ground_truth": full_answer,
                    "dialect": sample["dialect"],
                    "source": sample["source"],
                    "type": sample.get("type", "monolingual"),
                }
 
            except (Exception, asyncio.CancelledError) as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                return None
    return None 
 
 
def generate_instruction_batch(
    samples: list[dict],
    model: str = "gpt-4o-mini",
    batch_size: int = 10,
    output_path: str | None = None,
) -> list[dict]:
    """Generate dialectal instructions using litellm with async concurrency.
    One LLM call per sample, plain text output. Saves progress incrementally."""
 
    async def run_all():
        sem = asyncio.Semaphore(batch_size)
        pbar = tqdm(total=len(samples), desc="Generating instructions")
        all_results = []
        failed = 0
 
        async def wrapped(s):
            nonlocal failed
            try:
                result = await _generate_one(s, model, sem)
                pbar.update(1)
                return result
            except (Exception, asyncio.CancelledError) as e:
                failed += 1
                pbar.update(1)
                print("Failed:", e)
                return None
 
        # Process in chunks to save progress and avoid connection pool exhaustion
        chunk_size = batch_size * 10
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            chunk_results = await asyncio.gather(*[wrapped(s) for s in chunk])
            all_results.extend([r for r in chunk_results if r is not None])
 
            # Save progress every chunk
            print(output_path, len(all_results))
            if output_path and all_results:
                print("Saving to", output_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    for item in all_results:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
 
        pbar.close()
        if failed:
            print(f"  {failed} samples failed")
        return all_results
 
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(run_all())
    except RuntimeError:
        return asyncio.run(run_all())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate dialectal Arabic instruction data")
    parser.add_argument("--output_dir", type=str, default="data/dialect_instructions")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model name (passed to litellm)")
    parser.add_argument("--max_per_dataset", type=int, default=500,
                        help="Max samples per dataset (before LLM generation)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Samples per LLM API call")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--cache_dir", type=str, default="data/cache")

    # Manual download paths
    parser.add_argument("--habibi_path", type=str, default=None,
                        help="Local path to habibi.csv if download fails")
    parser.add_argument("--madar_path", type=str, default=None)
    parser.add_argument("--saudial_path", type=str, default=None)
    parser.add_argument("--masc_path", type=str, default=None)
    parser.add_argument("--joda_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ---- Load all datasets ----
    all_samples = []

    loaders = [
        ("HABIBI", lambda: load_habibi(args.cache_dir, args.habibi_path)),
        ("EDC", lambda: load_edc(args.cache_dir)),
        ("Atlaset", lambda: load_atlaset(args.cache_dir)),
        ("FLORES", lambda: load_flores(args.cache_dir)),
        ("PALM-DA", lambda: load_palm_dialect(args.cache_dir)),
        ("MADAR", lambda: load_madar(args.madar_path)),
        ("SauDial", lambda: load_saudial(args.saudial_path)),
        ("MASC", lambda: load_masc(args.masc_path)),
        ("JODA", lambda: load_joda(args.joda_path)),
    ]

    for name, loader_fn in loaders:
        try:
            samples = loader_fn()
            # Filter out unknowns and MSA
            samples = [s for s in samples if s["dialect"] not in ("", "unknown", "msa")]
            # Prioritize target dialects within each dataset
            if len(samples) > args.max_per_dataset:
                target = [s for s in samples if s["dialect"] in TARGET_DIALECTS]
                other = [s for s in samples if s["dialect"] not in TARGET_DIALECTS]
                # Keep all target up to max, fill rest with other
                if len(target) >= args.max_per_dataset:
                    samples = random.sample(target, args.max_per_dataset)
                else:
                    remaining = args.max_per_dataset - len(target)
                    samples = target + random.sample(other, min(remaining, len(other)))
            all_samples.extend(samples)
            print(f"  → Using {len(samples)} from {name}")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")

    print(f"\nTotal raw samples: {len(all_samples)}")

    # Prioritize target dialects (egy, mar, pse, sau, syr) to match evaluation
    target_samples = [s for s in all_samples if s["dialect"] in TARGET_DIALECTS]
    other_samples = [s for s in all_samples if s["dialect"] not in TARGET_DIALECTS]
    print(f"\nTarget dialect samples (egy/mar/pse/sau/syr): {len(target_samples)}")
    print(f"Other dialect samples: {len(other_samples)}")

    # Keep all target dialect samples, subsample others to max 20% of total
    max_other = max(int(len(target_samples) * 0.2), 100)
    if len(other_samples) > max_other:
        other_samples = random.sample(other_samples, max_other)
        print(f"Subsampled other dialects to {max_other}")
    all_samples = target_samples + other_samples
    print(f"Final sample count: {len(all_samples)}")

    # Print dialect distribution
    from collections import Counter
    dialect_counts = Counter(s["dialect"] for s in all_samples)
    print("\nDialect distribution:")
    for dialect, count in dialect_counts.most_common():
        name = COUNTRY_TO_DIALECT_NAME.get(dialect, dialect)
        print(f"  {dialect} ({name}): {count}")

    source_counts = Counter(s["source"] for s in all_samples)
    print("\nSource distribution:")
    for source, count in source_counts.most_common():
        print(f"  {source}: {count}")

    # ---- Split bitext (static) from all samples (LLM for monolingual questions) ----
    bitext_samples = [s for s in all_samples if s["type"] == "bitext" and "english" in s]

    print(f"\nBitext samples (free translation instructions): {len(bitext_samples)}")
    print(f"All samples (LLM monolingual question generation): {len(all_samples)}")

    # Create bitext translation instructions directly (no LLM cost)
    bitext_instructions = create_bitext_instructions(bitext_samples)
    print(f"Created {len(bitext_instructions)} translation instruction pairs")

    # Generate monolingual Q&A instructions via LLM for ALL samples
    # (including bitext — their dialect text becomes the answer to a novel question)

    print(f"\nGenerating monolingual Q&A instructions using {args.model}...")
    random.shuffle(all_samples)
    progress_path = os.path.join(args.output_dir, "progress.jsonl")
    mono_instructions = generate_instruction_batch(
        all_samples,
        model=args.model,
        batch_size=args.batch_size,
        output_path=progress_path,
    )
    print(f"Generated {len(mono_instructions)} monolingual Q&A instruction pairs")

    # Combine: monolingual Q&A (primary, matches eval) + translation (supplementary)
    instructions = mono_instructions + bitext_instructions
    print(f"\nTotal instruction pairs: {len(instructions)} "
          f"({len(mono_instructions)} Q&A + {len(bitext_instructions)} translation)")

    # ---- Save ----
    random.shuffle(instructions)
    if args.val_ratio > 0:
        split_idx = int(len(instructions) * (1 - args.val_ratio))
        train = instructions[:split_idx]
        val = instructions[split_idx:]
    else:
        train = instructions
        val = []

    train_path = os.path.join(args.output_dir, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(train)} train samples → {train_path}")

    if val:
        val_path = os.path.join(args.output_dir, "val.jsonl")
        with open(val_path, "w", encoding="utf-8") as f:
            for item in val:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(val)} val samples → {val_path}")

    # Also save as HF-compatible format (for TRL)
    train_trl_path = os.path.join(args.output_dir, "train_trl.jsonl")
    with open(train_trl_path, "w", encoding="utf-8") as f:
        for item in train:
            trl_item = {
                "prompt": item["prompt"],
                "ground_truth": item["ground_truth"],
            }
            f.write(json.dumps(trl_item, ensure_ascii=False) + "\n")
    print(f"Saved TRL-formatted train → {train_trl_path}")

    print("\nDone!")
    print(f"To use with train_grpo.py, update the data loading to read from {train_trl_path}")


if __name__ == "__main__":
    main()
