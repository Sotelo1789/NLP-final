"""
CSCI 182.06 — Natural Language Processing Final Project
Phase 1: Data Gathering & Cleaning
Author Dataset: Harry Potter Books (Book 1 - 7)

This script:
1. Reads the raw text files for Harry Potter Books 1-7
2. Cleans the text (lowercase, remove special chars, normalize whitespace)
3. Outputs:
   - raw_lyrics.txt          → all book text concatenated (raw, before cleaning)
   - cleaned_lyrics.txt      → cleaned version ready for tokenization
   - cleaning_report.txt     → summary stats before/after cleaning
"""

import csv
import re
import os

OUTPUT_DIR = "dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. LOAD RAW DATA
# ──────────────────────────────────────────────────────────────────────

import os

INPUT_FOLDER = "hp_books"

books = []

for filename in sorted(os.listdir(INPUT_FOLDER)):
    if filename.lower().endswith(".txt"):
        path = os.path.join(INPUT_FOLDER, filename)

        print("READING:", filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        books.append({
            "title": filename.replace(".txt", ""),
            "text": text
        })

print("Books collected:", len(books))

# ──────────────────────────────────────────────────────────────────────
# 2. SAVE RAW TEXT (Deliverable: Raw dataset)
# ──────────────────────────────────────────────────────────────────────

raw_text = ""
for book in books:
    raw_text += book["text"] + "\n\n"

raw_path = os.path.join(OUTPUT_DIR, "raw_lyrics.txt")
with open(raw_path, "w", encoding="utf-8") as f:
    f.write(raw_text)

print(f"Saved raw lyrics to {raw_path}")

# ──────────────────────────────────────────────────────────────────────
# 3. CLEANING PIPELINE
# ──────────────────────────────────────────────────────────────────────

def clean_text(text):
    """
    Cleaning steps:
    1. Lowercase everything
    2. Remove parentheses but keep the words inside them
       e.g. "(Yes)" → "yes", "(oh)" → "oh"
    3. Replace non-ASCII characters with ASCII equivalents
       e.g. é → e, à → a, œ → oe
    4. Remove all punctuation EXCEPT apostrophes (to preserve contractions
       like "don't" and "'round"
    5. Normalize whitespace (collapse multiple spaces/newlines into single space)
    6. Strip leading/trailing whitespace
    """

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove parentheses, keep content inside
    text = re.sub(r'[()]', '', text)

    # Step 3: Replace non-ASCII characters
    replacements = {
        'é': 'e',
        'è': 'e',
        'ê': 'e',
        'ë': 'e',
        'à': 'a',
        'â': 'a',
        'ä': 'a',
        'ù': 'u',
        'û': 'u',
        'ü': 'u',
        'ô': 'o',
        'ö': 'o',
        'î': 'i',
        'ï': 'i',
        'ç': 'c',
        'œ': 'oe',
        '\u2019': "'",   # right single quotation mark → apostrophe
        '\u2018': "'",   # left single quotation mark → apostrophe
        '\u201c': '',    # left double quotation mark → remove
        '\u201d': '',    # right double quotation mark → remove
        '\u2014': ' ',   # em dash → space
        '\u2013': ' ',   # en dash → space
        '\u2026': '',    # ellipsis → remove
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Step 4: Remove punctuation except apostrophes
    # Keep: letters, digits, apostrophes, spaces
    text = re.sub(r"[^a-z0-9' \n]", '', text)

    # Step 5: Normalize whitespace
    # Replace newlines with spaces (we want one continuous text stream for training)
    text = text.replace('\n', ' ')
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)

    # Step 6: Strip
    text = text.strip()

    return text


# Clean each book individually, then join
cleaned_books = []
for book in books:
    cleaned = clean_text(book["text"])
    cleaned_books.append(cleaned)

# Join all books with a single space (continuous text corpus)
cleaned_text = " ".join(cleaned_books)

cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_lyrics.txt")
with open(cleaned_path, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print(f"Saved cleaned lyrics to {cleaned_path}")

# ──────────────────────────────────────────────────────────────────────
# 4. CLEANING REPORT
# ──────────────────────────────────────────────────────────────────────

raw_words = raw_text.split()
cleaned_words = cleaned_text.split()
raw_vocab = set(raw_words)
cleaned_vocab = set(cleaned_words)

# Find non-ASCII that were replaced
raw_non_ascii = sorted(set(c for c in raw_text if ord(c) > 127))

report = f"""
{'='*60}
PHASE 1 — DATA CLEANING REPORT
Harry Potter Books 1-7
{'='*60}

DATASET OVERVIEW
  Books collected:          {len(books)}
  Author:               J.K Rowling

BEFORE CLEANING
  Total characters:         {len(raw_text):,}
  Total words:              {len(raw_words):,}
  Unique words (raw):       {len(raw_vocab):,}
  Non-ASCII characters:     {raw_non_ascii if raw_non_ascii else 'None'}

AFTER CLEANING
  Total characters:         {len(cleaned_text):,}
  Total words:              {len(cleaned_words):,}
  Unique words (cleaned):   {len(cleaned_vocab):,}
  Vocabulary reduction:     {len(raw_vocab) - len(cleaned_vocab):,} words removed
                            (due to lowercasing & punctuation removal)

CLEANING STEPS APPLIED
  1. Lowercased all text
  2. Removed parentheses, kept words inside
  3. Replaced non-ASCII chars with ASCII equivalents
  4. Removed punctuation (kept apostrophes for contractions)
  5. Normalized whitespace to single spaces
  6. Stripped leading/trailing whitespace

SAMPLE (first 500 chars of cleaned text):
{cleaned_text[:500]}...

{'='*60}
"""

report_path = os.path.join(OUTPUT_DIR, "cleaning_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"Report saved to {report_path}")
print(f"\nAll Phase 1 outputs saved to '{OUTPUT_DIR}/' folder:")
print(f"  - raw_lyrics.txt        (raw dataset deliverable)")
print(f"  - cleaned_lyrics.txt    (cleaned, ready for Phase 2)")
print(f"  - cleaning_report.txt   (summary stats)")
