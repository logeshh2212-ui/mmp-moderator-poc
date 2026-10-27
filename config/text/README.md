# Text word lists

Used by `app/gates/text.py` for text that OCR reads on images and video frames, and for
speech in video audio.

| File | Source | Licence |
|---|---|---|
| `ldnoobw_en.txt`, `ldnoobw_hi.txt` | [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) at commit `5faf2ba` | CC-BY-4.0 (attribution: Shutterstock / LDNOOBW contributors) |
| `extra_hi_devanagari.txt` | ours, to be filled by trust-and-safety | — |
| `allow_pet_words.txt` | ours: list words that are normal on a pet platform | — |

Matching is per whole word (or whole phrase), lower case. Changing any file changes the
settings fingerprint, so `model_version` and the cache key change too.
