# Ingest Triage Specification

This document defines how each document page is classified as `text-layer`, `image-scan`, or `corrupt-font` before downstream processing.

## Metrics
- **text_min_len:** 80 characters. Pages with fewer characters are considered to lack a usable text layer.
- **badchars_max_ratio:** 0.02 (2%). Ratio of unreadable characters such as `�`.
- **corrupt_font_ratio:** 0.15 (15%). Proportion of glyphs rendered with missing or broken fonts.
- **page_sample_for_manual:** 20 pages per batch sampled for manual auditing.

## Classification Rules
1. **text-layer**
   - Condition: detected text length ≥ `text_min_len` **and**
     `badchars_ratio` ≤ `badchars_max_ratio` **and** `corrupt_font_ratio` < 0.15.
   - Action: page is trusted; send directly to normalization without OCR.
   - Symptoms: selectable text, consistent fonts, minimal garbage characters.
2. **image-scan**
   - Condition: no text layer or text length < `text_min_len`.
   - Action: route to OCR with profile A.
   - Symptoms: purely raster image, zero or very short text layer.
3. **corrupt-font**
   - Condition: text length ≥ `text_min_len` but `corrupt_font_ratio` ≥ 0.15
     or `badchars_ratio` > `badchars_max_ratio`.
   - Action: treat as image-scan, trigger manual flag and log for review.
   - Symptoms: "�" replacement chars, gibberish encoding, invisible glyphs.

## Manual Sampling
- Randomly choose `page_sample_for_manual` pages per document for human review.
- If misclassification rate exceeds 5%, escalate for a full document audit.
- Record sample decisions in `logs/`.
