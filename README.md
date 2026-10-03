# Radio advert intake pipeline

Processes every recording from the radio stations, separates the voice from music and
noise, transcribes it, pulls out company names, stores everything in a database and two
CSV files, and shows live results on a dashboard.

## How each recording is handled

1. Register the file (station and time come from the file name, or the folder and file date).
2. Exact repeat? The file hash matches an earlier file, so reuse its result.
3. Decode to 16 kHz mono. Silent or shorter than 0.5 s means blank.
4. Speech detection (Silero VAD). Less than `min_speech_seconds` of speech means blank.
5. Compare with everything already processed (audio fingerprint). A repeat airing of a known ad reuses the earlier transcript and company names, so no heavy work runs.
6. New audio only: isolate the voice from music and noise (Demucs), then re-check for speech. Music only means blank.
7. Transcribe (faster-whisper, language detected per recording). No words means blank.
8. Find company names (known list and aliases, fuzzy match for misspellings, discovery of new names).
9. Save to the database, append to `transcripts.csv`, refresh `company_names.csv`.

Only new, unique audio costs GPU time. Ads repeat all day across stations, so most of the 3,000 daily files should be cheap matches.

## Outputs (in `work_dir`, default `./data`)

- `transcripts.csv`: one row per recording: station, time, verbal or blank (and why), language, transcript, companies, and which earlier recording it repeated.
- `company_names.csv`: one row per company: recordings mentioning it, stations, first and last seen, and `verified` (yes or needs review). New names found automatically start as "needs review". To confirm one, add it to `known_companies.txt` (`Name|alias|alias`) and run `init`.
- `radio.db`: the SQLite database the dashboard reads.

## Setup

```bash
sudo apt install ffmpeg
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm        # optional, English name discovery
cp config.yaml my.yaml                         # set paths.audio_root, workers, model
python -m radiopipe init --config my.yaml
```

## Use

```bash
python -m radiopipe run --config my.yaml --limit 20     # try a small sample first
python -m radiopipe run --config my.yaml                # work through the 850,000 backlog (resumable)
python -m radiopipe watch --config my.yaml              # then keep running: new files are processed as they land
python -m radiopipe dashboard --config my.yaml --target 850000   # http://localhost:8050
python -m radiopipe export --config my.yaml             # rebuild the CSVs from the database
python -m unittest discover -s tests                    # self-test with synthetic audio
```

Run `watch` and `dashboard` as two services (systemd, Docker or similar). `run` can be stopped and restarted at any time and continues where it left off.

## Dashboard

Total files handled, with verbal content, blank (and why: silent, no speech, music only), transcriptions made versus transcripts reused, errors, files waiting and time remaining, daily volume, hourly throughput, stations, languages, most-mentioned companies, names waiting for review, audio hours, and a searchable list of recent recordings with transcripts. It refreshes every 15 seconds and needs no internet connection.

## Capacity (assumptions, please check against your real audio)

I assumed an average ad length near 30 s. 850,000 files would then be about 7,000 hours of audio. With one modern GPU, faster-whisper large-v3 runs very roughly 20 to 50 times faster than real time, so transcribing everything without reuse would take weeks of GPU time. Fingerprint reuse cuts that sharply, because only unique ads are transcribed. Measure the unique share on a 5,000-file sample before sizing hardware.

## Known limits

- **Luganda and other local languages.** Whisper is much weaker than on English. Set `asr.language` per station if you know it, and plan to evaluate a fine-tuned model (for example MMS or a Makerere-trained one) by adding a backend in `asr.py`.
- **Same ad, different tag line.** An ad that differs only by a short dealer or location tag can be treated as a repeat and inherit the earlier transcript. Tighten `duration_tolerance_s` or raise `min_ratio` if this matters.
- **Repeats inside one batch.** Several copies of a brand-new ad processed at the same moment may each be transcribed once. This only wastes a little compute the first time an ad appears.
- **Company names are suggestions.** Auto-discovered names need a quick human check, which is what the `needs review` flag is for. Claude-assisted extraction (`companies.use_llm`) sends transcripts to the Anthropic API, so only enable it if that is acceptable.
- **Tested here with synthetic audio.** The full flow, duplicate matching, CSVs and dashboard were tested with generated recordings and a stand-in transcriber. The Silero, faster-whisper and Demucs code paths follow those libraries' documented APIs but could not be run in the build environment (no model downloads), so run `--limit 20` on real recordings first and check the output.
