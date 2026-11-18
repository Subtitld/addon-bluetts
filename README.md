# BlueTTS add-on for Subtitld

Offline **text-to-speech with voice cloning** for
[Subtitld](https://github.com/Subtitld/subtitld), powered by
[BlueTTS](https://github.com/maxmelichov/BlueTTS) (the BlueV3 model, v3.0.1)
by Maxim Melichov, run on [ONNX Runtime](https://onnxruntime.ai).

- Serves the `tts.synthesize` task over Subtitld's add-on protocol.
- **Seven languages, Hebrew first.** Hebrew, English, German, Spanish,
  Italian, Russian and Yiddish, at 44.1 kHz. Numbers, dates, prices and codes
  are read as words. English words in a Hebrew line are read in English.
- **15 stock voices, or the speaker's own.** Subtitld sends a clip of the
  speaker from the video, and the dub speaks in their voice. Every voice
  speaks every language, each at its own natural pace.
- **CPU only, and fast.** On an i7-13700T, about 5 times faster than real
  time (3 times for Hebrew, whose G2P is a large model), with about 700 MB of
  memory (about 2 GB for Hebrew). No GPU, no torch. Loading Hebrew's model
  takes about 15 seconds the first time.
- **Offline after the first line.** Each file downloads the first time a line
  needs it, then nothing leaves the machine: no telemetry, no account.

## Voices

| Voice | |
|-------|---|
| Speaker's own voice | Clones the clip of the speaker Subtitld sends with each line, or the fallback clip in Configure |
| Noa, Maya, Lily, Tamar, Shira, Yael, Emma, Sara | Women |
| Adam, Daniel, David, Jonathan, Ethan, Michael, Ariel | Men |

The stock voices are the release's own voice files, made from LibriTTS-R
speakers (CC-BY-4.0). They download with the model.

A good clip to clone is 8 to 10 seconds of one person speaking: clean, close
to the microphone, and not a phone recording or a re-encoded copy. BlueTTS
reads only the first 9 seconds or so, and copies the voice's timbre and pitch,
not its delivery: clone from the delivery you want. Clips under a second are
refused.

## Languages

| Language | Text front end | Downloads when first needed |
|----------|----------------|-----------------------------|
| English, German, Spanish, Italian, Russian | polyg2p | with the model |
| Hebrew | RenikudPlus | 1,231 MB (or 312 MB compact, see below) |
| Yiddish | Phonikud-yi with ReNikud-yi | 307 MB |

Spanish follows the project's region: Spain reads Castilian, any other region
(`es-MX`, `es-419`...) Latin American Spanish. Yiddish numbers must be spelled
out; a line with digits is refused with a message that says so.

## Options

Set these in **Settings ▸ Add-ons ▸ BlueTTS ▸ Configure**.

| Option | Default | Env var | Notes |
|--------|---------|---------|-------|
| CPU threads | automatic | `BLUETTS_THREADS` | 0 uses the logical processors, at most 8. |
| Quality steps | 8 | `BLUETTS_STEPS` | Flow steps per line. 8 is BlueTTS's own setting; 4 is about 1.7 times as fast with slightly lower quality. |
| Hebrew reading model | Accurate | `BLUETTS_HEBREW_G2P` | `accurate`: the release's RenikudPlus weights (1.2 GB, about 2 GB of memory). `compact`: the same model quantized to int8 (312 MB). On 94 Hebrew Wikipedia sentences the compact model read 8 of 1,590 words differently, mostly worse (stress, or "FIFA" as "pifa"). |
| Fallback reference clip | none | `BLUETTS_VOICE_REF_AUDIO` | Cloned when Subtitld has no clip of the speaker to send. |

Each line is spoken with the seed Subtitld sends, so the same line in the same
voice comes out the same every time. The request's `rate` (−100 to 100 %)
sets BlueTTS's speed, 0.5 to 2 times each voice's natural pace; Subtitld
sends 0 for add-on voices today.

## How it runs

The add-on runs upstream's own inference code, copied into
`bluetts_addon/vendor/` by `tools/vendor_upstream.py` at pinned revisions:

- `blue_onnx`, BlueTTS's inference package (MIT), at BlueTTS commit
  `cbe5150`. One patch (`tools/patches/`) imports librosa and soundfile only
  where the add-on never goes, takes the clone's samples from the add-on
  instead of a file, and reads the Yiddish front end from the add-on.
- The Yiddish front end of the BlueV3 release bundle, which the bundle ships
  as Python files: the add-on carries them, so no code is downloaded at run
  time; only the Yiddish model is.

So the text normalization, the G2P per language, the chunking, the flow steps
and the peak limit are upstream's. The add-on adds the protocol, the
downloads, a seed, and a cancel between flow steps. `tests/test_engine.py`
checks the add-on's output against upstream's `BlueTTS` class on the same
voice and seed (identical to the last bit of a 16-bit WAV on x86-64 Linux).

polyg2p, the G2P BlueV3 was trained on, is not on PyPI. `tools/get_polyg2p.py`
installs the release bundle's wheel on Linux and macOS, and builds the same
commit from source on Windows, where the bundle has none.
`tests/test_polyg2p.py` checks each platform's polyg2p against the bundle's
Linux wheel on 600 Tatoeba sentences; a build from source on Linux matched on
12,000.

## What is bundled, what is downloaded

| What | Size | Licence | Source |
|------|------|---------|--------|
| Add-on, ONNX Runtime, numpy, polyg2p, RenikudPlus's runtime, num2words (bundled) | see the release | Apache-2.0; MIT; BSD-3-Clause; LGPL-2.1 (num2words, as plain source files) | this repository, PyPI, the BlueV3 bundle |
| `blue_onnx` and the Yiddish front end's code (bundled) | 2 MB | MIT; two Yiddish lexicon tables CC BY-SA 4.0 | BlueTTS, the BlueV3 bundle |
| BlueV3 model, 15 voices, polyg2p data (downloaded) | 428 MB | MIT; vocoder **BigScience OpenRAIL-M**; voices CC-BY-4.0; German and Italian lexicons CC BY-SA | [notmax123/BlueV3-onnx](https://huggingface.co/notmax123/BlueV3-onnx) |
| Reference encoder, for cloning (downloaded) | 136 MB | MIT | the same |
| RenikudPlus, Hebrew (downloaded) | 1,231 MB | Apache-2.0 | the same |
| RenikudPlus int8, Hebrew compact (downloaded) | 312 MB | Apache-2.0 | [notmax123/RenikudPlus](https://huggingface.co/notmax123/RenikudPlus) |
| ReNikud-yi, Yiddish (downloaded) | 307 MB | MIT | the BlueV3 bundle |

The files go to Subtitld's model folder (`$SUBTITLD_MODELS_DIR/bluetts`), at
the revisions in `bluetts_addon/pins.json`, each checked by its sha256. An
interrupted download resumes. `$HF_ENDPOINT` points it at a mirror. The
bundle's `LICENSES.md` and `LICENSE.OpenRAIL-M` download with the model.

**The vocoder's licence restricts use.** BlueV3's audio decoder is fine-tuned
from Supertone's Supertonic-3 vocoder and carries the BigScience OpenRAIL-M
licence, whose use-based restrictions (no unlawful use, no deceiving people,
no impersonation without consent, among others) apply to the speech the
add-on makes. Clone a voice only with its owner's consent.

Each release archive carries `THIRD_PARTY_LICENSES/README.md`, which names
the notice for every native file in it; the build fails if one has none. The
Rust crates in polyg2p's extension are listed with their licences in
`THIRD_PARTY_LICENSES/polyg2p/CRATES.txt`. The Linux archive carries
libquadmath (LGPL-2.1), from numpy's wheel; its source is attached to the
same release.

## Limits

- **Seven languages.** A project in any other language is refused with a
  message that says so.
- **Memory.** Hebrew loads a 1.2 GB model: about 2 GB of memory with it, and
  2.5 GB once Yiddish is loaded too. The compact Hebrew model loads faster.
- **No style control.** Delivery comes from the voice, or the clip, and a
  little from punctuation. Very long sentences can sound uneven.
- **Platforms.** Releases ship Linux x86_64 (glibc 2.35 or newer), macOS arm64
  and Windows x86_64.
- **One line at a time.** A cancel stops the line between flow steps, or
  stops the download, which resumes on the next line.

## Build

```bash
python tools/get_polyg2p.py
pip install -c constraints.txt -e '.[build,dev]'
pytest
pyinstaller pyinstaller.spec --noconfirm
dist/bluetts-addon/bluetts-addon --self-test
BLUETTS_TEST_BINARY=$PWD/dist/bluetts-addon/bluetts-addon pytest tests/test_installed.py
```

`pytest` downloads every model group (2.4 GB) into `$SUBTITLD_MODELS_DIR`.

To move to another BlueTTS commit or bundle revision, change the pins in
`tools/vendor_upstream.py` and run it (the patch must still apply), then
`tools/pin_models.py`, `tools/get_polyg2p.py` with the new wheels' sha256,
`tools/crate_licences.py` against polyg2p's new commit, and regenerate the
parity fixtures in `tests/fixtures/` with upstream's own code.

A tag `vYY.MM.N` builds, tests and publishes every platform
(`.github/workflows/release.yml`).
