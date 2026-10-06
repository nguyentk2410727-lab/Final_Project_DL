# DATA.md - Dataset documentation & Reproduction guide

**Project:** Tiny Keyword Spotting under Noise and Resource Constraints (Topic 35)  
**Dataset:** Google Speech Commands Dataset v2  
**Target vocabulary:** 10-class core keywords (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`)

---

## 1. Official dataset overview & Download URL

* **Dataset name:** Google Speech Commands Dataset
* **Version:** Version 2 (`v0.02`), released by Pete Warden - Google AI (2018).
* **Official URL (Download link):**  
  [http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
* **Direct archive download:** `wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
* **Dataset size:** ~2.4 GB (compressed `.tar.gz`), ~3.3 GB uncompressed.
* **License:** Creative Commons BY 4.0 (CC-BY 4.0).
* **Audio characteristics:** 
  * Uncompressed linear PCM WAV, single-channel (mono).
  * Sampling rate: 16,000 Hz (16 kHz).
  * Bit depth: 16-bit signed integer (`int16`).
  * Nominal clip length: 1.0 second (16,000 samples).

---

## 2. Directory structure

The project expects the extracted Google Speech Commands dataset inside an `archive/` directory in the repository root:

```text
.
├── archive/                      # Extracted Google Speech Commands v2 dataset
│   ├── yes/, no/, up/, ...       # 10 keyword folders
│   ├── _background_noise_/       # Ambient noise files
│   ├── validation_list.txt       # Official Google validation split
│   └── testing_list.txt          # Official Google test split
├── checkpoints/                  # Best-epoch weights (.pt) and per-epoch training logs (_history.csv): 2 models × 3 seeds, plus *_specaug ablation runs
├── simple_kws.ipynb              # Main executable Jupyter Notebook (end-to-end pipeline)
├── 21_35_Report.docx             # Complete final project report (Academic format)
├── DATA.md                       # Comprehensive dataset documentation & split protocol
├── README.md                     # Repository documentation and reproduction instructions
├── requirements.txt              # Environment dependencies
├── topic.png                     # Project topic and objectives sheet
└── my_noise.wav                  # a random noise from the internet (we are using the hand clapping noise from this link 
                                  [https://pixabay.com/sound-effects/people-clapping-slight-echo-102675/])
```

---

## 3. Data split & Anti-leakage protocol

To ensure 100% academic reproducibility and prevent **speaker identity leakage (Data leakage)**, the dataset is split strictly using the official Google partitioning files:

* **Partitioning mechanism:** Google assigned speakers to validation and test sets based on a SHA-1 hash of their unique speaker IDs. No speaker in the training set appears in validation or testing.
* **Mapping logic:**
  * If a file path matches an entry in `archive/validation_list.txt` $\rightarrow$ Assigned to **Validation set**.
  * If a file path matches an entry in `archive/testing_list.txt` $\rightarrow$ Assigned to **Test set**.
  * All remaining audio files for the 10 target classes $\rightarrow$ Assigned to **Training set**.

### Partition file statistics:
| Split Set | Initial File Count | Clipped Files Dropped | Clean Files Retained |
| :--- | :---: | :---: | :---: |
| **Train** | 30,769 | 350 (~1.1%) | **30,419** |
| **Validation** | 3,703 | 0 (evaluation intact) | **3,703** |
| **Test** | 4,074 | 0 (evaluation intact) | **4,074** |
| **Total** | **38,546** | **350** | **38,196** |

---

## 4. Preprocessing & Data cleaning procedure

### 4.1. Microphone saturation & Clipping audit
* **Issue:** Many recordings contain severe clipping distortion caused by user vocalisation exceeding microphone pre-amplifier dynamic ranges.
* **Detection criterion:**
  $$\text{Clipping Ratio} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(|x_i| \ge 32,440\right) > 0.005$$
  (where $32,440 \approx 0.99 \times 32,768$ full-scale amplitude).
* **Important implementation note:** Samples are cast to `np.int32` before computing `np.abs()` (`np.abs(audio.astype(np.int32)) >= 32440`). This avoids two's-complement overflow on the extreme minimum value $-32,768$, which cannot be represented as $+32,768$ within signed 16-bit integers.
* **Action:** 350 corrupted training recordings were dropped to eliminate high-frequency square-wave harmonic splatter, retaining exactly 30,419 clean training files.

### 4.2. Amplitude scaling & zero-padding
* **Dynamic range normalization:** Raw 16-bit integer samples are divided by $32,768.0$, mapping values to continuous floating-point amplitudes in $[-1.0, 1.0]$ (`np.float32`).
* **Center zero-padding:** clips shorter than 1.0 second (2,825 of the 30,419 clean training clips, ≈ 9.3%) are centered and zero-padded on both sides to exactly 16,000 samples.

### 4.3. GPU audio Front-End
Features are converted dynamically on the GPU in batches:
1. **STFT:** $N\_FFT = 512$, window length = 480 (30 ms), hop length = 160 (10 ms).
2. **Mel filterbank:** 40 Mel frequency channels (HTK scale).
3. **Decibel conversion:** `AmplitudeToDB(top_db=80.0)`.
4. **Instance normalization:** Per-sample normalization $\frac{S - \mu}{\sigma + 10^{-6}}$ across 2D frequency-time axes.
* **Output tensor dimension:** `(Batch, 1, 40, 101)`.

### 4.4. Training-time waveform augmentation (training set only)
1. **Random time shift:** uniform integer shift in [−100 ms, +100 ms] (±1,600 samples), zero-filled.
2. **Background noise:** with probability 0.8, a random 1-second window from `train_noise_bank` is mixed at an SNR drawn uniformly from [−5, +20] dB (Section 5).
3. **Peak limiting:** if the mixture exceeds full scale, speech and noise are divided by the same gain $\max(1, \max|x|)$, so the SNR is unchanged.

SpecAugment is not used in the main experiments; it is evaluated only in the ablation.

---

## 5. Background noise partitioning (Seen vs. Unseen)

Background noise from `_background_noise_` is strictly partitioned as defined in the experimental notebook:

1. **Seen noise bank (`TRAIN_NOISE_NAMES` — 4 sources):**
   * Tracks: `doing_the_dishes.wav`, `exercise_bike.wav`, `pink_noise.wav`, `white_noise.wav`.
   * Temporal slicing:
     * First 80% duration $\rightarrow$ Training noise bank (`train_noise_bank`).
     * Next 10% duration $\rightarrow$ Validation noise bank (`val_noise_bank`).
     * Final 10% duration $\rightarrow$ Appended to `test_noise_bank` as seen test noise (indices 0–3).
2. **Unseen noise bank (`UNSEEN_NOISE_NAMES` — 2 sources):**
   * Tracks: `running_tap.wav`, `dude_miaowing.wav`.
   * Complete recordings (100% duration) $\rightarrow$ Appended to `test_noise_bank` as unseen test noise (indices 4–5), reserved exclusively for final evaluation to assess zero-shot noise generalizability.
3. **Noisy evaluation sets (mixed and peak-limited as in Section 4.4):**
   * Validation (model selection): each clip is mixed with a window from one of the 4 `val_noise_bank` segments at an SNR from {20, 10, 5, 0, −5} dB; noise source, window and SNR are drawn per clip from `np.random.default_rng([2026, idx])`.
   * Test (final evaluation only): clean test set plus 6 noise sources × 5 SNRs {20, 10, 5, 0, −5} dB = 30 noisy conditions; within a condition every clip receives a window from that source's test segment, drawn from `np.random.default_rng([2026, idx])`.

During training, ambient noise is dynamically injected into clean audio with an 80% probability at a random Signal-to-Noise Ratio (SNR) sampled uniformly from $[-5\text{ dB}, +20\text{ dB}]$:
$$\text{scale} = \sqrt{\frac{P_{\text{speech}}}{P_{\text{noise}} \times 10^{\text{SNR}/10}}}$$
$$\text{audio}_{\text{mixed}} = \text{speech} + \text{scale} \times \text{noise}$$

---

## 6. Reproduction scripts

To reproduce the exact dataset setup used in the experiments:

### Step 1: Download & Extract dataset
```bash
# In your terminal / bash:
mkdir -p archive
cd archive
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz
cd ..
```

### Step 2: Run Dataset Verification, Cleaning & Noise Partitioning in Python
Run the following self-contained script (a condensed version of Cells 1, 2, 3 and 5 of simple_kws.ipynb, with the same split, clipping and noise-partition logic):

```python
from pathlib import Path
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE
DATA_DIR = Path("archive")
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
label_to_id = {word: i for i, word in enumerate(LABELS)}

# 1. PCM audio loading functions

def read_pcm16(path):
    sr, pcm = wavfile.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr}: {path}")
    if pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError(f"Expected mono PCM16, got shape={pcm.shape}, dtype={pcm.dtype}: {path}")
    if pcm.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return pcm

def read_audio(path):
    return read_pcm16(path).astype(np.float32) / 32768.0

# 2. Parse official splits 

with open(DATA_DIR / "validation_list.txt") as f:
    val_set = set(line.strip() for line in f if line.strip())

with open(DATA_DIR / "testing_list.txt") as f:
    test_set = set(line.strip() for line in f if line.strip())

raw_train_files, raw_train_labels = [], []
val_files,       val_labels       = [], []
test_files,      test_labels      = [], []

for word in LABELS:
    for p in sorted((DATA_DIR / word).glob("*.wav")):
        rel_path = f"{word}/{p.name}"
        lid = label_to_id[word]
        if rel_path in val_set:
            val_files.append(str(p))
            val_labels.append(lid)
        elif rel_path in test_set:
            test_files.append(str(p))
            test_labels.append(lid)
        else:
            raw_train_files.append(str(p))
            raw_train_labels.append(lid)

# 3. Audio clipping audit 

clean_train_files, clean_train_labels = [], []
clipped_count = 0

for path_str, label in zip(raw_train_files, raw_train_labels):
    audio = read_pcm16(path_str)
    # int32 avoids abs(-32768) overflow
    near_full_scale_ratio = float(np.mean(np.abs(audio.astype(np.int32)) >= 32440))
    if near_full_scale_ratio > 0.005:
        clipped_count += 1
    else:
        clean_train_files.append(path_str)
        clean_train_labels.append(label)

train_files = clean_train_files
train_labels = clean_train_labels

# 4. Seen / unseen background noise split 

TRAIN_NOISE_NAMES = ("doing_the_dishes", "exercise_bike", "pink_noise", "white_noise")
UNSEEN_NOISE_NAMES = ("running_tap", "dude_miaowing")
NOISE_DIR = DATA_DIR / "_background_noise_"

train_noise_bank, val_noise_bank, test_noise_bank = [], [], []

# Seen sources: disjoint 80% / 10% / 10% segments
for name in TRAIN_NOISE_NAMES:
    audio = read_audio(NOISE_DIR / f"{name}.wav")
    t_train, t_val = int(0.80 * len(audio)), int(0.90 * len(audio))
    train_noise_bank.append(audio[:t_train])
    val_noise_bank.append(audio[t_train:t_val])
    test_noise_bank.append(audio[t_val:])

# Unseen sources: complete recordings for final test only
for name in UNSEEN_NOISE_NAMES:
    audio = read_audio(NOISE_DIR / f"{name}.wav")
    test_noise_bank.append(audio)

print("=== REPRODUCTION VERIFICATION SUMMARY ===")
print(f"  Raw Train Files     : {len(raw_train_files):,} (Expected: 30,769)")
print(f"  Clipped Dropped     : {clipped_count:,} (Expected: 350)")
print(f"  Clean Train Files   : {len(train_files):,} (Expected: 30,419)")
print(f"  Validation Files    : {len(val_files):,} (Expected: 3,703)")
print(f"  Test Files          : {len(test_files):,} (Expected: 4,074)")
print(f"  Train Noise Bank    : {len(train_noise_bank)} tracks (Expected: 4 seen segments)")
print(f"  Val Noise Bank      : {len(val_noise_bank)} tracks (Expected: 4 seen segments)")
print(f"  Test Noise Bank     : {len(test_noise_bank)} tracks (Expected: 6 = 4 seen + 2 unseen)")

assert len(raw_train_files) == 30769
assert clipped_count == 350
assert len(train_files) == 30419
assert len(val_files) == 3703
assert len(test_files) == 4074
assert len(train_noise_bank) == 4
assert len(val_noise_bank) == 4
assert len(test_noise_bank) == 6
print("\nSUCCESS")
```

---

## 7. Mathematical & Repeatability sanity checks

All processed audio files undergo continuous assertions in `simple_kws.ipynb`:
1. **Waveform assertion:** Every clip is asserted non-silent ($P_{\text{audio}} > 10^{-12}$); one training batch is asserted to have shape (B, 16000) and dtype float32.
2. **SNR precision:** the measured SNR matches the target within 0.001 dB (asserted), before and after peak limiting, for 100 real mixes (5 validation clips × 4 validation noises × 5 SNRs) plus 2 synthetic edge cases.
3. **Evaluation determinism:** Both validation and test noisy datasets use reproducible indexing seeded by `SEED = 2026` (`rng = np.random.default_rng([self.seed, idx])`), ensuring bit-for-bit repeatability across independent experimental runs.