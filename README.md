# Tiny Keyword Spotting under Noise and Resource Constraints

A reproducible PyTorch pipeline for **Keyword Spotting (KWS)** that compares two compact models for resource-limited devices and investigates the trade-off between recognition accuracy, noise robustness, model size and inference speed.

The pipeline covers raw audio loading, a microphone-clipping audit, GPU log-Mel feature extraction, waveform noise augmentation, multi-seed training, evaluation under seen and unseen background noise, CPU latency measurement, a SpecAugment ablation and inference demos (dataset clips, noisy clips, live microphone), on **Google Speech Commands v2 (10 classes)**.

---

## Key results (official test set, mean ± std over seeds 2026–2028)

| Model | Params | FP32 size | MACs | Largest activation (FP32) | End-to-end CPU latency* | Test clean | Test seen noise | Test unseen noise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DS-CNN (*Hello Edge*, 2017) | 23,050 | 90.04 KB | 21.67 M | 255 KB | 1.71 ms | 97.20 ± 0.15 % | 91.88 ± 0.30 % | 90.96 ± 0.20 % |
| BC-ResNet-1 (Interspeech 2021) | 9,166 | 35.80 KB | 2.48 M | 126 KB | 4.24 ms | 97.30 ± 0.20 % | 91.98 ± 0.11 % | 90.11 ± 1.14 % |

\* Laptop CPU, batch size = 1, one thread, log-Mel front-end included; absolute times depend on the machine.

* **Accuracy:** tied on clean speech and on seen noise types.
* **Noise robustness:** on unseen noise DS-CNN is more robust (better in 2 of 3 seeds).
* **Size and compute:** BC-ResNet-1 has 2.5× fewer parameters, 8.7× fewer MACs and half the activation memory.
* **Speed:** on a laptop CPU BC-ResNet-1 is about 2.5× slower despite fewer MACs. No microcontroller was measured.
* **SpecAugment ablation:** SpecAugment does not help either model (clean and seen noise unchanged within seed variation; about −1 point on unseen noise).

---

## Repository structure

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
└── my_noise.wav                  # a random noise from the internet (we are using the hand clapping noise from this link: 
                                  [https://pixabay.com/sound-effects/people-clapping-slight-echo-102675/])
```

---

##  Installation instructions

### 1. Prerequisites
* **Operating system:** Linux, macOS, or Windows 10/11
* **Python:** tested with Python 3.13.9 and PyTorch 2.11.0
* **Hardware:** CUDA-compatible NVIDIA GPU or CPU and a microphone for the live demo
* **Audio (for Linux):** sudo apt install libportaudio2

### 2. Clone repository & Setup virtual environment
```bash
# Clone the repository
git clone https://github.com/nguyentk2410727-lab/Final_Project_DL
cd Final_Project_DL

# Create and activate a virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify GPU acceleration:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Dataset preparation

The project targets the 10 core keywords: `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`.

### Automated download & Extraction:
On Linux
```bash
mkdir -p archive && cd archive
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz
cd ..
```
On Windows:
```powershell
New-Item -ItemType Directory -Force archive; Set-Location archive
curl.exe -L -o speech_commands_v0.02.tar.gz https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz
Set-Location ..
```


*For detailed documentation regarding speaker-hash partitioning, clipping audit (dropping 350 corrupted files), and seen vs. unseen noise splitting, please consult **[DATA.md](DATA.md)**.*

---

##  Steps for reproducing experimental results

All experiments, data verification, and model evaluations are packaged inside the modular notebook **`simple_kws.ipynb`**.

### Run the notebook: Interactive execution via Jupyter notebook
Launch the Jupyter interface and execute all cells sequentially:
```bash
jupyter notebook simple_kws.ipynb
```

### Execution flow
Cell numbers refer to `simple_kws.ipynb` (26 code cells, run in order):

1. **Cell 1:** imports, constants (16 kHz, 1-s clips, seed 2026) and PCM16 readers.
2. **Cell 2:** official validation / test lists → 30,769 / 3,703 / 4,074 clips; speaker-disjoint splits asserted.
3. **Cells 3–4:** clipping audit (`near_full_scale_ratio > 0.005`) → 350 dropped, 30,419 clean training clips; per-keyword summary.
4. **Cell 5:** noise split. Seen (`doing_the_dishes`, `exercise_bike`, `pink_noise`, `white_noise`): 80 / 10 / 10 % for train / validation / seen test. Unseen (`running_tap`, `dude_miaowing`): whole recordings, final test only.
5. **Cell 6:** EDA (class counts per split; duration, RMS and peak histograms).
6. **Cells 7–8:** 40-band log-Mel front-end (30 ms window, 10 ms hop, per-clip normalisation); SpecAugment layer, used only in the ablation.
7. **Cells 9–10:** center zero-padding, ±100 ms time shift, noise mixing at SNR ∈ [−5, 20] dB with p = 0.8, deterministic noisy validation / test sets, DataLoaders (batch 100).
8. **Cells 11–12:** sanity checks (non-silent clips, SNR mixing within 0.001 dB, deterministic noisy validation); clean-vs-noisy figure.
9. **Cell 13:** BC-ResNet-1 and DS-CNN; parameters, FP32 size, MACs.
10. **Cells 14–15:** training. SGD (momentum 0.9, weight decay 1e-3), batch 100, 100 epochs, linear warm-up to lr 0.1 over 5 epochs then cosine decay to 0, seeds 2026–2028; best checkpoint by (clean + noisy validation) / 2, saved to `checkpoints/`.
11. **Cells 16–18:** final test evaluation (clean + 30 noisy conditions); mean ± std over seeds, paired differences, per-SNR and per-noise tables; validation curves.
12. **Cell 19:** CPU latency (batch size = 1, 1 thread), MACs, largest activation.
13. **Cell 20:** trade-off summary table.
14. **Cells 21–22:** SpecAugment ablation (same recipe and seeds, SpecAugment on).
15. **Cells 23–26:** demos: inference on test clips (23), confusion matrix (24), noisy demo (25), live microphone demo (26).

**Runtime:** if `checkpoints/` contains the `.pt` and `_history.csv` files, cells 15 and 21 load them instead of training, and the notebook runs in a few minutes. Training from scratch takes much more time (6 main + 6 SpecAugment ablation runs).

---

## Inference & Demo
The last cells of `simple_kws.ipynb` contain four demos (run them after the cells above):

1. **Inference demo:** BC-ResNet-1 (seed 2026) on six clean test clips.
2. **Confusion matrix:** BC-ResNet-1 on the clean test set.
3. **Noisy demo:** one test clip per keyword, five mixed with seen and five with unseen test noise.
4. **Live microphone demo:** 3-second countdown, 1-second recording kept as an in-memory `.wav`, mixed with `my_noise.wav` at 20 / 10 / 5 / 0 / −5 dB; prints both models' predictions for every condition and shows playback, waveform, log-Mel spectrogram and probabilities. Needs a microphone; set `NOISE_FILE = None` to use the clean voice only.

The inference demo cell, as it appears in the notebook:

```python
#23: Inference demo: BC-ResNet-1 (seed 2026) on six clean test clips
demo_front_end = AudioFrontEnd().to(device).eval()
demo_model = BCResNet1(num_classes=len(LABELS)).to(device)
demo_model.load_state_dict(torch.load(CKPT_DIR / f"BC-ResNet-1_seed{SEED}_e{EPOCHS}.pt", map_location=device))
demo_model.eval()


def predict_keyword(wav_path):
    wav = torch.from_numpy(load_audio(wav_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(demo_model(demo_front_end(wav)), dim=1)[0].cpu().numpy()
    k = int(np.argmax(probs))
    return LABELS[k], float(probs[k])

for path, label in zip(test_files[::800], test_labels[::800]):
    word, conf = predict_keyword(path)
    print(f"{Path(path).name:28s} true: {LABELS[label]:5s} | predicted: {word:5s} ({100 * conf:.1f}%)")
```

---

## Academic references

If you build upon or reference this project, please cite the following foundational publications:

1. Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.arXiv:1804.03209.
2. Zhang, Y., Suda, N., Lai, L., & Chandra, V. (2017). Hello Edge: Keyword Spotting on Microcontrollers.arXiv:1711.07128.
3. Kim, B., Chang, S., Lee, J., & Sung, D. (2021). Broadcasted Residual Learning for Efficient Keyword Spotting. In INTERSPEECH 2021. arXiv:2106.04140.
4. Chang, S., Park, H., Cho, J., Park, H., Yun, S., & Hwang, K. (2021). Subspectral Normalization for Neural Audio Data Processing. 
5. Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.arXiv:1704.04861.
6. Park, D. S., et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. In INTERSPEECH 2019. arXiv:1904.08779.
7. Chen, G., Parada, C., & Heigold, G. (2014). Small-footprint Keyword Spotting Using Deep Neural Networks.
8. Sainath, T. N., & Parada, C. (2015). Convolutional Neural Networks for Small-footprint Keyword Spotting. 
9. Tang, R., & Lin, J. (2018). Deep Residual Learning for Small-footprint Keyword Spotting. 
10. Banbury, C., et al. (2021). MLPerf Tiny Benchmark.