📢：**Good news! 21,800 hours of multi-label Cantonese speech data and 10,000 hours of multi-label Chuan-Yu speech data are also available at [⭐WenetSpeech-Yue⭐](https://github.com/ASLP-lab/WenetSpeech-Yue) and [⭐WenetSpeech-Chuan⭐](https://github.com/ASLP-lab/WenetSpeech-Chuan).**


# WenetSpeech-Wu: Datasets, Benchmarks, and Models for a Unified Chinese Wu Dialect Speech Processing Ecosystem

<p align="center">
  Chengyou Wang<sup>1</sup>*, 
  Mingchen Shao<sup>1</sup>*, 
  Jingbin Hu<sup>1</sup>*, 
  Zeyu Zhu<sup>1</sup>*, 
  Hongfei Xue<sup>1</sup>, 
  Bingshen Mu<sup>1</sup>, 
  Xin Xu<sup>2</sup>, 
  Xingyi Duan<sup>6</sup>, 
  Binbin Zhang<sup>3</sup>, 
  Pengcheng Zhu<sup>3</sup>, 
  Chuang Ding<sup>4</sup>, 
  Xiaojun Zhang<sup>5</sup>, 
  Hui Bu<sup>2</sup>, 
  Lei Xie<sup>1</sup><sup>†</sup>
</p>

<p align="center">
  <sup>1</sup> Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University <br>
  <sup>2</sup> Beijing AISHELL Technology Co., Ltd. <br>
  <sup>3</sup> WeNet Open Source Community <br>
  <sup>4</sup> Moonstep AI <br>
  <sup>5</sup> Xi'an Jiaotong-Liverpool University <br>
  <sup>6</sup> YK Pao School
</p>

<p align="center">
📑 <a href="https://arxiv.org/abs/2601.11027">Paper</a> &nbsp&nbsp | &nbsp&nbsp 
🐙 <a href="https://github.com/ASLP-lab/WenetSpeech-Wu-Repo">GitHub</a> &nbsp&nbsp | &nbsp&nbsp 
🤗 <a href="https://huggingface.co/collections/ASLP-lab/wenetspeech-wu">HuggingFace</a>
<br>
<!-- 🖥️ <a href="">HuggingFace Space</a> &nbsp&nbsp | &nbsp&nbsp  -->
🎤 <a href="https://hujingbin1.github.io/WenetSpeechWu-Demo-Page-Public/">Demo Page</a> &nbsp&nbsp | &nbsp&nbsp 
💬 <a href="https://github.com/ASLP-lab/WenetSpeech-Wu-Repo?tab=readme-ov-file#contact">Contact Us</a>
</p>


This repository contains the official WenetSpeech-Wu dataset, the WenetSpeech-Wu benchmark, and related models.
<br>
<div align="center"><img width="800px" src="figs/overview2.png" /></div>
<br>


## Demo Video 
https://youtu.be/h293y859QSw



## Download
* The WenetSpeech-Wu dataset are available at [WenetSpeech-Wu](https://huggingface.co/datasets/ASLP-lab/WenetSpeech-Wu).
* The WenetSpeech-Wu benchmark are available at [WenetSpeech-Wu-Bench](https://huggingface.co/datasets/ASLP-lab/WenetSpeech-Wu-Bench).
* The ASR and understanding models are available at [WSWu-Understanding](https://huggingface.co/ASLP-lab/WenetSpeech-Wu-Speech-Understanding).
* The TTS and instruct TTS models are available at [WSWu-Generation](https://huggingface.co/ASLP-lab/WenetSpeech-Wu-Speech-Generation).



## Dataset
<br>
<div align="center"><img width="500px" src="figs/overall_v3.drawio.png" /></div>
<br>
WenetSpeech-Wu is the first large-scale Wu dialect speech corpus with multi-dimensional annotations. It contains rich metadata and annotations, including transcriptions with confidence scores, Wu-to-Mandarin translations, domain and sub-dialect labels, speaker attributes, emotion annotations, and audio quality measures. The dataset comprises approximately 8,000 hours of speech collected from diverse domains and covers eight Wu sub-dialects. To support a wide range of speech processing tasks with heterogeneous quality requirements, we further adopt a task-specific data quality grading strategy.

<br><br>
<p align="center">
<img src="figs/Statistical_overview_of_WenetSpeech-Wu.png" 
     alt="Statistical overview of WenetSpeech-Wu"
     width=80%>
</p>
<br>

## WenetSpeech-Wu-Bench

We introduce WenetSpeech-Wu-Bench, the first publicly available, manually curated benchmark for Wu dialect speech processing, covering ASR, Wu-to-Mandarin AST, speaker attributes, emotion recognition, TTS, and instruct TTS, and providing a unified platform for fair evaluation.

- **ASR:** Wu dialect ASR (9.75 hour, including Shanghainese, Suzhounese, and Mandarin code-mixed speech). Evaluated by CER.
- **Wu→Mandarin AST:** Speech translation from Wu dialects to Mandarin (3k utterances, 4.4h). Evaluated by BLEU.
- **Speaker Attributes & Emotion:** Speaker gender/age prediction and emotion recognition on Wu dialect. Evaluated by classification accuracy.
- **TTS:** Wu dialect TTS with speaker prompting (242 sentences, 12 speakers). Evaluated by speaker similarity, CER, and MOS.
- **Instruct TTS:** Instruction-following TTS with prosodic and emotional control. Evaluated by automatic accuracy and subjective MOS.


## Data Construction Pipeline for WenetSpeech-Wu

We propose an automatic and scalable pipeline for constructing a large-scale Wu dialect speech dataset with multi-dimensional annotations, as illustrated in the figure below. The pipeline is designed to enable efficient data collection, robust automatic transcription, and diverse downstream annotations.

<br>
<p align="center">
<img src="figs/pipeline.png" 
     alt="Data construction pipeline for WenetSpeech-Wu"
     width=70%>
</p>
<br>

## ASR & Understanding Leaderboard
Bold and underlined values denote the best and second-best results.

ASR results (CER%) on various test sets
<br>
<p align="center">
<table align="center“ border="0" cellspacing="0" cellpadding="6" style="border-collapse:collapse; margin:auto;">
  <tr>
    <th align="left" rowspan="2">Model</th>
    <th align="center" colspan="2">In-House</th>
    <th align="center">WS-Wu-Bench</th>
  </tr>
  <tr>
    <th align="center">Dialogue</th>
    <th align="center">Reading</th>
    <th align="center">ASR</th>
  </tr>

  <tr><td align="left" colspan="4"><b>ASR Models</b></td></tr>
  <tr>
    <td align="left" style="background-color:#eee;">Paraformer</td>
    <td align="center" style="background-color:#eee;">63.13</td>
    <td align="center" style="background-color:#eee;">66.85</td>
    <td align="center" style="background-color:#eee;">64.92</td>
  </tr>
  <tr>
    <td align="left" style="background-color:#eee;">SenseVoice-small</td>
    <td align="center" style="background-color:#eee;">29.20</td>
    <td align="center" style="background-color:#eee;">31.00</td>
    <td align="center" style="background-color:#eee;">46.85</td>
  </tr>
  <tr>
    <td align="left" style="background-color:#eee;">Whisper-medium</td>
    <td align="center" style="background-color:#eee;">79.31</td>
    <td align="center" style="background-color:#eee;">83.94</td>
    <td align="center" style="background-color:#eee;">78.24</td>
  </tr>
  <tr>
    <td align="left" style="background-color:#eee;">FireRedASR-AED-L</td>
    <td align="center" style="background-color:#eee;">51.34</td>
    <td align="center" style="background-color:#eee;">59.92</td>
    <td align="center" style="background-color:#eee;">56.69</td>
  </tr>
  <tr>
    <td align="left" style="background-color:#eee;">Step-Audio2-mini</td>
    <td align="center" style="background-color:#eee;">24.27</td>
    <td align="center" style="background-color:#eee;">24.01</td>
    <td align="center" style="background-color:#eee;">26.72</td>
  </tr>

  <tr>
    <td align="left" style="background-color:#fdd;">Qwen3-ASR</td>
    <td align="center" style="background-color:#fdd;">23.96</td>
    <td align="center" style="background-color:#fdd;">24.13</td>
    <td align="center" style="background-color:#fdd;">29.31</td>
  </tr>
  <tr>
    <td align="left" style="background-color:#fdd;">Tencent-Cloud-ASR</td>
    <td align="center" style="background-color:#fdd;">23.25</td>
    <td align="center" style="background-color:#fdd;">25.26</td>
    <td align="center" style="background-color:#fdd;">29.48</td>
  </tr>
  <tr>
    <td align="left" style="background-color:#fdd;">Gemini-2.5-pro</td>
    <td align="center" style="background-color:#fdd;">85.50</td>
    <td align="center" style="background-color:#fdd;">84.67</td>
    <td align="center" style="background-color:#fdd;">89.99</td>
  </tr>

  <tr>
    <td align="left" style="background-color:#dfd;">Conformer-U2pp-Wu ⭐</td>
    <td align="center" style="background-color:#dfd;">15.20</td>
    <td align="center" style="background-color:#dfd;">12.24</td>
    <td align="center" style="background-color:#dfd;">15.14</td>
  </tr>
  <tr>
    <td align="left" style="background-color:#dfd;">Whisper-medium-Wu ⭐</td>
    <td align="center" style="background-color:#dfd;">14.19</td>
    <td align="center" style="background-color:#dfd;">11.09</td>
    <td align="center" style="background-color:#dfd;"><ins>14.33</ins></td>
  </tr>
  <tr>
    <td align="left" style="background-color:#dfd;">Step-Audio2-Wu-ASR ⭐</td>
    <td align="center" style="background-color:#dfd;"><ins>8.68</ins></td>
    <td align="center" style="background-color:#dfd;">7.86</td>
    <td align="center" style="background-color:#dfd;"><b>12.85</b></td>
  </tr>

  <tr><td align="left" colspan="4"><b>Annotation Models</b></td></tr>
  <tr>
    <td align="left" style="background-color:#eee;">Dolphin-small</td>
    <td align="center" style="background-color:#eee;">24.78</td>
    <td align="center" style="background-color:#eee;">27.29</td>
    <td align="center" style="background-color:#eee;">26.93</td>
  </tr>
  <tr>
    <td align="left" style="background-color:#fdd;">TeleASR</td>
    <td align="center" style="background-color:#fdd;">29.07</td>
    <td align="center" style="background-color:#fdd;">21.18</td>
    <td align="center" style="background-color:#fdd;">30.81</td>
  </tr>
  <tr>
    <td align="left" style="background-color:#b7e4b0;">Step-Audio2-FT</td>
    <td align="center" style="background-color:#b7e4b0;"><b>8.02</b></td>
    <td align="center" style="background-color:#b7e4b0;"><b>6.14</b></td>
    <td align="center" style="background-color:#b7e4b0;">15.64</td>
  </tr>
  <tr>
    <td align="left" style="background-color:#8fd498;">Tele-CTC-FT</td>
    <td align="center" style="background-color:#8fd498;">11.90</td>
    <td align="center" style="background-color:#8fd498;"><ins>7.23</ins></td>
    <td align="center" style="background-color:#8fd498;">23.85</td>
  </tr>
</table>
</p>

Speech understanding performance on WenetSpeech-Wu-Bench
<br>
<p align="center">
<table >
  <tr>
    <th align="left">Model</th>
    <th align="center">ASR</th>
    <th align="center">AST</th>
    <th align="center">Gender</th>
    <th align="center">Age</th>
    <th align="center">Emotion</th>
  </tr>

  <tr>
    <td align="left" style="background-color:#eee;">Qwen3-Omni</td>
    <td align="center" style="background-color:#eee;">44.27</td>
    <td align="center" style="background-color:#eee;">33.31</td>
    <td align="center" style="background-color:#eee;"><b>0.977</b></td>
    <td align="center" style="background-color:#eee;"><ins>0.541</ins></td>
    <td align="center" style="background-color:#eee;"><ins>0.667</ins></td>
  </tr>

  <tr>
    <td align="left" style="background-color:#eee;">Step-Audio2-mini</td>
    <td align="center" style="background-color:#eee;"><ins>26.72</ins></td>
    <td align="center" style="background-color:#eee;"><ins>37.81</ins></td>
    <td align="center" style="background-color:#eee;">0.855</td>
    <td align="center" style="background-color:#eee;">0.370</td>
    <td align="center" style="background-color:#eee;">0.460</td>
  </tr>

  <tr>
    <td align="left" style="background-color:#dfd;">Step-Audio2-Wu-Und⭐</td>
    <td align="center" style="background-color:#dfd;"><b>13.23</b></td>
    <td align="center" style="background-color:#dfd;"><b>53.13</b></td>
    <td align="center" style="background-color:#dfd;"><ins>0.956</ins></td>
    <td align="center" style="background-color:#dfd;"><b>0.729</b></td>
    <td align="center" style="background-color:#dfd;"><b>0.712</b></td>
  </tr>
</table>
</p>

## TTS and Instruct TTS Leaderboard
Bold and underlined values denote the best and second-best results.

TTS results on WenetSpeech-Wu-Bench. 
<p align="center">
<table border="0" cellspacing="0" cellpadding="6" style="border-collapse:collapse;">
  <tr>
    <th align="left">Model</th>
    <th align="center">CER (%)↓</th>
    <th align="center">SIM ↑</th>
    <th align="center">IMOS ↑</th>
    <th align="center">SMOS ↑</th>
    <th align="center">AMOS ↑</th>
    <th align="center">CER (%)↓</th>
    <th align="center">SIM ↑</th>
    <th align="center">IMOS ↑</th>
    <th align="center">SMOS ↑</th>
    <th align="center">AMOS ↑</th>
  </tr>

  <tr>
    <td align="left">Qwen3-TTS†</td>
    <td align="center"><ins>5.95</ins></td>
    <td align="center">--</td>
    <td align="center"><ins>4.35</ins></td>
    <td align="center">--</td>
    <td align="center"><ins>4.19</ins></td>
    <td align="center"><ins>16.45</ins></td>
    <td align="center">--</td>
    <td align="center"><ins>4.03</ins></td>
    <td align="center">--</td>
    <td align="center"><b>3.91</b></td>
  </tr>

  <tr>
    <td align="left">DiaMoE-TTS</td>
    <td align="center">57.05</td>
    <td align="center">0.702</td>
    <td align="center">3.11</td>
    <td align="center">3.43</td>
    <td align="center">3.52</td>
    <td align="center">82.52</td>
    <td align="center">0.587</td>
    <td align="center">2.83</td>
    <td align="center">3.14</td>
    <td align="center">3.22</td>
  </tr>

  <tr>
    <td align="left">CosyVoice2</td>
    <td align="center">10.33</td>
    <td align="center">0.713</td>
    <td align="center">3.83</td>
    <td align="center">3.71</td>
    <td align="center">3.84</td>
    <td align="center">82.49</td>
    <td align="center"><ins>0.618</ins></td>
    <td align="center">3.24</td>
    <td align="center">3.42</td>
    <td align="center">3.37</td>
  </tr>

  <tr>
    <td align="left" style="background-color:#dfd;">CosyVoice2-Wu-CPT⭐</td>
    <td align="center">6.35</td>
    <td align="center"><b>0.727</b></td>
    <td align="center">4.01</td>
    <td align="center"><b>3.84</b></td>
    <td align="center">3.92</td>
    <td align="center">32.97</td>
    <td align="center"><b>0.620</b></td>
    <td align="center">3.72</td>
    <td align="center"><b>3.55</b></td>
    <td align="center">3.63</td>
  </tr>

  <tr>
    <td align="left" style="background-color:#dfd;">CosyVoice2-Wu-SFT⭐</td>
    <td align="center">6.19</td>
    <td align="center"><ins>0.726</ins></td>
    <td align="center">4.32</td>
    <td align="center"><ins>3.78</ins></td>
    <td align="center">4.11</td>
    <td align="center">25.00</td>
    <td align="center">0.601</td>
    <td align="center">3.96</td>
    <td align="center"><ins>3.48</ins></td>
    <td align="center">3.76</td>
  </tr>

  <tr>
    <td align="left" style="background-color:#dfd;">CosyVoice2-Wu-SS⭐</td>
    <td align="center"><b>5.42</b></td>
    <td align="center">--</td>
    <td align="center"><b>4.37</b></td>
    <td align="center">--</td>
    <td align="center"><b>4.21</b></td>
    <td align="center"><b>15.45</b></td>
    <td align="center">--</td>
    <td align="center"><b>4.04</b></td>
    <td align="center">--</td>
    <td align="center"><ins>3.88</ins></td>
  </tr>
</table>
</p>

Performance of instruct TTS model.
<p align="center">
<table border="0" cellspacing="0" cellpadding="6" style="border-collapse:collapse;">
  <tr>
    <th align="left">Type</th>
    <th align="center">Metric</th>
    <th align="center">CosyVoice2-Wu-SFT⭐</th>
    <th align="center">CosyVoice2-Wu-instruct⭐</th>
  </tr>

  <tr>
    <td align="left" rowspan="5">Emotion</td>
    <td align="center">Happy ↑</td>
    <td align="center">0.87</td>
    <td align="center"><b>0.94</b></td>
  </tr>
  <tr>
    <td align="center">Angry ↑</td>
    <td align="center">0.83</td>
    <td align="center"><b>0.87</b></td>
  </tr>
  <tr>
    <td align="center">Sad ↑</td>
    <td align="center">0.84</td>
    <td align="center"><b>0.88</b></td>
  </tr>
  <tr>
    <td align="center">Surprised ↑</td>
    <td align="center">0.67</td>
    <td align="center"><b>0.73</b></td>
  </tr>
  <tr>
    <td align="center">EMOS ↑</td>
    <td align="center">3.66</td>
    <td align="center"><b>3.83</b></td>
  </tr>

  <tr>
    <td align="left" rowspan="3">Prosody</td>
    <td align="center">Pitch ↑</td>
    <td align="center">0.24</td>
    <td align="center"><b>0.74</b></td>
  </tr>
  <tr>
    <td align="center">Speech Rate ↑</td>
    <td align="center">0.26</td>
    <td align="center"><b>0.82</b></td>
  </tr>
  <tr>
    <td align="center">PMOS ↑</td>
    <td align="center">2.13</td>
    <td align="center"><b>3.68</b></td>
  </tr>
</table>
</p>


## ASR & Speech Understanding Inference

This section describes the inference procedures for different speech models used in our experiments, including **Conformer-U2pp-Wu**, **Whisper-Medium-Wu**, **Step-Audio2-Wu-ASR** and **Step-Audio2-Wu-Und**.
Different models are trained and inferred under different frameworks, with corresponding data formats.

For more details, please refer to [Understanding](Understanding/README.md)

---
<!--

### Data Format

#### Conformer-U2pp / Whisper-Medium

The inference data is provided in **JSONL** format, where each line corresponds to one utterance:

```json
{"key": "xxxx", "wav": "xxxxx", "txt": "xxxx"}
````

* `key`: utterance ID
* `wav`: path to the audio file
* `txt`: reference transcription (optional during inference)

---

#### Step-Audio2

The inference data follows a **multi-modal dialogue format**, where audio is provided explicitly:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<audio>语音说了什么"
    },
    {
      "role": "assistant",
      "content": "xxxx"
    }
  ],
  "audios": [
    "xxxx"
  ]
}
```

* `messages`: dialogue-style input/output
* `audios`: path(s) to the audio file(s)

---
 -->
## Conformer-U2pp-Wu

```bash
dir=exp
data_type=raw
decode_checkpoint=$dir/u2++.pt
decode_modes="attention attention_rescoring ctc_prefix_beam_search ctc_greedy_search"
decode_batch=4
test_result_dir=./results
ctc_weight=0.0
reverse_weight=0.0
decoding_chunk_size=-1

python wenet/bin/recognize.py --gpu 0 \
  --modes ${decode_modes} \
  --config $dir/train.yaml \
  --data_type $data_type \
  --test_data $test_dir/$test_set/data.jsonl \
  --checkpoint $decode_checkpoint \
  --beam_size 10 \
  --batch_size ${decode_batch} \
  --blank_penalty 0.0 \
  --ctc_weight $ctc_weight \
  --reverse_weight $reverse_weight \
  --result_dir $test_result_dir \
  ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
```

This setup supports multiple decoding strategies, including attention-based and CTC-based decoding.

---

## Whisper-Medium-Wu

```bash
dir=exp
data_type=raw
decode_checkpoint=$dir/whisper.pt
decode_modes="attention attention_rescoring ctc_prefix_beam_search ctc_greedy_search"
decode_batch=4
test_result_dir=./results
ctc_weight=0.0
reverse_weight=0.0
decoding_chunk_size=-1

python wenet/bin/recognize.py --gpu 0 \
  --modes ${decode_modes} \
  --config $dir/train.yaml \
  --data_type $data_type \
  --test_data $test_dir/$test_set/data.jsonl \
  --checkpoint $decode_checkpoint \
  --beam_size 10 \
  --batch_size ${decode_batch} \
  --blank_penalty 0.0 \
  --ctc_weight $ctc_weight \
  --reverse_weight $reverse_weight \
  --result_dir $test_result_dir \
  ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
```


## Step-Audio2-Wu-ASR & Step-Audio2-Wu-Und
Please download the origin model:[Step-Audio 2 mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini)

```bash
model_dir=Step-Audio-2-mini 
adapter_dir=./checkpoints

CUDA_VISIBLE_DEVICES=0 \
swift infer \
  --model $model_dir \
  --adapters $adapter_dir \
  --val_dataset data.jsonl \
  --max_new_tokens 512 \
  --torch_dtype bfloat16 \
  --result_path results.jsonl
```

---

<!--
## Speech Understanding Tasks

The inference script is identical to that of Step-Audio2 described above; only the user prompt needs to be modified for different tasks.
```json
{
  "ASR": "<audio>请记录下你所听到的语音内容。",
  "AST": "<audio>请仔细聆听这段语音，然后将其内容翻译成普通话。",
  "age": "<audio>请根据语音的声学特征，判断说话人的年龄，从儿童、少年、青年、中年、老年中选一个标签。",
  "gender": "<audio>请根据语音的声学特征，判断说话人的性别，从男性、女性中选一个标签。",
  "emotion": "<audio>请根据语音的声学特征和语义，判断语音的情感，从中立、高兴、难过、惊讶、生气选一个标签。"
}
```
 -->
## TTS Inference

### Install

**Clone and install**

- Clone the repo
``` sh
git clone https://github.com/ASLP-lab/WenetSpeech-Wu-Repo.git
cd WenetSpeech-Wu-Repo/Generation
```

- Create Conda env:

``` sh
conda create -n cosyvoice python=3.10
conda activate cosyvoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

```

### Model download

``` python
from huggingface_hub import snapshot_download
snapshot_download('ASLP-lab/WenetSpeech-Wu-Speech-Generation', local_dir='pretrained_models')
```

### Usage

#### CosyVoice2-Wu-SFT

Prepare Model
``` sh
ln -s ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2/* ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-SFT/
mv ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-SFT/SFT.pt ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-SFT/llm.pt
``` 

Inference
``` python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice_base = CosyVoice2(
    'ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2',
    load_jit=False, load_trt=False, load_vllm=False, fp16=False
)

cosyvoice_sft = CosyVoice2(
    'ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-SFT',
    load_jit=False, load_trt=False, load_vllm=False, fp16=False
)


prompt_speech_16k = load_wav('figs/A0002_S0003_0_G0003_G0004_33.wav', 16000)
prompt_text = "最少辰光阿拉是做撒呃喃，有钞票就是到银行里保本保息。"
text = "<|wuyu|>"+"阿拉屋里向养了一只小猫，伊老欢喜晒太阳的，每日下半天总归蹲辣窗口。"

for i, j in enumerate(cosyvoice_base.inference_instruct2(text, '用上海话说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('A0002_S0003_0_G0003_G0004_33_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice_sft.inference_zero_shot(text, prompt_text, prompt_speech_16k , stream=False)):
    torchaudio.save('A0002_S0003_0_G0003_G0004_33_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```


#### CosyVoice2-Wu-instruct

Prepare Model
``` sh
ln -s ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2/* ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-emotion/
mv ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-emotion/instruct_Emo.pt ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-emotion/llm.pt


ln -s ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2/* ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-prosody/
mv ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-prosody/instruct_Pro.pt ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-prosody/llm.pt
```

Inference
``` python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice_emo = CosyVoice2(
    'ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-emotion',
    load_jit=False, load_trt=False, load_vllm=False, fp16=False
)

cosyvoice_pro = CosyVoice2(
    'ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-prosody',
    load_jit=False, load_trt=False, load_vllm=False, fp16=False
)


prompt_speech_16k = load_wav('figs/A0002_S0003_0_G0003_G0004_33.wav', 16000)
prompt_text = "最少辰光阿拉是做撒呃喃，有钞票就是到银行里保本保息。"
text = "阿拉屋里向养了一只小猫，伊老欢喜晒太阳的，每日下半天总归蹲辣窗口。"

emo_text = "<|开心|><|wuyu|>"+text
for i, j in enumerate(cosyvoice_emo.inference_instruct2(emo_text, '用开心的情感说', prompt_speech_16k, stream=False)):
    torchaudio.save('A0002_S0003_0_G0003_G0004_33_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

pro_text = "<|男性|><|语速快|><|基频高|><|wuyu|>"+text
for i, j in enumerate(cosyvoice_pro.inference_instruct2(pro_text, '这是一位男性，音调很高语速很快地说',prompt_speech_16k, stream=False)):
    torchaudio.save('A0002_S0003_0_G0003_G0004_33_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

```


## Contributors

| <img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/nwpu.png" width="200px"> | <img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/aishelltech.png" width="200px">  | <img src="figs/wenet.png" width="200px"> | <img src="figs/XJTLU_Logo.jpg" width="200px"> |
| ---- | ---- | ---- | ---- |
## Citation
Please cite our paper if you find this work useful:
```
@misc{wang2026wenetspeechwudatasetsbenchmarksmodels,
      title={WenetSpeech-Wu: Datasets, Benchmarks, and Models for a Unified Chinese Wu Dialect Speech Processing Ecosystem}, 
      author={Chengyou Wang and Mingchen Shao and Jingbin Hu and Zeyu Zhu and Hongfei Xue and Bingshen Mu and Xin Xu and Xingyi Duan and Binbin Zhang and Pengcheng Zhu and Chuang Ding and Xiaojun Zhang and Hui Bu and Lei Xie},
      year={2026},
      eprint={2601.11027},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2601.11027}, 
}

```
## Contact

If you are interested in leaving a message to our research team, feel free to email asd6404112a@mail.nwpu.edu.cn or mcshao@mail.nwpu.edu.cn .


You’re welcome to join our WeChat group for technical discussions, updates.
<p align="center">
  <span style="display: inline-block; margin-right: 10px;">
    <img src="figs/wechat.jpg" width="300" alt="WeChat Group QR Code"/>
  </span>
</p>
