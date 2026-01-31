
# TTS & Intruct TTS Models


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

``` sh
ln -s ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2/* ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-SFT/
mv ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-SFT/SFT.pt ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-SFT/llm.pt
``` 

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

``` sh
ln -s ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2/* ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-emotion/
mv ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-emotion/instruct_Emo.pt ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-emotion/llm.pt


ln -s ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2/* ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-prosody/
mv ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-prosody/instruct_Pro.pt ASLP-lab/WenetSpeech-Wu-Speech-Generation/CosyVoice2-Wu-instruct-prosody/llm.pt
```

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