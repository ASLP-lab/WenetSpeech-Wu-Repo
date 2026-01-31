# ASR & Speech Understanding Model
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


## ASR & Speech Understanding 

This section describes the inference procedures for different speech models used in our experiments, including **Conformer-U2pp-Wu**, **Whisper-Medium-Wu**, **Step-Audio2-Wu-ASR** and **Step-Audio2-Wu-Und**.
Different models are trained and inferred under different frameworks, with corresponding data formats.

---

**Clone**

- Clone the repo for Conformer-U2pp-Wu, Whisper-Medium-Wu
``` sh
git clone https://github.com/wenet-e2e/wenet.git
cd examples/aishell/whisper
```

- Clone the repo for Step-Audio2-Wu-ASR，Step-Audio2-Wu-Und
``` sh
git clone https://github.com/modelscope/ms-swift.git
pip install transformers==4.53.3
```



### Data Format

#### Conformer-U2pp-Wu & Whisper-Medium-Wu

The inference data is provided in **JSONL** format, where each line corresponds to one utterance:

```json
{"key": "xxxx", "wav": "xxxxx", "txt": "xxxx"}
````

* `key`: utterance ID
* `wav`: path to the audio file
* `txt`: reference transcription (optional during inference)

---

#### Step-Audio2-Wu-ASR 

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


#### Step-Audio2-Wu-Und

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
---

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



```

