from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import numpy as np
import torch
import os
from utils.commons.hparams import set_hparams, hparams


wav2vec2_processor = None
hubert_model = None


def get_hubert_from_16k_wav(wav_16k_name: str):
    speech_16k, _ = sf.read(wav_16k_name)
    if speech_16k.ndim == 2:  # 合并左右声道
        speech_16k = 0.5 * (speech_16k[:, 0] + speech_16k[:, 1])
    hubert = get_hubert_from_16k_speech(speech_16k)
    return hubert


@torch.no_grad()
def get_hubert_from_16k_speech(speech: np.ndarray, device: str = "cuda"):
    global hubert_model, wav2vec2_processor
    if hubert_model is None:
        print("加载HuBERT Model...")
        hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = hubert_model.to(device)
    if wav2vec2_processor is None:
        print("加载Wav2Vec2 Processor...")
        wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values  # [1, T]
    input_values_all = input_values_all.to(device)

    # 对于较长的音频序列, 由于内存限制, 我们无法一次性处理它们
    # HuBERT使用步长为[5,2,2,2,2,2,2]的CNN, 使得总步长为320
    # 卷积核为[10,3,3,3,3,2,2], 使得总卷积为400
    # 因此, 该CNN等价于一个具有核k=400和步长s=320的大型Conv1D
    # 最终得到的时间步长：T = (L - (k - s)) // s

    kernel = 400
    stride = 320
    clip_length = stride * 1000  # 每段clip的长度->汇聚为1000点
    num_iter = input_values_all.shape[1] // clip_length  # 每次处理clip_length个点
    expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride

    res_lst = []
    # 处理定长片段
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length + (kernel - stride)  # 确保每个片段匹配卷积, 得到1000点
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length + (kernel - stride))
        input_values = input_values_all[:, start_idx:end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])

    # 处理最后不定长片段
    input_values = input_values_all[:, clip_length * num_iter :]
    if input_values.shape[1] >= kernel:
        hidden_states = hubert_model.forward(input_values).last_hidden_state
        res_lst.append(hidden_states[0])

    ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
    assert abs(ret.shape[0] - expected_T) <= 1

    return ret


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--video_id", type=str, default="May", help="")
    args = parser.parse_args()
    ### Process Single Long Audio for NeRF dataset
    person_id = args.video_id
    wav_16k_name = f"data/processed/videos/{person_id}/aud.wav"
    hubert_npy_name = f"data/processed/videos/{person_id}/aud_hubert.npy"
    speech_16k, _ = sf.read(wav_16k_name)
    hubert_hidden = get_hubert_from_16k_speech(speech_16k)
    np.save(hubert_npy_name, hubert_hidden.detach().numpy())
    print(f"Saved at {hubert_npy_name}")
