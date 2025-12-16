# Wan Text-to-Video

轻量封装了 Wan 的文生视频推理：

```
aigc_homework/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ model.py      # 载入 Wan T2V 1.3B/14B
│  └─ sample.py     # 推理脚本：文本 -> video.mp4
└─ results/
   └─ video.mp4     # 输出样例（占位演示，可被真实生成结果覆盖）
```
原论文：
[**Wan: Open and Advanced Large-Scale Video Generative Models**](https://arxiv.org/abs/2503.20314) <be>


## 1) 项目做了什么
- 抽取 Wan 文生视频推理流程，提供一个最小可运行脚本 `src/sample.py`。
- 默认使用 1.3B（轻量）；作业示例固定跑 T2V-1.3B。
- 将生成结果保存为 `results/video.mp4`，便于直接提交作业。

## 2) 如何配置和运行
1. 安装依赖（复用仓库原始依赖即可）：
   ```bash
   cd wan_t2v_assignment
   pip install -r requirements.txt
   ```
2. 下载权重：

   ```bash
   # 任选其一：HuggingFace / ModelScope
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
   ```
   or
   ```bash
   sh download_model.sh
   ```
3. 运行推理：
   ```bash
   python src/sample.py \
     --prompt "A cozy campfire at night with glowing embers and fireflies." 
   ```
   - GPU 推荐 >= 12GB VRAM（1.3B），14B 需更高显存。
   - 若显存吃紧，可加 `--t5_cpu` 或保持默认的模型 offload。

## 3) 期望输出
- 生成的 MP4 视频保存在 `results/video.mp4`（默认 16 FPS，81 帧，对应约 5 秒）。
- 当前仓库内的 `results/video.mp4` 是占位演示视频，跑通脚本后会被真实模型输出覆盖。




