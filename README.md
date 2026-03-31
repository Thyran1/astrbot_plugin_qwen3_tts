<div align="center">


# astrbot_plugin_qwen3_tts

_基于本地部署Qwen3-TTS的文字转语音插件_


[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![AstrBot](https://img.shields.io/badge/AstrBot-4.0%2B-orange.svg)](https://github.com/Soulter/AstrBot)
[![GitHub](https://img.shields.io/badge/作者-Thyran1-blue)](https://github.com/Thyran1)

</div>

---

## 1. 介绍

`astrbot_plugin_qwen3_tts` 基于本地部署的Qwen3-TTS模型将 AstrBot 文本输出转换为语音输出，支持自定义音色、语音分段。


---

## 2. 安装

### 2.1 部署 Qwen3-TTS

请先完成 Qwen3-TTS 本体部署：

- 官方仓库：[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- 参考指南：[【Qwen3TTS】橘雪莉也能看懂的Qwen3TTS部署教程！](https://www.bilibili.com/video/BV1htcMzFEZB/?spm_id_from=333.1387.favlist.content.click&vd_source=494da3930317b211d09a6c8dd49325b7)

### 2.2 安装 AstrBot 插件

在 AstrBot 插件市场搜索 `astrbot_plugin_qwen3_tts` 并安装。

---

## 3. 快速开始

### 3.1 启动 Qwen3-TTS webui

Qwen3-TTS官方文档中提供了3种模型，分别为CustomVoice model、VoiceDesign model、Base model。此插件需使用Base model。
##
激活 Qwen3-tts 环境
```bat
conda activate Qwen3-tts
```
启用 Base model
```bat
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000
```
打开浏览器输入，启动 weiui
```bat
http://localhost:8000/
```
在 webui 中点击 `Save/Load Voice（保存/加载克隆音色）`

上传`参考音频`，输入`参考音频文本`

点击`Save Voice File（保存音色文件）`

保存音色文件`(.pt)` 

### 3.2 在 AstrBot 面板配置插件


1. `use_gradio_tts` `启用Qwen3-TTS`：Qwen3-TTS文字转语音开关
2. `gradio_server_url` `Qwen3-TTS本地端口地址`：Qwen3-TTS API 地址，默认是 `http://localhost:8000/`
3. `gradio_prompt_file` `音色文件路径`：参考音频路径（必填，建议先用插件默认值）
4. `分段配置`：可根据需求设置分段，模拟正常人语音习惯


---
---

## 4. 配置速查

### 4.1 Qwen3-TTS端口配置

| 字段 | 说明                 | 建议/取值                        |
| --- |--------------------|------------------------------|
| `use_gradio_tts` | 文字转语音开关            | `开 `                         |
| `gradio_server_url` | Qwen3-TTS API 端口地址 | 默认为 `http://localhost:8000/` |
| `gradio_server_timeout` | API请求超时时间          | `30`                         |
| `gradio_prompt_file` | 音色文件（.pt）路径        | `D:\...\...\.pt`             |
| `gradio_save_audio` | 是否保存生成的音频文件        | `开  `                        |
| `gradio_auto_clear_audio` | 是否自动清理生成的音频文件      | `关  `                        |
| `gradio_max_save_file` | 最多保存的文件数量          | `100  `                      |


### 4.2 TTS配置

| 字段                       | 说明        | 建议/取值         |
|--------------------------|-----------|---------------|
| `gradio_tts_probability` | TTS触发概率   | `0.8`         |
| `tts_min_length`         | TTS最小文本长度 | `5` |
| `tts_max_length`         | TTS最大文本长度 | `100`         |

### 4.3 分段配置

这些参数会作为每次请求 `/tts` 的默认值：

| 字段                           | 说明        | 建议/取值           |
|------------------------------|-----------|-----------------|
| `enable_split`               | 启用分段      | `开 `            |
| `split_llmonly`              | 只处理LLM结果  | `开`             |
| `force_split_chars`          | 强制分段的标点符号 | `. 。？！；; \n \s` |
| `max_segments`               | 最大分段数量    | 建议`5-8`         |
| `delay_strategy`             | 延迟模式      |                 |
| `linear_base`                | 按字数延迟基础时间 |                 |
| `linear_factor`              | 按字数延迟系数   |                 |
| `linear_max`                 | 按字数延迟最大值  | 建议`10`           |
| `random_min`                 | 随机延迟最小值   |                 |
| `random_max`                 | 随机延迟最大值   |                 |
| `fixed_delay`                | 固定延迟时间    |                 |
| `enable_probabilistic_split` | 启用概率分段    |                 |a
| `probabilistic_split_chars`  | 概率分段字符    | 推荐`, ，`         |
| `split_probability`          | 概率分段触发概率  |                 |


---

## 
##  贡献
- ⭐ 如果对你有用请Star 这个项目！



##  致谢

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)， an open-source series of TTS models developed by the Qwen team at Alibaba Cloud




