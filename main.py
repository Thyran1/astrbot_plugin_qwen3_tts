import asyncio
import concurrent.futures
import os
import random
import re
import shutil
import threading
import time
from pathlib import Path

# Qwen3 TTS 相关导入
from gradio_client import Client, handle_file

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import BaseMessageComponent, Plain, Record, Reply
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, register
from astrbot.core.star import StarTools
from astrbot.core.star.session_llm_manager import SessionServiceManager


@register(
    "astrbot_plugin_qwen3_tts",
    "Thyran1",
    "基于本地部署的Qwen3-TTS，为astrbot提供文本转语音(TTS)服务，可自定义音色",
    "1.0.0",
)
class Qwen3TTS(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.plugin_name = "astrbot_plugin_qwen3_tts"
        # 数据目录（使用框架规范）
        self.data_dir: Path = StarTools.get_data_dir(self.plugin_name)
        self.data_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

        self.use_gradio_tts = config.get("use_gradio_tts", False)

        # Qwen3-tts端口配置
        self.gradio_server_url = config.get("client", {}).get("gradio_server_url")
        self.gradio_server_timeout = config.get("client", {}).get(
            "gradio_server_timeout", 30.0
        )
        self.gradio_prompt_file = config.get("client", {}).get("gradio_prompt_file")
        self.gradio_save_audio = True
        # self.gradio_save_audio = config.get("client", {}).get("gradio_save_audio", True)
        self.gradio_auto_clear_audio = config.get("client", {}).get(
            "gradio_auto_clear_audio", False
        )
        self.gradio_max_save_file = config.get("client", {}).get(
            "gradio_max_save_file", 100
        )
        self._gradio_client = None  # 懒加载
        self._gradio_client_lock = threading.Lock()  # 线程锁
        self._gradio_predict_lock = threading.Lock()
        max_workers = 5
        self.tts_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )

        # 分段配置
        self.pair_map = {
            '"': '"',
            "《": "》",
            "（": "）",
            "(": ")",
            "[": "]",
            "{": "}",
            "【": "】",
            "<": ">",
        }
        self.quote_chars = {'"', "`"}

        # tts配置
        self.gradio_tts_probability = config.get("tts_control", {}).get(
            "gradio_tts_probability", 0.5
        )
        self.tts_min_length = config.get("tts_control", {}).get("tts_min_length", 5)
        self.tts_max_length = config.get("tts_control", {}).get("tts_max_length", 30)

        # 分段控制（统一使用 split_control 命名空间）
        self.enable_probabilistic_split = config.get("split_control", {}).get(
            "enable_probabilistic_split", False
        )
        self.split_llmonly = config.get("split_control", {}).get("split_llmonly", True)
        self.force_split_chars = config.get("split_control", {}).get(
            "force_split_chars", ". 。？！；; \\n"
        )
        self.probabilistic_split_chars = config.get("split_control", {}).get(
            "probabilistic_split_chars", ",，"
        )
        self.split_probability = config.get("split_control", {}).get(
            "split_probability", 0.5
        )

        # 延迟策略配置（统一放在 split_control 下）
        self.delay_strategy = config.get("split_control", {}).get(
            "delay_strategy", "按字数"
        )
        self.random_min = (
            config.get("split_control", {})
            .get("random_control", {})
            .get("random_min", 1.0)
        )
        self.random_max = (
            config.get("split_control", {})
            .get("random_control", {})
            .get("random_max", 3.0)
        )
        self.linear_base = (
            config.get("split_control", {})
            .get("linear_control", {})
            .get("linear_base", 0.5)
        )
        self.linear_factor = (
            config.get("split_control", {})
            .get("linear_control", {})
            .get("linear_factor", 0.1)
        )
        self.linear_max = (
            config.get("split_control", {})
            .get("linear_control", {})
            .get("linear_max", 10.0)
        )
        self.fixed_delay = (
            config.get("split_control", {})
            .get("fixed_control", {})
            .get("fixed_delay", 1.5)
        )

    # ---------- Qwen3-tts 端口调用 ----------
    def _get_gradio_client(self):
        """线程安全的 Gradio 客户端获取"""
        with self._gradio_client_lock:
            if self._gradio_client is None:
                self._gradio_client = Client(self.gradio_server_url)
            return self._gradio_client

    def _merge_continuous_plain(
        self, components: list[BaseMessageComponent]
    ) -> list[BaseMessageComponent]:
        """合并列表中连续的 Plain 组件为一个 Plain，其他组件保持原序"""
        if not components:
            return []
        merged = []
        current_text = ""
        for comp in components:
            if isinstance(comp, Plain):
                current_text += comp.text
            else:
                if current_text:
                    merged.append(Plain(current_text))
                    current_text = ""
                merged.append(comp)
        if current_text:
            merged.append(Plain(current_text))
        return merged

    def _cleanup_old_audio(self):
        """清理旧的音频文件，保留最近的max_save_file个，避免磁盘膨胀"""
        if not self.gradio_auto_clear_audio:
            logger.debug("[Qwen3-TTS] 自动清理未开启")
            return

        try:
            files = []
            for f in self.data_dir.iterdir():
                if f.name.startswith("tts_") and f.suffix == ".wav":
                    files.append(f)

            if len(files) <= self.gradio_max_save_file:
                logger.debug(
                    f"[Qwen3-TTS] 音频文件数量 {len(files)} ≤ {self.gradio_max_save_file}，无需清理"
                )
                return

            # 按修改时间倒序，保留最近的文件
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            now = time.time()
            to_delete = []

            for f in files[self.gradio_max_save_file :]:
                file_age = now - f.stat().st_mtime
                if file_age > 300:
                    to_delete.append(f)
                else:
                    # 虽然数量超出，但文件太新，暂不删除
                    logger.debug(
                        f"[Qwen3-TTS] 跳过删除新文件: {f.name}, 年龄 {file_age:.1f}s < {300}s"
                    )

            for old_file in to_delete:
                try:
                    if old_file.exists():
                        old_file.unlink()
                        logger.debug(f"[Qwen3-TTS] 清理旧音频: {old_file.name}")
                except Exception as e:
                    logger.warning(f"[Qwen3-TTS] 删除文件 {old_file.name} 失败: {e}")

        except Exception as e:
            logger.exception(f"[Qwen3-TTS] 清理旧音频失败: {e}")

    async def generate_tts(self, text: str) -> Path | None:
        """生成 TTS 音频文件，返回文件路径"""
        try:
            timestamp = str(time.time_ns())
            wav_file = self.data_dir / f"tts_{timestamp}.wav"

            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(
                    self.tts_executor, self._call_gradio_tts, text, wav_file
                ),
                timeout=self.gradio_server_timeout,
            )
            # 生成成功后清理旧文件（异步非阻塞）
            asyncio.create_task(asyncio.to_thread(self._cleanup_old_audio))

            return wav_file if wav_file.exists() else None

        except asyncio.TimeoutError:
            logger.error(f"generate_tts 超时: {text[:50]}...")
            return None
        except Exception as e:
            logger.error(f"generate_tts error: {e}")
            return None

    def _call_gradio_tts(self, text: str, output_path: Path):
        """同步调用 Qwen3-tts 服务（在线程池中执行）"""
        try:
            prompt_file = Path(self.gradio_prompt_file)
            if not prompt_file.exists():
                raise FileNotFoundError(f"提示文件不存在: {prompt_file}")

            client = self._get_gradio_client()
            """互斥锁"""
            with self._gradio_predict_lock:
                result = client.predict(
                    file_obj=handle_file(self.gradio_prompt_file),
                    text=text,
                    lang_disp="Auto",
                    api_name="/load_prompt_and_gen",
                )
            logger.debug(f"[Qwen3-TTS] 原始返回: {result}")

            if not isinstance(result, (list, tuple)) or len(result) < 2:
                raise ValueError(f"返回格式错误: {result}")

            audio_path, status = result[0], result[1]

            # 类型保护：确保 status 为字符串
            if not isinstance(status, str):
                raise Exception(f"Qwen3-TTS 返回状态不是字符串: {status}")

            if not audio_path or not isinstance(audio_path, str):
                raise ValueError(f"音频路径无效: {audio_path}")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"临时音频文件不存在: {audio_path}")

            # ----- 路径安全性校验（信任边界） -----
            # 定义允许的安全目录列表
            safe_dirs = [
                self.data_dir,  # 插件数据目录
                Path("/tmp"),  # Linux/macOS 临时目录
                Path("/var/tmp"),  # Linux/macOS 持久临时目录
            ]

            # 仅添加非空的环境变量
            for env_var in ["TEMP", "TMP"]:
                env_path = os.environ.get(env_var)
                if env_path:
                    safe_dirs.append(Path(env_path))

            # 过滤空路径并转为绝对路径（确保 resolve 时不会出错）
            safe_dirs = [d.resolve() for d in safe_dirs if d and d.exists()]

            # 获取音频文件的真实路径（解析符号链接，避免软链接绕过）
            real_audio_path = Path(audio_path).resolve()

            # 检查是否在任一安全目录内
            is_safe = any(
                real_audio_path.is_relative_to(safe_dir) or real_audio_path == safe_dir
                for safe_dir in safe_dirs
            )

            if not is_safe:
                raise ValueError(f"音频路径不在安全目录内: {audio_path}")

            if not real_audio_path.exists():
                raise FileNotFoundError(f"临时音频文件不存在: {real_audio_path}")

            # 状态检查
            error_keywords = ["error", "fail", "失败", "异常"]
            if any(keyword in status.lower() for keyword in error_keywords):
                raise Exception(f"Qwen3-TTS 返回错误状态: {status}")

            # 复制文件
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(real_audio_path, output_path)
            logger.info(f"[Qwen3-TTS] 已保存至: {output_path}")

        except Exception as e:
            logger.error(f"_call_gradio_tts error: {e}")
            raise

    # ------------------------------------------------------------------------

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        setattr(event, "__is_llm_reply", True)

    @filter.on_decorating_result(priority=-100000000000000000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        # 1. 基础防重入与校验
        if getattr(event, "__splitter_processed", False):
            return

        result = event.get_result()
        if not result or not result.chain:
            return

        enable_split = self.config.get("enable_split", False)
        if not enable_split:
            # 对整个消息链应用 TTS
            processed_chain = await self._process_tts_for_segment(event, result.chain)
            result.chain.clear()
            result.chain.extend(processed_chain)
            setattr(event, "__splitter_processed", True)
            return

        # 2. 作用范围检查
        split_llmonly = self.split_llmonly
        is_llm_reply = getattr(event, "__is_llm_reply", False)

        if split_llmonly and not is_llm_reply:
            return

        # 标记已处理
        setattr(event, "__splitter_processed", True)

        # 3. 执行分段
        """
        if self.enable_probabilistic_split:
            split_chars = self.config.get("split_control", {}).get("split_chars") + self.config.get("split_control", {}).get("probabilistic_split_chars", "，,")

        else:
            split_chars = self.config.get("split_control", {}).get("split_chars")
        """

        # 概率分段优化
        force_split_chars = self.force_split_chars
        probabilistic_split_chars = self.probabilistic_split_chars

        split_chars = ""

        if not self.enable_probabilistic_split:
            split_chars = force_split_chars

        else:
            if not force_split_chars:
                split_chars = probabilistic_split_chars
            else:
                if not probabilistic_split_chars:
                    split_chars = force_split_chars
                else:
                    split_chars = force_split_chars + probabilistic_split_chars

        if not split_chars:
            processed_chain = await self._process_tts_for_segment(event, result.chain)
            result.chain.clear()
            result.chain.extend(processed_chain)
            return
            # 对用户提供的标点符号进行转义，并构建字符类

        # 1. 定义需要识别的「正则特殊令牌」（可根据需要扩展，如 \d、\w、\t、\n 等）
        special_token_re = re.compile(r"\\[sdwtn]")  # 匹配 \s、\d、\w、\t、\n

        # 2. 从 split_chars 中提取「特殊令牌」和「普通字符」
        special_tokens = special_token_re.findall(
            split_chars
        )  # 提取所有特殊令牌（如 ['\\s', '\\n']）
        normal_chars = special_token_re.sub(
            "", split_chars
        )  # 移除特殊令牌，剩下普通字符

        # 3. 构建正则模式的各个分支
        pattern_parts = []

        # 处理普通字符：转义后放入字符类 [...]
        if normal_chars:
            escaped_normal = re.escape(normal_chars)
            pattern_parts.append(f"[{escaped_normal}]")

        # 处理特殊令牌：直接保留其正则含义
        pattern_parts.extend(special_tokens)

        # 4. 组合成最终正则（匹配一个或多个分隔符）
        if not pattern_parts:
            split_pattern = None  # 无分隔符时的兜底处理
        else:
            split_pattern = f"(?:{'|'.join(pattern_parts)})+"

        # split_pattern = f"[{re.escape(split_chars)}]+"

        do_split = True
        max_segs = self.config.get("split_control", {}).get("max_segments", 7)
        enable_reply = False

        strategies = {
            "image": "单独",
            "at": "跟随下段",
            "face": "嵌入",
            "default": "跟随下段",
        }

        segments = self.split_chain(
            result.chain, split_pattern, strategies, enable_reply
        )

        # 4. 最大分段数限制
        if len(segments) > max_segs and max_segs > 0:
            final_segments = segments[: max_segs - 1]
            # 合并超出部分的所有组件
            merged_raw = []
            for seg in segments[max_segs - 1 :]:
                merged_raw.extend(seg)
            # 合并连续的 Plain 组件
            merged_last = self._merge_continuous_plain(merged_raw)
            final_segments.append(merged_last)
            segments = final_segments

        # 如果只有一段，且不需要清理，直接放行
        if len(segments) <= 1:
            processed_chain = await self._process_tts_for_segment(event, result.chain)
            result.chain.clear()
            result.chain.extend(processed_chain)
            return

        # 5. 注入引用 (Reply) - 仅第一段
        if enable_reply and segments and event.message_obj.message_id:
            has_reply = any(isinstance(c, Reply) for c in segments[0])
            if not has_reply:
                segments[0].insert(0, Reply(id=event.message_obj.message_id))

        # 6. 发送前 N-1 段
        for i in range(len(segments) - 1):
            segment_chain = segments[i]

            # 空内容检查
            text_content = "".join(
                [c.text for c in segment_chain if isinstance(c, Plain)]
            )
            has_media = any(not isinstance(c, Plain) for c in segment_chain)
            if not text_content.strip() and not has_media:
                continue

            try:
                # 处理TTS
                segment_chain = await self._process_tts_for_segment(
                    event, segment_chain
                )

                # 日志输出
                self._log_segment(i + 1, len(segments), segment_chain, "")

                # 使用 event.send() 发送消息
                mc = MessageChain()
                mc.chain = segment_chain
                await event.send(mc)

                # 延迟
                wait_time = self.calculate_delay(text_content)
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"发送分段 {i + 1} 失败: {e}")

        # 7. 处理最后一段
        last_segment = segments[-1]

        last_text = "".join([c.text for c in last_segment if isinstance(c, Plain)])
        last_has_media = any(not isinstance(c, Plain) for c in last_segment)

        if not last_text.strip() and not last_has_media:
            result.chain.clear()
            result.chain.append(Plain("."))
        else:
            # 日志输出
            self._log_segment(len(segments), len(segments), last_segment, "")

            # 对最后一段也应用 TTS
            last_segment = await self._process_tts_for_segment(event, last_segment)

            result.chain.clear()
            result.chain.extend(last_segment)

    def _log_segment(
        self, index: int, total: int, chain: list[BaseMessageComponent], method: str
    ):
        """输出单行段落内容日志"""
        content_str = ""
        for comp in chain:
            if isinstance(comp, Plain):
                content_str += comp.text
            else:
                content_str += f"[{type(comp).__name__}]"

        log_content = content_str.replace("\n", "\\n")
        logger.info(f"第 {index}/{total} 段 ({method}): {log_content}")

    async def _process_tts_for_segment(
        self, event: AstrMessageEvent, segment: list[BaseMessageComponent]
    ) -> list[BaseMessageComponent]:
        """为分段处理 TTS（根据配置使用 Gradio TTS 或框架 TTS）"""
        enable_tts_for_segments = self.config.get("enable_tts_for_segments", True)
        if not enable_tts_for_segments:
            return segment

        total_text_len = sum(
            len(comp.text) for comp in segment if isinstance(comp, Plain)
        )
        if total_text_len < self.tts_min_length or total_text_len > self.tts_max_length:
            logger.info(
                f"段落字数 {total_text_len} 超出范围 [{self.tts_min_length}, {self.tts_max_length}]，跳过 TTS"
            )
            return segment

        # ----- Gradio TTS -----
        if self.use_gradio_tts and self.gradio_server_url and self.gradio_prompt_file:
            rand_float = random.random()
            if rand_float > self.gradio_tts_probability:
                return segment

            new_segment = []
            for comp in segment:
                if isinstance(comp, Plain) and len(comp.text) > 1:
                    try:
                        logger.info(f"[Qwen3-TTS] 请求: {comp.text[:50]}...")
                        audio_path = await self.generate_tts(comp.text)
                        if audio_path:
                            new_segment.append(
                                Record(file=str(audio_path), url=str(audio_path))
                            )
                        else:
                            logger.warning("[Qwen3-TTS] 生成失败，使用原文本")
                            new_segment.append(comp)
                    except Exception as e:
                        logger.error(f"[Qwen3-TTS] 处理失败: {e}，使用原文本")
                        new_segment.append(comp)
                else:
                    new_segment.append(comp)
            return new_segment

        # ----- 框架 TTS -----
        try:
            all_config = self.context.get_config(event.unified_msg_origin)
            tts_config = all_config.get("provider_tts_settings", {})
            tts_enabled = tts_config.get("enable", False)

            if not tts_enabled:
                return segment

            tts_provider = self.context.get_using_tts_provider(event.unified_msg_origin)
            if not tts_provider:
                return segment

            result = event.get_result()
            if not result or not result.is_llm_result():
                return segment

            if not await SessionServiceManager.should_process_tts_request(event):
                return segment

            tts_trigger_probability = tts_config.get("trigger_probability", 1.0)
            try:
                tts_trigger_probability = max(
                    0.0, min(float(tts_trigger_probability), 1.0)
                )
            except (TypeError, ValueError):
                tts_trigger_probability = 1.0

            tts_random = random.random()
            if tts_random > tts_trigger_probability:
                logger.info(
                    f"框架 TTS 概率 {tts_random:.2f} > 设定概率 {tts_trigger_probability:.2f}，跳过 TTS"
                )
                return segment

            dual_output = tts_config.get("dual_output", False)

            new_segment = []
            for comp in segment:
                if isinstance(comp, Plain) and len(comp.text) > 1:
                    try:
                        logger.info(f"框架 TTS 请求: {comp.text[:50]}...")
                        audio_path = await tts_provider.get_audio(comp.text)
                        if audio_path:
                            new_segment.append(Record(file=audio_path, url=audio_path))
                            if dual_output:
                                new_segment.append(comp)
                        else:
                            new_segment.append(comp)
                    except Exception as e:
                        logger.error(f"框架 TTS 失败: {e}")
                        new_segment.append(comp)
                else:
                    new_segment.append(comp)
            return new_segment

        except Exception as e:
            logger.error(f"TTS 配置检查失败: {e}，跳过TTS处理")
            return segment

    def calculate_delay(self, text: str) -> float:
        """根据配置的延迟策略计算等待时间"""
        if self.delay_strategy == "随机":
            return random.uniform(self.random_min, self.random_max)
        elif self.delay_strategy == "按字数":
            return min(
                self.linear_max, self.linear_base + (len(text) * self.linear_factor)
            )

        else:
            return self.fixed_delay

    def split_chain(
        self,
        chain: list[BaseMessageComponent],
        pattern: str,
        strategies: dict[str, str],
        enable_reply: bool,
    ) -> list[list[BaseMessageComponent]]:
        """将消息链按正则 pattern 切分，并根据组件类型采取不同策略"""
        segments = []
        current_chain_buffer = []

        for component in chain:
            if isinstance(component, Plain):
                text = component.text
                if not text:
                    continue

                self._process_text_(text, pattern, segments, current_chain_buffer)
            else:
                c_type = type(component).__name__.lower()
                if "reply" in c_type:
                    if enable_reply:
                        current_chain_buffer.append(component)
                    continue

                if "image" in c_type:
                    strategy = strategies["image"]
                elif "at" in c_type:
                    strategy = strategies["at"]
                elif "face" in c_type:
                    strategy = strategies["face"]
                else:
                    strategy = strategies["default"]

                if strategy == "单独":
                    if current_chain_buffer:
                        segments.append(current_chain_buffer[:])
                        current_chain_buffer.clear()
                    segments.append([component])
                elif strategy == "跟随上段":
                    if current_chain_buffer:
                        current_chain_buffer.append(component)
                    elif segments:
                        segments[-1].append(component)
                    else:
                        current_chain_buffer.append(component)
                else:  # 跟随下段
                    current_chain_buffer.append(component)

        if current_chain_buffer:
            segments.append(current_chain_buffer)
        return [seg for seg in segments if seg]

    def _process_text_(self, text: str, pattern: str, segments: list, buffer: list):
        """智能切分文本，支持括号/引号平衡和概率切分"""
        stack = []
        compiled_pattern = re.compile(pattern)
        i = 0
        n = len(text)
        current_chunk = ""

        while i < n:
            char = text[i]
            is_opener = char in self.pair_map
            if char in self.quote_chars:
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    stack.append(char)
                current_chunk += char
                i += 1
                continue

            if stack:
                expected_closer = self.pair_map.get(stack[-1])
                if char == expected_closer:
                    stack.pop()
                elif is_opener:
                    stack.append(char)
                current_chunk += char
                i += 1
                continue

            if is_opener:
                stack.append(char)
                current_chunk += char
                i += 1
                continue

            match = compiled_pattern.match(text, pos=i)
            if match:
                delimiter = match.group()
                # 概率切分判断
                if self.enable_probabilistic_split and any(
                    ch in self.probabilistic_split_chars for ch in delimiter
                ):
                    if random.random() < self.split_probability:
                        current_chunk += delimiter
                        buffer.append(Plain(current_chunk))
                        segments.append(buffer[:])
                        buffer.clear()
                        current_chunk = ""
                    else:
                        current_chunk += delimiter
                else:
                    current_chunk += delimiter
                    buffer.append(Plain(current_chunk))
                    segments.append(buffer[:])
                    buffer.clear()
                    current_chunk = ""
                i += len(delimiter)
            else:
                current_chunk += char
                i += 1

        if current_chunk:
            buffer.append(Plain(current_chunk))
