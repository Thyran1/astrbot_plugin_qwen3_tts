import re
import math
import random
import asyncio
import hashlib
import os
import shutil
import time
from typing import List, Dict

from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import LLMResponse
from astrbot.api.message_components import Plain, BaseMessageComponent, Reply, Record
from astrbot.core.star.session_llm_manager import SessionServiceManager

# Qwen3 TTS 相关导入
from gradio_client import Client, handle_file

@register("astrbot_plugin_qwen3_tts", "Thyran1", "基于本地部署的Qwen3-TTS，为astrbot提供文本转语音(TTS)服务，可自定义音色", "1.0.0")
class MessageSplitterPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config



        self.use_gradio_tts = config.get("use_gradio_tts", False)

        # Qwen3-tts端口配置
        self.gradio_server_url = config.get("client", {}).get("gradio_server_url")
        self.gradio_prompt_file = config.get("client", {}).get("gradio_prompt_file")
        self.gradio_save_audio = config.get("client", {}).get("gradio_save_audio",True)


        self._gradio_client = None  # 懒加载


        #分段配置
        self.pair_map = {
            '"': '"', '《': '》', '（': '）', '(': ')',
            '[': ']', '{': '}', "'": "'", '【': '】', '<': '>'
        }
        self.quote_chars = {'"', "'", "`"}

        #tts配置

        self.gradio_tts_probability = config.get("tts_control", {}).get("gradio_tts_probability", 0.5)
        self.tts_min_length = config.get("tts_control", {}).get("tts_min_length", 5)
        self.tts_max_length = config.get("tts_control", {}).get("tts_max_length", 30)

        # 分段配置/概率分割
        #self.enable_split = config.get("enable_split", False)
        self.enable_probabilistic_split = config.get("split_control", {}).get("enable_probabilistic_split", False)
        self.probabilistic_split_chars = config.get("split_control", {}).get("probabilistic_split_chars", ", ，")
        self.split_probability = config.get("split_control", {}).get("split_probability", 0.5)

    # ---------- Qwen3-tts 端口调用 ----------
    def _get_gradio_client(self):
        if self._gradio_client is None:
            self._gradio_client = Client(self.gradio_server_url)
        return self._gradio_client

    def _merge_continuous_plain(self, components: List[BaseMessageComponent]) -> List[BaseMessageComponent]:
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

    async def generate_tts(self, text: str) -> str:
        """生成 TTS 音频文件，返回文件路径"""
        try:
            data_dir = self.get_plugin_data_dir()

            if self.gradio_save_audio:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                wav_file = os.path.join(data_dir, f"tts_{timestamp}.wav")
            else:
                text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                wav_file = os.path.join(data_dir, f"temp_{text_hash}.wav")

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._call_gradio_tts,
                text,
                wav_file
            )

            return wav_file if os.path.exists(wav_file) else None

        except Exception as e:
            logger.error(f"generate_tts error: {e}")
            return None

    def _call_gradio_tts(self, text: str, output_path: str):
        """同步调用 Qwen3-tts 服务（在线程池中执行）"""
        try:
            if not os.path.exists(self.gradio_prompt_file):
                raise FileNotFoundError(f"提示文件不存在: {self.gradio_prompt_file}")

            client = self._get_gradio_client()
            result = client.predict(
                file_obj=handle_file(self.gradio_prompt_file),
                text=text,
                lang_disp="Auto",
                api_name="/load_prompt_and_gen"
            )
            logger.debug(f"[Qwen3-TTS] 原始返回: {result}")

            if not isinstance(result, (list, tuple)) or len(result) < 2:
                raise ValueError(f"返回格式错误: {result}")
            audio_path, status = result[0], result[1]

            if not audio_path or not isinstance(audio_path, str):
                raise ValueError(f"音频路径无效: {audio_path}")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"临时音频文件不存在: {audio_path}")

            # 放宽状态检查
            error_keywords = ["error", "fail", "失败", "异常"]
            if any(keyword in status.lower() for keyword in error_keywords):
                raise Exception(f"Qwen3-TTS 返回错误状态: {status}")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(audio_path, output_path)
            logger.info(f"[Qwen3-TTS] 已保存至: {output_path}")

        except Exception as e:
            logger.error(f"_call_gradio_tts error: {e}")
            raise

    def get_plugin_data_dir(self):
        """获取插件数据存储目录"""
        current_dir = os.path.dirname(__file__)
        plugin_name = "message_splitter"  # 可根据需要修改
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(current_dir)),
            "plugin_data",
            plugin_name
        )
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
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

        enable_split = self.config.get("enable_split",False)
        if not enable_split:
            # 对整个消息链应用 TTS
            processed_chain = await self._process_tts_for_segment(event, result.chain)
            result.chain.clear()
            result.chain.extend(processed_chain)
            setattr(event, "__splitter_processed", True)
            return

        # 2. 作用范围检查
        split_llmonly = self.config.get("split_char", {}).get("split_llmonly", True)
        is_llm_reply = getattr(event, "__is_llm_reply", False)

        if split_llmonly and not is_llm_reply:
            return

        # 3. 长度限制检查
        split_limit = 0
        total_text_len = sum(len(c.text) for c in result.chain if isinstance(c, Plain))

        if split_limit > 0 and total_text_len < split_limit:
            return

        # 标记已处理
        setattr(event, "__splitter_processed", True)

        # 4. 概率分段
        if self.enable_probabilistic_split:
         split_chars = self.config.get("split_control", {}).get("split_char", ". 。？！；; \\n \\s") + self.config.get("split_control",{}).get("probabilistic_split_chars", "，,")

        else:
            split_chars = self.config.get("split_control", {}).get("split_char", ". 。？！；; \\n \\s")

        split_pattern = f"[{re.escape(split_chars)}]+"



        do_split = True
        max_segs = self.config.get("split_control", {}).get("max_segments", 7)
        enable_reply = False

        strategies = {
            'image': "单独",
            'at': "跟随下段",
            'face': "嵌入",
            'default': "跟随下段"
        }

        # 5. 执行分段
        segments = self.split_chain(result.chain, split_pattern, do_split, strategies, enable_reply)

        # 6. 最大分段数限制


        if len(segments) > max_segs and max_segs > 0:
            # logger.warning(f"分段数({len(segments)}) 超过限制({max_segs})，合并剩余段落。")
            final_segments = segments[:max_segs - 1]
            # 合并超出部分的所有组件
            merged_raw = []
            for seg in segments[max_segs - 1:]:
                merged_raw.extend(seg)
            # 合并连续的 Plain 组件
            merged_last = self._merge_continuous_plain(merged_raw)
            final_segments.append(merged_last)
            segments = final_segments



        # 如果只有一段，且不需要清理，直接放行
        if len(segments) <= 1 :
            return

        # 7. 注入引用 (Reply) - 仅第一段
        if enable_reply and segments and event.message_obj.message_id:
            has_reply = any(isinstance(c, Reply) for c in segments[0])
            if not has_reply:
                segments[0].insert(0, Reply(id=event.message_obj.message_id))





        # 发送前 N-1 段
        for i in range(len(segments) - 1):
            segment_chain = segments[i]

            # 空内容检查
            text_content = "".join([c.text for c in segment_chain if isinstance(c, Plain)])
            has_media = any(not isinstance(c, Plain) for c in segment_chain)
            if not text_content.strip() and not has_media:
                continue

            try:
                # --- 处理TTS（根据配置选择 Gradio 或框架 TTS）---
                segment_chain = await self._process_tts_for_segment(event, segment_chain)
                # ---------------

                # --- 日志输出 ---
                self._log_segment(i + 1, len(segments), segment_chain, "")
                # ---------------

                mc = MessageChain()
                mc.chain = segment_chain
                await self.context.send_message(event.unified_msg_origin, mc)

                # 延迟
                wait_time = self.calculate_delay(text_content)
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"发送分段 {i+1} 失败: {e}")

        # 9. 处理最后一段
        last_segment = segments[-1]

        last_text = "".join([c.text for c in last_segment if isinstance(c, Plain)])
        last_has_media = any(not isinstance(c, Plain) for c in last_segment)

        if not last_text.strip() and not last_has_media:
            result.chain.clear()
            result.chain.append(Plain("."))

        else:
            # --- 日志输出 ---
            self._log_segment(len(segments), len(segments), last_segment, "")
            # ---------------

            # 对最后一段也应用 TTS（如果需要）
            last_segment = await self._process_tts_for_segment(event, last_segment)

            result.chain.clear()
            result.chain.extend(last_segment)

    def _log_segment(self, index: int, total: int, chain: List[BaseMessageComponent], method: str):
        """输出单行段落内容日志"""
        content_str = ""
        for comp in chain:
            if isinstance(comp, Plain):
                content_str += comp.text
            else:
                content_str += f"[{type(comp).__name__}]"

        # 替换换行符以便在单行日志中显示
        log_content = content_str.replace('\n', '\\n')
        logger.info(f"第 {index}/{total} 段 ({method}): {log_content}")

    async def _process_tts_for_segment(self, event: AstrMessageEvent, segment: List[BaseMessageComponent]) -> List[BaseMessageComponent]:
        """为分段处理 TTS（根据配置使用 Gradio TTS 或框架 TTS）"""
        # 检查是否启用分段 TTS（总开关）
        enable_tts_for_segments = self.config.get("enable_tts_for_segments", True)
        if not enable_tts_for_segments:
            return segment


        # 计算段落纯文本总长度
        total_text_len = sum(len(comp.text) for comp in segment if isinstance(comp, Plain))
        # 如果段落总字数小于阈值，直接返回原段落（不进行 TTS）
        if total_text_len < self.tts_min_length:
            #logger.debug(f"[Splitter] 段落字数 {total_text_len} < {self.tts_min_length}，跳过 TTS")
            logger.info(f"段落字数 {total_text_len} < {self.tts_min_length}，跳过 TTS")
            #self._log_segment("小于字数，跳过tts")
            return segment

        if total_text_len > self.tts_max_length:
            #logger.debug(f"段落字数 {total_text_len} > {self.tts_min_length}，跳过 TTS")
            #self._log_segment("大于字数，跳过tts")
            logger.info(f"段落字数 {total_text_len} > {self.tts_max_length}，跳过 TTS")
            return segment



        # ----- 如果启用了 Gradio TTS 且配置有效，则使用 Gradio TTS -----
        if self.use_gradio_tts and self.gradio_server_url and self.gradio_prompt_file:
            # 概率检查
            rand_float = random.random()
            if rand_float > self.gradio_tts_probability:

               # logger.info(f"Qwen3-tts 随机值 {rand_int} > 设定概率 {self.gradio_tts_probability}，跳过 TTS")
                return segment

            new_segment = []
            for comp in segment:
                if isinstance(comp, Plain) and len(comp.text) > 1:
                    try:
                        logger.info(f"[Qwen3-TTS] 请求: {comp.text[:50]}...")
                        audio_path = await self.generate_tts(comp.text)
                        if audio_path:
                            # 替换为语音组件
                            new_segment.append(Record(file=audio_path, url=audio_path))
                            # 是否同时保留文本？可根据需要配置
                            # if dual_output: new_segment.append(comp)
                        else:
                            logger.warning(f"[Qwen3-TTS] 生成失败，使用原文本")
                            new_segment.append(comp)
                    except Exception as e:
                        logger.error(f"[Qwen3-TTS] 处理失败: {e}，使用原文本")
                        new_segment.append(comp)
                else:
                    new_segment.append(comp)
            return new_segment

        # ----- 否则回退到框架 TTS（原有逻辑）-----
        else:
            # 获取框架TTS配置
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
                    tts_trigger_probability = max(0.0, min(float(tts_trigger_probability), 1.0))
                except (TypeError, ValueError):
                    tts_trigger_probability = 1.0

                if random.random() > tts_trigger_probability:
                    logger.info(f"框架 TTS 概率 {random.random():.2f} > 设定概率 {tts_trigger_probability:.2f}，跳过 TTS")
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
        strategy = self.config.get("split_char", {}).get("delay_strategy", "按字数")
        if strategy == "随机":
            return random.uniform(self.config.get("split_char", {}).get("random_control", {}).get("random_min", 1.0), self.config.get("split_char", {}).get("random_control", {}).get("random_max", 3.0))

        elif strategy == "按字数":
            return self.config.get("split_char", {}).get("linear_control", {}).get("linear_base", 0.5) + (len(text) * self.config.get("split_char", {}).get("linear_control", {}).get("linear_factor", 0.1))
        else:
            return self.config.get("split_char", {}).get("fixed_control", {}).get("fixed_delay", 1.5)

    def split_chain(self, chain: List[BaseMessageComponent], pattern: str, do_split: bool, strategies: Dict[str, str], enable_reply: bool) -> List[List[BaseMessageComponent]]:
        segments = []
        current_chain_buffer = []

        for component in chain:
            if isinstance(component, Plain):
                text = component.text
                if not text: continue

                self._process_text_smart(text, pattern, segments, current_chain_buffer)
            else:
                c_type = type(component).__name__.lower()
                if 'reply' in c_type:
                    if enable_reply: current_chain_buffer.append(component)
                    continue

                if 'image' in c_type: strategy = strategies['image']
                elif 'at' in c_type: strategy = strategies['at']
                elif 'face' in c_type: strategy = strategies['face']
                else: strategy = strategies['default']

                if strategy == "单独":
                    if current_chain_buffer:
                        segments.append(current_chain_buffer[:])
                        current_chain_buffer.clear()
                    segments.append([component])
                elif strategy == "跟随上段":
                    if current_chain_buffer: current_chain_buffer.append(component)
                    elif segments: segments[-1].append(component)
                    else: current_chain_buffer.append(component)
                else:
                    current_chain_buffer.append(component)

        if current_chain_buffer:
            segments.append(current_chain_buffer)
        return [seg for seg in segments if seg]

    def _process_text_simple(self, text: str, pattern: str, segments: list, buffer: list):
        parts = re.split(f"({pattern})", text)
        temp_text = ""
        for part in parts:
            if not part: continue
            if re.fullmatch(pattern, part):
                temp_text += part
                buffer.append(Plain(temp_text))
                segments.append(buffer[:])
                buffer.clear()
                temp_text = ""
            else:
                if temp_text: buffer.append(Plain(temp_text)); temp_text = ""
                buffer.append(Plain(part))
        if temp_text: buffer.append(Plain(temp_text))

    def _process_text_smart(self, text: str, pattern: str, segments: list, buffer: list):
        stack = []
        compiled_pattern = re.compile(pattern)
        i = 0
        n = len(text)
        current_chunk = ""

        while i < n:
            char = text[i]
            is_opener = char in self.pair_map
            if char in self.quote_chars:
                if stack and stack[-1] == char: stack.pop()
                else: stack.append(char)
                current_chunk += char; i += 1; continue
            if stack:
                expected_closer = self.pair_map.get(stack[-1])
                if char == expected_closer: stack.pop()
                elif is_opener: stack.append(char)
                current_chunk += char; i += 1; continue
            if is_opener:
                stack.append(char); current_chunk += char; i += 1; continue

            match = compiled_pattern.match(text, pos=i)
            if match:
                delimiter = match.group()
                # 检查是否应使用概率切分
                if self.enable_probabilistic_split and any(ch in self.probabilistic_split_chars for ch in delimiter):
                    if random.random() < self.split_probability:
                       # logger.info("→ 切分")
                        # 按概率切分
                        current_chunk += delimiter
                        buffer.append(Plain(current_chunk))
                        segments.append(buffer[:])
                        buffer.clear()
                        current_chunk = ""
                    else:
                        # 不切分，继续累积
                        current_chunk += delimiter
                else:
                    # 始终切分（原逻辑）
                    current_chunk += delimiter
                    buffer.append(Plain(current_chunk))
                    segments.append(buffer[:])
                    buffer.clear()
                    current_chunk = ""
                i += len(delimiter)
            else:
                current_chunk += char; i += 1
        if current_chunk: buffer.append(Plain(current_chunk))