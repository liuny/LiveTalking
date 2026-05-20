import time
import json
import base64
import requests
import numpy as np

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register


@register("tts", "custom_llm_tts")
class CustomLLMTTS(BaseTTS):
    """自定义 LLM+TTS 统一接口适配模块"""

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.server_url = getattr(opt, 'TTS_SERVER', 'http://192.168.4.250:9372')
        self.api_path = "/api/v1/chat/sendchat_with_voice"

        # 票据信息（从请求头提取，存入 opt 配置）
        self.uid = getattr(opt, 'uid', '0')
        self.random = getattr(opt, 'random', '')
        self.expire = getattr(opt, 'expire', '')
        self.ticket = getattr(opt, 'ticket', '')

        # voice_config 固定配置
        self.voice_config = {
            "voice_type": "custom",
            "voice_desc": "年轻女性，声音温柔甜美"
        }

        logger.info(f"CustomLLMTTS initialized: server={self.server_url}, uid={self.uid}")

    def txt_to_audio(self, msg: tuple[str, dict]):
        """将文本转换为音频（调用统一接口）"""
        text, textevent = msg

        t = time.time()
        logger.info(f"开始处理文本: {text[:50]}{'...' if len(text) > 50 else ''}")

        try:
            # 构建请求
            url = f"{self.server_url}{self.api_path}"
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "uid": str(self.uid),
                "random": self.random,
                "expire": self.expire,
                "ticket": self.ticket
            }
            body = {
                "message": text,
                "session_id": self.parent.sessionid,
                "voice_config": self.voice_config
            }

            # 流式请求
            response = requests.post(url, json=body, headers=headers, stream=True, timeout=60)

            if response.status_code != 200:
                logger.error(f"接口返回错误: {response.status_code} - {response.text}")
                return

            # 解析响应
            audio_segments = []
            reply_text = None
            buffer = ''

            for chunk in response.iter_content(chunk_size=None):
                if not chunk:
                    continue

                buffer += chunk.decode('utf-8')

                # 按换行分割 JSON
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        msg_type = data.get("type")

                        if msg_type == "audio_chunk":
                            # base64 解码
                            audio_base64 = data.get("audio_data")
                            if audio_base64:
                                audio_bytes = base64.b64decode(audio_base64)

                                # PCM Int16 → Float32
                                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                                audio_segments.append(audio_float32)

                        elif msg_type == "stream_end":
                            reply_text = data.get("text")
                            duration_ms = data.get("total_duration_ms", 0)
                            logger.info(f"LLM 回复: {reply_text}, 音频时长: {duration_ms}ms")
                            break

                        elif msg_type == "error":
                            error_msg = data.get("message", "未知错误")
                            logger.error(f"接口返回错误: {error_msg}")
                            return

                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 解析失败: {line[:50]} - {e}")
                        continue

            # 合并音频段
            if not audio_segments:
                logger.warning("未收到音频数据")
                return

            audio_array = np.concatenate(audio_segments)
            logger.info(f"音频合并完成: {audio_array.shape[0]} 采样点, "
                       f"时长: {audio_array.shape[0]/self.sample_rate:.2f}s")

            # 分 chunk 推送
            self._push_audio_chunks(audio_array, text, textevent)

            logger.info(f"-------custom_llm_tts time: {time.time()-t:.4f}s")

        except requests.exceptions.RequestException as e:
            logger.exception(f"网络请求异常: {e}")
        except Exception as e:
            logger.exception(f"custom_llm_tts 异常: {e}")

    def _push_audio_chunks(self, audio_array: np.ndarray, text: str, textevent: dict):
        """分 chunk 推送音频给数字人"""
        chunk_size = self.chunk  # 320 samples (20ms)
        idx = 0
        first = True
        streamlen = len(audio_array)

        while streamlen - idx >= chunk_size and self.state == State.RUNNING:
            eventpoint = {}
            streamlen -= chunk_size

            if first:
                eventpoint = {'status': 'start', 'text': text}
                first = False
            elif streamlen < chunk_size:
                eventpoint = {'status': 'end', 'text': text}

            eventpoint.update(**textevent)

            self.parent.put_audio_frame(audio_array[idx:idx+chunk_size], eventpoint)
            idx += chunk_size