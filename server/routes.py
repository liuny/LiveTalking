###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import json
import numpy as np
import asyncio
import os
import cv2
from aiohttp import web
from datetime import datetime

from utils.logger import logger


# ─── 路由工具函数 ──────────────────────────────────────────────────────────

def json_ok(data=None):
    """返回成功 JSON 响应"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """返回错误 JSON 响应"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager

def get_session(request, sessionid: str):
    """从 app 中获取 session 实例"""
    return session_manager.get_session(sessionid)


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')

        if params['type'] == 'echo':
            avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            # custom_llm_tts 模式：直接将用户消息发给 TTS，绕过 LiveTalking 的 LLM
            if hasattr(avatar_session, 'opt') and avatar_session.opt.tts == 'custom_llm_tts':
                logger.info(f"custom_llm_tts 模式: 直接发送用户消息 '{params['text']}'")
                avatar_session.put_msg_txt(params['text'], datainfo)
            else:
                # 其他 TTS 模式：使用 LiveTalking 的 LLM 处理
                llm_response = request.app.get("llm_response")
                if llm_response:
                    asyncio.get_event_loop().run_in_executor(
                        None, llm_response, params['text'], avatar_session, datainfo
                    )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """上传音频文件"""
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """设置自定义状态（动作编排）"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """录制控制"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def is_speaking(request):
    """查询是否正在说话"""
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())


async def websocket_text(request):
    """WebSocket 路由：推送数字人文字回复"""
    sessionid = request.match_info.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # 保存 websocket 连接到 session
    avatar_session.websocket = ws
    logger.info(f"WebSocket connected for session {sessionid}")

    async def send_task():
        """发送任务：不断检查队列并发送"""
        while not ws.closed:
            try:
                msg = avatar_session.text_queue.get_nowait()
                await ws.send_json(msg)
                logger.info(f"WebSocket sent: {msg}")
            except:
                pass
            await asyncio.sleep(0.1)

    async def receive_task():
        """接收任务：等待客户端消息"""
        while not ws.closed:
            msg = await ws.receive()
            if msg.type == web.WSMsgType.CLOSE:
                logger.info(f"WebSocket closed by client")
                break
            elif msg.type == web.WSMsgType.ERROR:
                logger.warning(f"WebSocket error: {ws.exception()}")
                break

    try:
        # 并行运行发送和接收任务
        await asyncio.gather(send_task(), receive_task())
    except Exception as e:
        logger.info(f"WebSocket exception: {e}")
    finally:
        avatar_session.websocket = None
        await ws.close()

    return ws


# ─── 路由注册 ──────────────────────────────────────────────────────────────

def setup_routes(app):
    """注册所有路由到 aiohttp app"""
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_get("/ws/text/{sessionid}", websocket_text)
    app.router.add_static('/', path='web')

    # Avatar 自定义形象接口
    app.router.add_post("/api/avatar/upload", avatar_upload)
    app.router.add_get("/api/avatar/status", avatar_status)
    app.router.add_get("/api/avatar/image/{avatar_id}", avatar_image)


# ─── Avatar 自定义形象接口 ────────────────────────────────────────────────

from server.task_manager import task_manager
from server.rtc_manager import parse_ticket_from_auth_header

TMP_DIR = "data/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# 视频校验配置
MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50MB
MIN_DURATION = 3   # 最短 3 秒
MAX_DURATION = 15  # 最长 15 秒
MIN_WIDTH = 1280   # 720p 宽度
MIN_HEIGHT = 720   # 720p 高度


async def avatar_upload(request):
    """
    上传视频接口

    从请求头获取 uid 作为 user_id
    校验视频格式、时长、大小、分辨率
    创建任务，返回 task_id
    """
    try:
        # 从 Authorization header 获取用户信息
        ticket_info = parse_ticket_from_auth_header(request.headers)
        user_id = int(ticket_info.get('uid', '0'))
        if user_id == 0:
            return json_error("缺少用户认证信息")

        # 获取上传文件
        reader = await request.multipart()
        video_path = None

        async for field in reader:
            if field.name == 'file':
                # 检查文件格式
                filename = field.filename
                if not filename.lower().endswith('.mp4'):
                    return json_error("视频格式必须为 mp4")

                # 保存临时文件
                task_id_temp = datetime.now().strftime("%Y%m%d%H%M%S")
                video_path = os.path.join(TMP_DIR, f"{task_id_temp}_{user_id}.mp4")

                size = 0
                with open(video_path, 'wb') as f:
                    while True:
                        chunk = await field.read_chunk()
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_VIDEO_SIZE:
                            f.close()
                            os.remove(video_path)
                            return json_error(f"视频大小超过 {MAX_VIDEO_SIZE // 1024 // 1024}MB")

                # 校验视频
                valid, message = validate_uploaded_video(video_path)
                if not valid:
                    os.remove(video_path)
                    return json_error(message)

                # 检查用户是否有活跃任务
                active_task = task_manager.get_user_active_task(user_id)
                if active_task:
                    if active_task["status"] == "pending":
                        return json_error("已有任务在排队中，请等待完成后再上传")
                    elif active_task["status"] == "processing":
                        return json_error("已有任务正在生成，请等待完成后再上传")

                # 创建任务
                task = task_manager.create_task(user_id, video_path)

                return json_ok({
                    "task_id": task["task_id"],
                    "avatar_id": task["avatar_id"]
                })

        return json_error("未找到上传文件")

    except Exception as e:
        logger.exception('avatar_upload exception:')
        return json_error(str(e))


def validate_uploaded_video(video_path: str) -> tuple:
    """
    校验上传视频

    检查时长、分辨率

    Args:
        video_path: 视频路径

    Returns:
        (valid, message): 是否有效，以及错误信息
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "视频文件无法打开"

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 计算时长
    if fps > 0:
        duration = frame_count / fps
    else:
        return False, "无法获取视频帧率"

    # 校验时长
    if duration < MIN_DURATION:
        return False, f"视频时长 {duration:.1f} 秒，需要至少 {MIN_DURATION} 秒"
    if duration > MAX_DURATION:
        return False, f"视频时长 {duration:.1f} 秒，最长 {MAX_DURATION} 秒"

    # 校验分辨率（720p：宽>=1280 或 高>=720）
    if width < MIN_WIDTH and height < MIN_HEIGHT:
        return False, f"视频分辨率过低（{width}x{height}），请上传 720p 及以上视频"

    logger.info(f"Video validated: {width}x{height}, {duration:.1f}s")
    return True, "success"


async def avatar_status(request):
    """
    查询任务状态接口

    返回 status + avatar_preview_url（成功时）
    """
    try:
        task_id = request.query.get('task_id', '')
        if not task_id:
            return json_error("缺少 task_id 参数")

        task = task_manager.get_task(task_id)
        if task is None:
            return json_error("任务不存在")

        response = {"status": task["status"]}

        if task["status"] == "success":
            response["avatar_id"] = task["avatar_id"]
            # 构造图片 URL
            host = request.host
            response["avatar_preview_url"] = f"http://{host}/api/avatar/image/{task['avatar_id']}"

        elif task["status"] == "failed":
            response["message"] = task.get("message", "生成失败")

        return json_ok(response)

    except Exception as e:
        logger.exception('avatar_status exception:')
        return json_error(str(e))


async def avatar_image(request):
    """
    获取形象预览图片接口

    返回首帧图片文件流（Content-Type: image/png）
    """
    try:
        avatar_id = request.match_info.get('avatar_id', '')
        img_path = f"./data/avatars/{avatar_id}/full_imgs/00000000.png"

        if not os.path.exists(img_path):
            return json_error("avatar 不存在")

        with open(img_path, 'rb') as f:
            img_data = f.read()

        return web.Response(
            content_type='image/png',
            body=img_data
        )

    except Exception as e:
        logger.exception('avatar_image exception:')
        return json_error(str(e))
