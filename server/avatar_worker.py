###############################################################################
#  Avatar 生成队列处理器
#
#  核心流程（全部同步执行，确保时序正确）：
#  1. process_video_resolution() - ffmpeg 缩放视频（同步等待）
#  2. generate_avatar() - 执行 genavatar 脚本（同步等待）
#  3. validate_avatar() - 验证文件完整性（此时文件已完整）
#  4. update_status() - 更新任务状态
#  5. cleanup_tmp_files() - 清理临时文件
###############################################################################

import threading
import time
import os
import subprocess
import cv2
import glob
import pickle

from utils.logger import logger
from server.task_manager import task_manager

# 临时文件目录
TMP_DIR = "data/tmp"
os.makedirs(TMP_DIR, exist_ok=True)


def avatar_worker():
    """
    后台线程：处理 avatar 生成任务

    无限循环，从 TaskManager 获取下一个任务并处理
    """
    logger.info("Avatar worker thread started")

    while True:
        task = task_manager.get_next_task()

        if task is None:
            # 调试：打印队列状态
            queue_len = len(task_manager.queue)
            if queue_len > 0:
                logger.info(f"Queue has {queue_len} tasks but none returned")
            time.sleep(1)  # 无任务时休眠 1 秒
            continue

        task_id = task["task_id"]
        avatar_id = task["avatar_id"]
        video_path = task["video_path"]

        try:
            logger.info(f"Processing task: task_id={task_id}, avatar_id={avatar_id}")

            # ========== 1. 视频分辨率处理（同步执行）==========
            processed_path = process_video_resolution(video_path)

            # ========== 2. 执行 genavatar（同步执行，等待完成）==========
            generate_avatar(processed_path, avatar_id)
            # 此时 genavatar 已完成，coords.pkl 已写入

            # ========== 3. 验证文件完整性（此时文件已完整）==========
            valid, message = validate_avatar(avatar_id)
            if not valid:
                task_manager.update_status(task_id, "failed", message=message)
                cleanup_tmp_files(video_path, processed_path)
                logger.warning(f"Avatar validation failed: {message}")
                continue

            # ========== 4. 更新成功状态 ==========
            task_manager.update_status(task_id, "success")
            logger.info(f"Avatar generated successfully: avatar_id={avatar_id}")

            # ========== 5. 清理临时文件 ==========
            cleanup_tmp_files(video_path, processed_path)

        except Exception as e:
            logger.exception(f"Avatar generation failed: task_id={task_id}")
            task_manager.update_status(task_id, "failed", message=str(e))
            cleanup_tmp_files(video_path, processed_path)


def process_video_resolution(video_path: str) -> str:
    """
    视频分辨率处理：低于 720p 时缩放至 720p

    使用 ffmpeg 同步执行，等待缩放完成

    Args:
        video_path: 原视频路径

    Returns:
        processed_path: 处理后的视频路径（可能和原路径相同）
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    logger.info(f"Video resolution: {width}x{height}")

    # 720p 标准：宽>=1280 或 高>=720
    if width >= 1280 or height >= 720:
        logger.info("Video resolution meets 720p, no processing needed")
        return video_path

    # 需要缩放至 720p
    target_width = 1280
    target_height = 720
    output_path = video_path.replace(".mp4", "_720p.mp4")

    logger.info(f"Scaling video to 720p: {output_path}")

    # 使用 ffmpeg 缩放（同步执行，等待完成）
    result = subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"scale={target_width}:{target_height}",
        "-c:v", "libx264", "-preset", "fast",
        output_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"ffmpeg scaling failed: {result.stderr}")

    logger.info(f"Video scaled successfully: {output_path}")
    return output_path


def generate_avatar(video_path: str, avatar_id: str):
    """
    执行 genavatar 脚本（同步执行，等待完成）

    subprocess.run() 会等待脚本执行完成才返回，
    此时所有文件（full_imgs, face_imgs, coords.pkl）已写入完毕

    Args:
        video_path: 视频路径
        avatar_id: avatar ID
    """
    logger.info(f"Running genavatar: video_path={video_path}, avatar_id={avatar_id}")

    result = subprocess.run([
        "python", "-m", "avatars.wav2lip.genavatar",
        "--video_path", video_path,
        "--avatar_id", avatar_id,
        "--img_size", "256"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"genavatar 执行失败: {result.stderr}")

    logger.info(f"genavatar completed successfully: avatar_id={avatar_id}")


def validate_avatar(avatar_id: str) -> tuple:
    """
    验证 avatar 文件完整性

    此时 genavatar 已同步执行完成，文件一定是完整的，
    不会有"正在写入"的时序问题

    Args:
        avatar_id: avatar ID

    Returns:
        (valid, message): 是否有效，以及错误信息
    """
    avatar_path = f"./data/avatars/{avatar_id}"
    min_frames = 60

    logger.info(f"Validating avatar: avatar_id={avatar_id}")

    # 1. 目录存在
    if not os.path.exists(avatar_path):
        return False, "avatar 目录不存在"

    # 2. 帧数检查
    full_imgs = glob.glob(f"{avatar_path}/full_imgs/*.png")
    face_imgs = glob.glob(f"{avatar_path}/face_imgs/*.png")

    if len(full_imgs) < min_frames:
        return False, f"full_imgs 仅 {len(full_imgs)} 帧，需要 {min_frames} 帧"
    if len(face_imgs) < min_frames:
        return False, f"face_imgs 仅 {len(face_imgs)} 帧，需要 {min_frames} 帧"
    if len(full_imgs) != len(face_imgs):
        return False, "帧数不匹配"

    # 3. coords.pkl 检查
    coords_path = f"{avatar_path}/coords.pkl"
    if not os.path.exists(coords_path):
        return False, "coords.pkl 不存在"

    with open(coords_path, 'rb') as f:
        coords = pickle.load(f)
    if len(coords) != len(full_imgs):
        return False, "coords 数量与帧数不匹配"

    # 4. 首帧可读
    first_img = f"{avatar_path}/full_imgs/00000000.png"
    if not os.path.exists(first_img):
        return False, "首帧图片不存在"
    img = cv2.imread(first_img)
    if img is None:
        return False, "首帧图片损坏"

    logger.info(f"Avatar validation passed: {len(full_imgs)} frames")
    return True, "success"


def cleanup_tmp_files(video_path: str, processed_path: str):
    """
    清理临时视频文件

    Args:
        video_path: 原视频路径
        processed_path: 处理后的视频路径
    """
    try:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"Cleaned up: {video_path}")

        if processed_path != video_path and processed_path and os.path.exists(processed_path):
            os.remove(processed_path)
            logger.info(f"Cleaned up: {processed_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup tmp files: {e}")


def start_avatar_worker():
    """
    启动 avatar 生成队列处理器线程

    在 app.py 的 main() 函数中调用
    """
    worker_thread = threading.Thread(target=avatar_worker, daemon=True)
    worker_thread.start()
    logger.info("Avatar worker thread initialized")