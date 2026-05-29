###############################################################################
#  Avatar 任务管理器
#  - 内存缓存 + JSON 文件持久化
#  - 任务超时检查
#  - 用户单任务限制
###############################################################################

import json
import uuid
import threading
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from utils.logger import logger

# 配置
TASK_FILE = Path("data/avatar_tasks.json")
MAX_PROCESSING_TIME = 300  # 5分钟处理超时
MAX_PENDING_TIME = 3600    # 1小时排队超时


class TaskManager:
    """
    Avatar 任务管理器（单例模式）

    功能：
    - 任务创建、查询、状态更新
    - 内存缓存 + JSON 文件持久化
    - 任务超时检查
    - 用户单任务限制（同一用户只允许一个活跃任务）
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
            logger.info(f"TaskManager singleton created: id={id(cls._instance)}")
        else:
            logger.info(f"TaskManager singleton reused: id={id(cls._instance)}")
        return cls._instance

    def _init(self):
        """初始化任务管理器"""
        self.tasks: dict = {}         # task_id -> task（内存缓存）
        self.queue: list = []         # pending 任务队列（task_id列表）
        self.lock = threading.Lock()  # 线程锁
        self.current_task: Optional[dict] = None  # 当前执行的任务

        # 确保 data 目录存在
        TASK_FILE.parent.mkdir(parents=True, exist_ok=True)

        # 启动时加载未完成的任务
        self._load_tasks()
        logger.info(f"TaskManager initialized, loaded {len(self.tasks)} tasks, {len(self.queue)} pending")

    def _load_tasks(self):
        """启动时从 JSON 文件加载未完成的任务"""
        if TASK_FILE.exists():
            try:
                with open(TASK_FILE) as f:
                    tasks = json.load(f)
                    for t in tasks:
                        # 清理超时和失败的任务（不恢复）
                        if t["status"] in ["pending", "processing"]:
                            # 检查是否超时
                            self._check_timeout(t)
                            if t["status"] == "failed":
                                logger.info(f"Cleaned up timeout task: {t['task_id']}")
                                continue

                        self.tasks[t["task_id"]] = t
                        # 恢复 pending 任务到队列
                        if t["status"] == "pending":
                            self.queue.append(t["task_id"])
            except Exception as e:
                logger.exception(f"Failed to load tasks from {TASK_FILE}: {e}")

    def _save_tasks(self):
        """状态变更时保存到 JSON 文件"""
        with self.lock:
            try:
                with open(TASK_FILE, "w") as f:
                    json.dump(list(self.tasks.values()), f, indent=2, default=str)
            except Exception as e:
                logger.exception(f"Failed to save tasks to {TASK_FILE}: {e}")

    def create_task(self, user_id: int, video_path: str) -> dict:
        """
        创建新任务

        Args:
            user_id: 用户ID
            video_path: 临时视频文件路径

        Returns:
            task: 新创建的任务
        """
        task_id = str(uuid.uuid4())
        avatar_id = f"wav2lip256_avatar_{user_id}"

        task = {
            "task_id": task_id,
            "user_id": user_id,
            "avatar_id": avatar_id,
            "video_path": video_path,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }

        with self.lock:
            self.tasks[task_id] = task
            self.queue.append(task_id)

        self._save_tasks()
        logger.info(f"Task created: task_id={task_id}, user_id={user_id}, avatar_id={avatar_id}")

        return task

    def get_user_active_task(self, user_id: int) -> Optional[dict]:
        """
        查询用户活跃任务（用于单任务限制）

        Args:
            user_id: 用户ID

        Returns:
            task: 活跃任务（pending/processing），或 None
        """
        for t in self.tasks.values():
            if t["user_id"] == user_id and t["status"] in ["pending", "processing"]:
                return t
        return None

    def get_task(self, task_id: str) -> Optional[dict]:
        """
        查询任务状态

        Args:
            task_id: 任务ID

        Returns:
            task: 任务信息，或 None
        """
        return self.tasks.get(task_id)

    def get_user_latest_task(self, user_id: int) -> Optional[dict]:
        """
        查询用户最近的任务（包括已完成的）

        Args:
            user_id: 用户ID

        Returns:
            task: 最近的任务，或 None
        """
        user_tasks = [t for t in self.tasks.values() if t["user_id"] == user_id]
        if not user_tasks:
            return None

        # 按创建时间排序，返回最新的
        user_tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return user_tasks[0]

    def get_next_task(self) -> Optional[dict]:
        """
        获取下一个待处理任务（含超时检查）

        流程：
        1. 检查当前任务是否超时
        2. 如果当前任务超时，标记为 failed
        3. 从队列取出下一个 pending 任务
        4. 检查队列任务是否超时
        5. 返回有效任务或 None

        Returns:
            task: 待处理任务（status=processing），或 None
        """
        # 调试：入口日志
        logger.info(f"get_next_task called: queue_len={len(self.queue)}, current_task={self.current_task is not None}")

        with self.lock:
            # 1. 检查当前任务超时
            if self.current_task:
                logger.info(f"Checking current_task timeout: task_id={self.current_task.get('task_id')}")
                self._check_timeout(self.current_task)
                if self.current_task["status"] == "failed":
                    self._save_tasks()
                    self.current_task = None

            # 2. 当前无执行任务时，取出队列头部
            logger.info(f"After timeout check: current_task={self.current_task is not None}, queue_len={len(self.queue)}")
            if self.current_task is None and self.queue:
                # 检查队列任务超时
                while self.queue:
                    task_id = self.queue[0]
                    task = self.tasks.get(task_id)

                    logger.info(f"Checking queue task: task_id={task_id}, task_exists={task is not None}, task_status={task.get('status') if task else 'N/A'}")

                    if task is None:
                        # 任务不存在，移除
                        logger.warning(f"Task {task_id} not in tasks dict, removing from queue")
                        self.queue.pop(0)
                        continue

                    self._check_timeout(task)

                    logger.info(f"After _check_timeout: task_id={task_id}, status={task['status']}")

                    if task["status"] == "failed":
                        # 任务超时，移除并保存
                        logger.info(f"Task {task_id} timeout/failed, removing from queue")
                        self.queue.pop(0)
                        self._save_tasks()
                        continue

                    logger.info(f"Task {task_id} is valid, starting processing...")
                    # 取出有效任务
                    self.queue.pop(0)
                    task["status"] = "processing"
                    task["processing_start_time"] = datetime.now().isoformat()
                    self.current_task = task
                    self._save_tasks()
                    logger.info(f"Task started processing: task_id={task_id}")
                    return task

        logger.info(f"get_next_task returning None")
        return None

    def _check_timeout(self, task: dict):
        """
        检查任务超时

        Args:
            task: 任务信息
        """
        logger.info(f"_check_timeout called: task_id={task['task_id']}, status={task['status']}")
        now = datetime.now()
        logger.info(f"now={now.isoformat()}")

        # 处理超时
        if task["status"] == "processing":
            logger.info(f"Checking processing timeout")
            if "processing_start_time" in task:
                start = datetime.fromisoformat(task["processing_start_time"])
                elapsed = (now - start).total_seconds()
                logger.info(f"processing elapsed={elapsed}s, max={MAX_PROCESSING_TIME}s")
                if elapsed > MAX_PROCESSING_TIME:
                    task["status"] = "failed"
                    task["message"] = "生成超时，请重试"
                    logger.warning(f"Task processing timeout: task_id={task['task_id']}")

        # 排队超时
        elif task["status"] == "pending":
            logger.info(f"Checking pending timeout")
            if "created_at" in task:
                created = datetime.fromisoformat(task["created_at"])
                elapsed = (now - created).total_seconds()
                logger.info(f"pending elapsed={elapsed}s, max={MAX_PENDING_TIME}s, created={created.isoformat()}")
                if elapsed > MAX_PENDING_TIME:
                    task["status"] = "failed"
                    task["message"] = "排队超时，请重试"
                    logger.warning(f"Task pending timeout: task_id={task['task_id']}")

        logger.info(f"_check_timeout done: task_id={task['task_id']}, status={task['status']}")

    def update_status(self, task_id: str, status: str, **kwargs):
        """
        更新任务状态

        Args:
            task_id: 任务ID
            status: 新状态（success/failed）
            **kwargs: 其他字段（avatar_id, message 等）
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                logger.warning(f"Task not found: task_id={task_id}")
                return

            task["status"] = status

            for k, v in kwargs.items():
                task[k] = v

            if status in ["success", "failed"]:
                task["finished_at"] = datetime.now().isoformat()

                # 清除当前任务
                if self.current_task and self.current_task.get("task_id") == task_id:
                    self.current_task = None

        self._save_tasks()
        logger.info(f"Task status updated: task_id={task_id}, status={status}")


# 全局单例
task_manager = TaskManager()