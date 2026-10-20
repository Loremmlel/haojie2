"""服务器生命周期复用控制器、进程锁和监控；远端仅承载恢复归档。"""

import threading
import time
from pathlib import Path

from ..console.config.resources import hardware, monitor, server_defaults
from ..console.controller import Controller
from ..console.storage import Lease, Store
from .backup import Backups, BackupWorker, import_root


def wait_for(controller, predicate, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if controller.state == "error":
            raise RuntimeError(controller.error)
        if predicate():
            return
        time.sleep(0.05)
    status = controller.status()
    details = {
        key: status[key]
        for key in ("state", "updates", "counts", "runtime", "performance", "logs")
    }
    details["logs"] = details["logs"][-5:]
    raise TimeoutError(f"训练控制器未在期限内完成操作；已保留最近恢复点。当前状态：{details}")


class Session:
    """一个本地训练根只允许一个服务，恢复成功后才接受新训练配置。

    新实验采用服务器初始配置；恢复实验只应用显式覆盖。调用者负责最终 close。
    """

    def __init__(
        self,
        engine,
        root,
        *,
        device="auto",
        settings=None,
        backup_dir=None,
        backup_seconds=600,
        backup_keep=3,
        restore=None,
        import_weights=None,
        scratch=None,
        tiny=False,
        starts=None,
    ):
        root = Path(root).absolute()
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("设置文件必须是 JSON 字段对象")
        if not Path(engine).is_file():
            raise ValueError("原生引擎不存在；请先构建或指定 --engine")
        if restore and import_weights:
            raise ValueError("完整恢复与仅导入权重不能同时选择")
        self.stop = threading.Event()
        self.controller = self.lease = self.backup = None
        self.restoration = None
        self.messages = []
        if backup_dir:
            destination = Path(backup_dir).resolve()
            if (
                destination == root.resolve()
                or destination in root.resolve().parents
                or root.resolve() in destination.parents
            ):
                raise ValueError("活跃训练根和备份目录必须互不包含；训练根应位于本地磁盘")
        self.backups = Backups(backup_dir, backup_keep) if backup_dir else None
        nonempty = root.exists() and any(root.iterdir())
        if restore:
            if nonempty:
                raise ValueError("恢复目标已有文件；请使用新的本地训练根，原目录保持原样")
            source = Path(restore).resolve()
            if source.is_file() and source.suffix == ".json":
                backups = Backups(source.parent, backup_keep, create=False)
                self.restoration = backups.restore(root, selected=source, log=self.messages.append)
                if self.restoration is None:
                    raise ValueError("所选文件不是本工具的完成备份清单")
            elif source.is_dir() and (source / "haojie-backup-store.json").exists():
                self.restoration = Backups(source, backup_keep, create=False).restore(
                    root, log=self.messages.append
                )
                if self.restoration is None:
                    raise ValueError("所选目录尚无完成备份")
            else:
                self.restoration = import_root(source, root)
        elif not nonempty and self.backups and not import_weights:
            self.restoration = self.backups.restore(root, log=self.messages.append)
        if root.exists() and any(root.iterdir()) and not (root / "service.lock").exists():
            raise ValueError("非空目录不是受管训练根；请使用 --restore 导入至新的本地目录")
        try:
            store = Store(root)
            self.lease = Lease(store)
            if import_weights and any(row["recoveries"] for row in store.experiments()):
                raise ValueError("已有恢复点不能被导入权重覆盖；请指定新的本地训练根")
            self.controller = Controller(
                str(Path(engine).resolve()),
                store,
                device=device,
                tiny=tiny,
                imported=Path(import_weights).resolve() if import_weights else None,
                starts=starts,
            )
            controller = self.controller
            if controller.store.names("recovery"):
                controller.command("restore")
                wait_for(controller, lambda: controller.state == "paused")
            else:
                controller.command(
                    "settings", {**server_defaults(str(controller.device)), **(settings or {})}
                )
                # 保存同时初始化模型；权重导入仍走控制器的兼容检查。
                controller.command("save")
                wait_for(
                    controller,
                    lambda: controller.saved_at is not None and controller.state == "paused",
                )
            if settings:
                controller.command("settings", settings)
            for message in self.messages:
                controller.log(message)
            if self.restoration:
                controller.log(f"已导入恢复状态：{self.restoration}")
            self.hardware = hardware()
            threading.Thread(
                target=monitor, args=(controller,), name="haojie-monitor", daemon=True
            ).start()
            if self.backups:
                self.backup = BackupWorker(
                    controller, self.backups, seconds=backup_seconds, scratch=scratch
                )
                self.backup.request()
        except BaseException:
            self.close(save=False)
            raise

    def status(self):
        return {
            **self.controller.status(),
            "hardware": self.hardware,
            "backup": self.backup.status() if self.backup else None,
            "restoration": self.restoration,
        }

    def command(self, action, values=None):
        if action == "backup":
            if self.backup is None:
                raise ValueError("请在启动时设置 --backup-dir")
            self.backup.request()
        elif action == "shutdown":
            self.stop.set()
        else:
            self.controller.command(action, values)

    def close(self, *, save=True, seconds=120):
        """正常退出先暂停并等待最终备份；写线程未结束时保持根目录锁。"""
        errors = []
        controller = self.controller
        if controller is not None:
            try:
                controller.command("pause")
                if save and controller.trainer is not None:
                    wait_for(controller, lambda: controller.state == "paused", seconds)
                    if self.backup:
                        self.backup.request().result(timeout=seconds)
                    else:
                        controller.snapshot(lambda _: None).result(timeout=seconds)
            except Exception as error:
                controller.log(f"退出保存未完成：{error}；此前恢复点保留")
                errors.append(str(error))
            try:
                controller.close()
            except Exception as error:
                errors.append(str(error))
        if self.backup is not None:
            try:
                self.backup.close()
                self.backup = None
            except Exception as error:
                errors.append(str(error))
        stopped = controller is None or not controller.thread.is_alive()
        if stopped and self.backup is None:
            if self.lease is not None:
                self.lease.close()
                self.lease = None
            self.controller = None
        return errors
