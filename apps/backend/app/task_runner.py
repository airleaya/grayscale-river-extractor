"""
In-memory asynchronous task runner for the river-extraction MVP.

The runner is intentionally simple, but it already enforces the most important
system boundary: the API creates and observes tasks, while the pipeline does
work behind a progress-reporting interface.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
from time import monotonic, sleep

from .logging_utils import get_logger
from .models import (
    ArtifactRecord,
    PipelineStage,
    RiverTaskRecord,
    RiverTaskRequest,
    RiverTaskSnapshot,
    TaskStatus,
    utc_now,
)
from .pipeline import ProgressReporter, RiverPipeline, prepare_initial_result

app_logger = get_logger("river.task_runner")
task_logger = get_logger("river.tasks")


class InMemoryProgressReporter(ProgressReporter):
    """
    Thread-safe progress bridge between the pipeline and the task record.

    It centralizes all progress mutation in one small object so stage
    implementations never reach into the task store directly.
    """

    def __init__(self, task_runner: "InMemoryTaskRunner", task_id: str) -> None:
        self._task_runner = task_runner
        self._task_id = task_id
        self._stage_started_at = monotonic()
        self._last_heartbeat_at = 0.0

    def begin_stage(self, stage: PipelineStage, total_units: int, message: str) -> None:
        self._stage_started_at = monotonic()
        self._task_runner._update_progress(
            task_id=self._task_id,
            stage=stage,
            processed_units=0,
            total_units=total_units,
            message=message,
        )

    def advance(self, units: int = 1, message: str = "") -> None:
        self._task_runner._advance_progress(self._task_id, units, message, self._stage_started_at)

    def complete_stage(self, message: str = "") -> None:
        self._task_runner._complete_stage(self._task_id, message)

    def log(self, message: str) -> None:
        self._task_runner._append_log(self._task_id, message)

    def publish_artifact(self, artifact: ArtifactRecord) -> None:
        self._task_runner._publish_artifact(self._task_id, artifact)

    def heartbeat(self, message: str = "", force: bool = False) -> None:
        now = monotonic()
        if not force and now - self._last_heartbeat_at < 0.75:
            return

        self._last_heartbeat_at = now
        self._task_runner._heartbeat(self._task_id, message)

    def is_canceled(self) -> bool:
        return self._task_runner._is_canceled(self._task_id)


class InMemoryTaskRunner:
    """
    Minimal task runner used by the backend API.

    This implementation is intentionally in-memory for fast iteration, while
    still preserving the lifecycle and progress semantics needed for long jobs.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, RiverTaskRecord] = {}
        self._cancel_events: dict[str, Event] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="river-task")
        self._pipeline = RiverPipeline()

    def create_task(self, request: RiverTaskRequest) -> str:
        """Create a new task record and schedule background execution."""

        task = RiverTaskRecord(request=request)
        with self._lock:
            self._tasks[task.task_id] = task
            self._cancel_events[task.task_id] = Event()

            queued_message = "Task created and queued. Waiting for an available worker."
            task.progress.message = queued_message
            task.recent_logs.append(queued_message)
            task.updated_at = utc_now()
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = queued_message

        task_logger.info(
            "task_id=%s | created | input=%s | output=%s",
            task.task_id,
            request.input_path,
            request.output_path,
        )
        self._executor.submit(self._run_task, task.task_id)
        Thread(
            target=self._monitor_queued_task,
            args=(task.task_id,),
            daemon=True,
            name=f"river-queue-monitor-{task.task_id[:8]}",
        ).start()
        return task.task_id

    def get_task(self, task_id: str) -> RiverTaskSnapshot | None:
        """Return a stable snapshot for API responses."""

        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None

            return RiverTaskSnapshot(
                task_id=record.task_id,
                status=record.status,
                progress=record.progress.model_copy(deep=True),
                created_at=record.created_at,
                updated_at=record.updated_at,
                result=record.result.model_copy(deep=True) if record.result else None,
                error=record.error,
                recent_logs=list(record.recent_logs),
            )

    def cancel_task(self, task_id: str) -> RiverTaskSnapshot | None:
        """Mark a task for cancellation and return the latest snapshot."""

        with self._lock:
            record = self._tasks.get(task_id)
            cancel_event = self._cancel_events.get(task_id)
            if record is None or cancel_event is None:
                return None

            cancel_event.set()
            if record.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                record.status = TaskStatus.CANCELED
                record.updated_at = utc_now()
                record.recent_logs.append("Cancellation requested.")
                record.recent_logs = record.recent_logs[-12:]
                record.progress.last_heartbeat_at = record.updated_at
                record.progress.last_heartbeat_message = "Cancellation requested."

        task_logger.warning("task_id=%s | cancel_requested", task_id)
        return self.get_task(task_id)

    def _run_task(self, task_id: str) -> None:
        """Execute the pipeline for one task inside the background worker."""

        reporter = InMemoryProgressReporter(self, task_id)
        with self._lock:
            task = self._require_task_unlocked(task_id)
            task.status = TaskStatus.RUNNING
            task.result = prepare_initial_result(task_id, task.request)
            task.updated_at = utc_now()
            task.recent_logs.append("Task started.")
            task.recent_logs = task.recent_logs[-12:]
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = "Task started."

        task_logger.info("task_id=%s | started", task_id)
        try:
            request = self._get_request(task_id)
            result = self._pipeline.run(task_id, request, reporter)
            if self._is_canceled(task_id):
                task_logger.warning("task_id=%s | stopped_after_cancel", task_id)
                return

            with self._lock:
                task = self._require_task_unlocked(task_id)
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.updated_at = utc_now()
                task.recent_logs.append("Task completed successfully.")
                task.recent_logs = task.recent_logs[-12:]
                task.progress.last_heartbeat_at = task.updated_at
                task.progress.last_heartbeat_message = "Task completed successfully."
            task_logger.info("task_id=%s | completed", task_id)
        except Exception as exc:
            with self._lock:
                task = self._require_task_unlocked(task_id)
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                task.updated_at = utc_now()
                task.recent_logs.append(f"Task failed: {exc}")
                task.recent_logs = task.recent_logs[-12:]
                task.progress.last_heartbeat_at = task.updated_at
                task.progress.last_heartbeat_message = f"Task failed: {exc}"
            task_logger.exception("task_id=%s | failed", task_id)

    def _monitor_queued_task(self, task_id: str) -> None:
        """Emit queue-state heartbeats until the task starts running."""

        while True:
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None:
                    return
                if task.status != TaskStatus.QUEUED:
                    return

                queued_tasks = sorted(
                    (item for item in self._tasks.values() if item.status == TaskStatus.QUEUED),
                    key=lambda item: (item.created_at, item.task_id),
                )
                queued_ahead = 0
                for queue_index, queued_task in enumerate(queued_tasks):
                    if queued_task.task_id == task_id:
                        queued_ahead = queue_index
                        break

                running_tasks = sum(1 for item in self._tasks.values() if item.status == TaskStatus.RUNNING)
                queued_message = (
                    "Task queued: "
                    f"{queued_ahead} ahead, {running_tasks} running, waiting for an available worker."
                )

            self._heartbeat(task_id, queued_message)
            sleep(3.0)

    def _publish_artifact(self, task_id: str, artifact: ArtifactRecord) -> None:
        """Attach one produced artifact to the task snapshot as soon as it exists."""

        with self._lock:
            task = self._require_task_unlocked(task_id)
            if task.result is None:
                return

            task.result.artifacts[artifact.key] = artifact
            task.result.input_preview = task.result.artifacts["input_preview"].preview_path
            task.result.terrain_preprocessed = task.result.artifacts["terrain_preprocessed"].preview_path
            task.result.flow_direction = task.result.artifacts["flow_direction"].preview_path
            task.result.flow_accumulation = task.result.artifacts["flow_accumulation"].preview_path
            task.result.channel_mask = task.result.artifacts["channel_mask"].preview_path
            task.updated_at = utc_now()
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = f"Artifact ready: {artifact.label}"

        task_logger.info(
            "task_id=%s | artifact=%s | status=%s | preview=%s",
            task_id,
            artifact.key,
            artifact.status.value,
            artifact.preview_path,
        )

    def _get_request(self, task_id: str) -> RiverTaskRequest:
        """Fetch the stored request without leaking task-store details."""

        with self._lock:
            task = self._require_task_unlocked(task_id)
            return task.request

    def _append_log(self, task_id: str, message: str) -> None:
        """Append a bounded log line to keep the snapshot light."""

        with self._lock:
            task = self._require_task_unlocked(task_id)
            task.recent_logs.append(message)
            task.recent_logs = task.recent_logs[-12:]
            task.updated_at = utc_now()
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = message

        task_logger.info("task_id=%s | event=%s", task_id, message)

    def _update_progress(
        self,
        task_id: str,
        stage: PipelineStage,
        processed_units: int,
        total_units: int,
        message: str,
    ) -> None:
        """Replace the current stage progress with a fresh progress record."""

        with self._lock:
            task = self._require_task_unlocked(task_id)
            task.progress.stage = stage
            task.progress.processed_units = processed_units
            task.progress.total_units = max(1, total_units)
            task.progress.percent = 0.0
            task.progress.message = message
            task.progress.eta_seconds = None
            task.updated_at = utc_now()
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = message

        app_logger.info(
            "task_id=%s | stage=%s | progress_reset total_units=%s | message=%s",
            task_id,
            stage.value,
            max(1, total_units),
            message,
        )

    def _advance_progress(
        self,
        task_id: str,
        units: int,
        message: str,
        stage_started_at: float,
    ) -> None:
        """
        Advance progress and estimate remaining time for the current stage.

        The ETA is intentionally simple. The main goal is to give the UI enough
        movement and context that long jobs do not look frozen.
        """

        with self._lock:
            task = self._require_task_unlocked(task_id)
            task.progress.processed_units = min(
                task.progress.total_units,
                task.progress.processed_units + max(0, units),
            )

            processed = task.progress.processed_units
            total = max(1, task.progress.total_units)
            task.progress.percent = round((processed / total) * 100, 2)
            task.progress.message = message or task.progress.message

            elapsed = max(monotonic() - stage_started_at, 0.001)
            if processed > 0 and processed < total:
                seconds_per_unit = elapsed / processed
                remaining_units = total - processed
                task.progress.eta_seconds = round(seconds_per_unit * remaining_units, 2)
            else:
                task.progress.eta_seconds = 0.0

            task.updated_at = utc_now()
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = task.progress.message

        if processed == total or processed == 1 or processed % max(1, total // 4) == 0:
            app_logger.info(
                "task_id=%s | stage=%s | progress=%s/%s | percent=%.2f | eta=%s",
                task_id,
                task.progress.stage.value,
                processed,
                total,
                task.progress.percent,
                task.progress.eta_seconds,
            )

    def _complete_stage(self, task_id: str, message: str) -> None:
        """Mark the active stage as fully processed."""

        with self._lock:
            task = self._require_task_unlocked(task_id)
            task.progress.processed_units = task.progress.total_units
            task.progress.percent = 100.0
            task.progress.message = message or task.progress.message
            task.progress.eta_seconds = 0.0
            task.updated_at = utc_now()
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = task.progress.message

        app_logger.info(
            "task_id=%s | stage=%s | completed | message=%s",
            task_id,
            task.progress.stage.value,
            message,
        )

    def _heartbeat(self, task_id: str, message: str) -> None:
        """Refresh liveness even when the current stage cannot advance units yet."""

        with self._lock:
            task = self._require_task_unlocked(task_id)
            heartbeat_message = message or task.progress.message
            timestamp = utc_now()
            task.updated_at = timestamp
            task.progress.last_heartbeat_at = timestamp
            task.progress.last_heartbeat_message = heartbeat_message
            if heartbeat_message:
                task.progress.message = heartbeat_message

    def _is_canceled(self, task_id: str) -> bool:
        """Return whether a cancel event was raised for the task."""

        with self._lock:
            cancel_event = self._cancel_events.get(task_id)
            return bool(cancel_event and cancel_event.is_set())

    def _require_task_unlocked(self, task_id: str) -> RiverTaskRecord:
        """
        Fetch a task record while the caller already owns the lock.

        The split helper avoids nested lock acquisition and keeps the rest of the
        task-runner methods shallow.
        """

        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} does not exist.")

        return task
