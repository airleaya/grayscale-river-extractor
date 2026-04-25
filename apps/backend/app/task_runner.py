"""
Persistent asynchronous task runner for the river-extraction backend.

The runner keeps task snapshots on disk so the frontend can reconnect after a
page refresh or browser restart, and it supports pause/resume/rerun semantics
at the stage boundary level.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
from time import monotonic, sleep

from .logging_utils import get_logger
from .models import (
    ArtifactRecord,
    ArtifactStatus,
    DraftTaskState,
    ParallelChunkProgress,
    ParallelChunkStatus,
    ParallelWorkProgress,
    PIPELINE_STAGE_SEQUENCE,
    PipelineStage,
    RiverTaskRecord,
    RiverTaskRequest,
    RiverTaskSnapshot,
    TaskProgress,
    TaskStatus,
    default_task_name,
    get_stage_index,
    get_next_stage,
    utc_now,
)
from .pipeline import ProgressReporter, RiverPipeline, prepare_initial_result
from .storage import clear_task_directory, delete_task_directory, load_task_records, save_task_record

app_logger = get_logger("river.task_runner")
task_logger = get_logger("river.tasks")


class TaskPauseRequested(RuntimeError):
    """Raised when a running task reaches a safe point and should pause."""


class InMemoryProgressReporter(ProgressReporter):
    """
    Thread-safe progress bridge between the pipeline and the task record.

    It centralizes progress mutation and pause/cancel checks so stage
    implementations only emit progress through a single surface.
    """

    def __init__(self, task_runner: "InMemoryTaskRunner", task_id: str) -> None:
        self._task_runner = task_runner
        self._task_id = task_id
        self._stage_started_at = monotonic()
        self._last_heartbeat_at = 0.0

    def begin_stage(self, stage: PipelineStage, total_units: int, message: str) -> None:
        self._task_runner._checkpoint_task_control(self._task_id)
        self._stage_started_at = monotonic()
        self._task_runner._update_progress(
            task_id=self._task_id,
            stage=stage,
            processed_units=0,
            total_units=total_units,
            message=message,
        )

    def advance(self, units: int = 1, message: str = "") -> None:
        self._task_runner._checkpoint_task_control(self._task_id)
        self._task_runner._advance_progress(self._task_id, units, message, self._stage_started_at)

    def complete_stage(self, message: str = "") -> None:
        self._task_runner._checkpoint_task_control(self._task_id)
        self._task_runner._complete_stage(self._task_id, message)

    def log(self, message: str) -> None:
        self._task_runner._checkpoint_task_control(self._task_id)
        self._task_runner._append_log(self._task_id, message)

    def publish_artifact(self, artifact: ArtifactRecord) -> None:
        self._task_runner._checkpoint_task_control(self._task_id)
        self._task_runner._publish_artifact(self._task_id, artifact)

    def heartbeat(self, message: str = "", force: bool = False) -> None:
        self._task_runner._checkpoint_task_control(self._task_id)
        now = monotonic()
        if not force and now - self._last_heartbeat_at < 0.75:
            return

        self._last_heartbeat_at = now
        self._task_runner._heartbeat(self._task_id, message)

    def set_parallel_work(
        self,
        label: str,
        strategy: str,
        chunks: list[tuple[str, str, int]],
    ) -> None:
        self._task_runner._checkpoint_task_control(self._task_id)
        self._task_runner._set_parallel_work(self._task_id, label, strategy, chunks)

    def update_parallel_chunk(
        self,
        chunk_id: str,
        status: str,
        processed_units: int,
        total_units: int,
        detail: str = "",
    ) -> None:
        self._task_runner._checkpoint_task_control(self._task_id)
        self._task_runner._update_parallel_chunk(
            self._task_id,
            chunk_id,
            status,
            processed_units,
            total_units,
            detail,
        )

    def clear_parallel_work(self) -> None:
        self._task_runner._checkpoint_task_control(self._task_id)
        self._task_runner._clear_parallel_work(self._task_id)

    def is_canceled(self) -> bool:
        return self._task_runner._is_canceled(self._task_id)


class InMemoryTaskRunner:
    """
    Persistent task runner used by the backend API.

    The runner keeps the authoritative task registry in memory for fast access,
    while persisting task records to disk so the UI can recover them later.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, RiverTaskRecord] = {}
        self._cancel_events: dict[str, Event] = {}
        self._pause_events: dict[str, Event] = {}
        self._active_workers: set[str] = set()
        self._last_persist_at: dict[str, float] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="river-task")
        self._pipeline = RiverPipeline()
        self._load_persisted_tasks()

    def create_task(self, request: RiverTaskRequest) -> str:
        """Create a new task record and schedule background execution."""

        task = RiverTaskRecord(
            name=default_task_name(),
            draft_state=DraftTaskState(
                input_path=request.input_path,
                mask_path=request.mask_path,
                output_path=request.output_path,
                config=request.config.model_copy(deep=True),
                inherit_intermediates=request.inherit_intermediates,
                inherit_stage_outputs=list(request.inherit_stage_outputs or []),
            ),
            request=request,
            status=TaskStatus.QUEUED,
        )
        with self._lock:
            self._tasks[task.task_id] = task
            self._cancel_events[task.task_id] = Event()
            self._pause_events[task.task_id] = Event()

            queued_message = "Task created and queued. Waiting for an available worker."
            task.progress.message = queued_message
            task.recent_logs.append(queued_message)
            task.updated_at = utc_now()
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = queued_message
            self._persist_task_unlocked(task, force=True)

        task_logger.info(
            "task_id=%s | created | input=%s | output=%s",
            task.task_id,
            request.input_path,
            request.output_path,
        )
        self._submit_task(task.task_id)
        return task.task_id

    def create_draft_task(self, name: str | None = None) -> RiverTaskSnapshot:
        """Create one editable task draft without scheduling execution."""

        draft_name = name.strip() if name is not None and name.strip() else default_task_name()
        draft_message = "Task draft created. Configure inputs and stages, then start from the right panel."
        task = RiverTaskRecord(
            name=draft_name,
            draft_state=DraftTaskState(),
            request=None,
            status=TaskStatus.DRAFT,
            progress=TaskProgress(
                stage=PIPELINE_STAGE_SEQUENCE[0],
                total_units=1,
                message=draft_message,
            ),
        )
        task.updated_at = utc_now()
        task.progress.last_heartbeat_at = task.updated_at
        task.progress.last_heartbeat_message = draft_message
        task.recent_logs.append(draft_message)

        with self._lock:
            self._tasks[task.task_id] = task
            self._cancel_events[task.task_id] = Event()
            self._pause_events[task.task_id] = Event()
            self._persist_task_unlocked(task, force=True)

        task_logger.info("task_id=%s | draft_created | name=%s", task.task_id, task.name)
        return self.get_task(task.task_id)  # type: ignore[return-value]

    def update_draft_state(self, task_id: str, draft_state: DraftTaskState) -> RiverTaskSnapshot | None:
        """Persist editable left-panel state for one draft task."""

        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None

            if record.status != TaskStatus.DRAFT:
                return self._build_snapshot(record)

            record.draft_state = draft_state.model_copy(deep=True)
            record.updated_at = utc_now()
            record.progress.last_heartbeat_at = record.updated_at
            record.progress.last_heartbeat_message = "Draft configuration saved."
            if not record.progress.message:
                record.progress.message = "Draft configuration saved."
            self._persist_task_unlocked(record, force=True)
            return self._build_snapshot(record)

    def start_task(self, task_id: str, request: RiverTaskRequest) -> RiverTaskSnapshot | None:
        """Start one draft task with the provided execution request."""

        with self._lock:
            record = self._tasks.get(task_id)
            cancel_event = self._cancel_events.get(task_id)
            pause_event = self._pause_events.get(task_id)
            if record is None or cancel_event is None or pause_event is None:
                return None

            if record.status != TaskStatus.DRAFT:
                return self._build_snapshot(record)

            cancel_event.clear()
            pause_event.clear()
            queued_message = "Task configured and queued. Waiting for an available worker."
            record.request = request
            record.draft_state = DraftTaskState(
                input_path=request.input_path,
                mask_path=request.mask_path,
                output_path=request.output_path,
                config=request.config.model_copy(deep=True),
                inherit_intermediates=request.inherit_intermediates,
                inherit_stage_outputs=list(request.inherit_stage_outputs or []),
            )
            record.status = TaskStatus.QUEUED
            record.progress = TaskProgress(
                stage=PIPELINE_STAGE_SEQUENCE[0],
                total_units=1,
                message=queued_message,
            )
            record.result = None
            record.error = None
            record.last_completed_stage = None
            record.completed_stages = []
            record.updated_at = utc_now()
            record.progress.last_heartbeat_at = record.updated_at
            record.progress.last_heartbeat_message = queued_message
            record.recent_logs.append(queued_message)
            record.recent_logs = record.recent_logs[-24:]
            self._persist_task_unlocked(record, force=True)

        task_logger.info(
            "task_id=%s | draft_started | input=%s | output=%s",
            task_id,
            request.input_path,
            request.output_path,
        )
        self._submit_task(task_id)
        return self.get_task(task_id)

    def rename_task(self, task_id: str, name: str) -> RiverTaskSnapshot | None:
        """Rename one existing task record."""

        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Task name cannot be empty.")

        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None

            rename_message = f"Task renamed to {normalized_name}."
            record.name = normalized_name
            record.updated_at = utc_now()
            record.recent_logs.append(rename_message)
            record.recent_logs = record.recent_logs[-24:]
            record.progress.last_heartbeat_at = record.updated_at
            record.progress.last_heartbeat_message = rename_message
            self._persist_task_unlocked(record, force=True)
            return self._build_snapshot(record)

    def delete_task(self, task_id: str) -> RiverTaskSnapshot | None:
        """Delete one non-running task and its persisted artifacts."""

        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None

            if record.status in {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.PAUSING}:
                raise RuntimeError("Active tasks cannot be deleted while they are still running.")

            snapshot = self._build_snapshot(record)
            self._tasks.pop(task_id, None)
            self._cancel_events.pop(task_id, None)
            self._pause_events.pop(task_id, None)
            self._last_persist_at.pop(task_id, None)

        delete_task_directory(task_id)
        task_logger.info("task_id=%s | deleted", task_id)
        return snapshot

    def list_tasks(self) -> list[RiverTaskSnapshot]:
        """Return stable snapshots for all known tasks."""

        with self._lock:
            ordered_records = sorted(
                self._tasks.values(),
                key=lambda item: (item.updated_at, item.created_at, item.task_id),
                reverse=True,
            )
            return [self._build_snapshot(record) for record in ordered_records]

    def get_task(self, task_id: str) -> RiverTaskSnapshot | None:
        """Return a stable snapshot for API responses."""

        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None

            return self._build_snapshot(record)

    def pause_task(self, task_id: str) -> RiverTaskSnapshot | None:
        """Request pausing for a queued or running task."""

        with self._lock:
            record = self._tasks.get(task_id)
            pause_event = self._pause_events.get(task_id)
            if record is None or pause_event is None:
                return None

            if record.status == TaskStatus.QUEUED:
                record.status = TaskStatus.PAUSED
                pause_message = "Task paused before execution."
            elif record.status == TaskStatus.RUNNING:
                record.status = TaskStatus.PAUSING
                pause_message = "Pause requested. Waiting for the next safe point."
            elif record.status in {TaskStatus.PAUSING, TaskStatus.PAUSED}:
                return self._build_snapshot(record)
            else:
                return self._build_snapshot(record)

            pause_event.set()
            record.updated_at = utc_now()
            record.recent_logs.append(pause_message)
            record.recent_logs = record.recent_logs[-24:]
            record.progress.last_heartbeat_at = record.updated_at
            record.progress.last_heartbeat_message = pause_message
            record.progress.message = pause_message
            self._persist_task_unlocked(record, force=True)

        task_logger.warning("task_id=%s | pause_requested", task_id)
        return self.get_task(task_id)

    def resume_task(self, task_id: str) -> RiverTaskSnapshot | None:
        """Resume a paused task from the next safe stage boundary."""

        with self._lock:
            record = self._tasks.get(task_id)
            pause_event = self._pause_events.get(task_id)
            cancel_event = self._cancel_events.get(task_id)
            if record is None or pause_event is None or cancel_event is None:
                return None

            if record.status != TaskStatus.PAUSED:
                return self._build_snapshot(record)

            pause_event.clear()
            cancel_event.clear()
            resume_message = "Task resumed and re-queued."
            record.status = TaskStatus.QUEUED
            record.error = None
            record.updated_at = utc_now()
            record.recent_logs.append(resume_message)
            record.recent_logs = record.recent_logs[-24:]
            record.progress.last_heartbeat_at = record.updated_at
            record.progress.last_heartbeat_message = resume_message
            record.progress.message = resume_message
            self._persist_task_unlocked(record, force=True)

        task_logger.info("task_id=%s | resumed", task_id)
        self._submit_task(task_id)
        return self.get_task(task_id)

    def continue_task(
        self,
        task_id: str,
        end_stage: PipelineStage | None = None,
        inherit_intermediates: bool | None = None,
        inherit_stage_outputs: list[PipelineStage] | None = None,
    ) -> RiverTaskSnapshot | None:
        """
        Continue one existing task toward a target stage.

        When `inherit_intermediates` is enabled, the runner reuses completed
        stage outputs already present under the task directory and continues
        from the next stage boundary. When it is disabled, the task is reset
        and recomputed from the beginning while keeping the same task id.
        """

        with self._lock:
            record = self._tasks.get(task_id)
            pause_event = self._pause_events.get(task_id)
            cancel_event = self._cancel_events.get(task_id)
            if record is None or pause_event is None or cancel_event is None:
                return None

            if record.request is None:
                return self._build_snapshot(record)

            if record.status in {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.PAUSING}:
                return self._build_snapshot(record)

            effective_end_stage = end_stage or record.request.end_stage or PIPELINE_STAGE_SEQUENCE[-1]
            effective_inherit = (
                record.request.inherit_intermediates
                if inherit_intermediates is None
                else inherit_intermediates
            )
            effective_inherit_stages = self._resolve_effective_inherit_stages(
                record,
                inherit_intermediates=effective_inherit,
                inherit_stage_outputs=inherit_stage_outputs,
            )
            record.request.end_stage = effective_end_stage
            record.request.inherit_intermediates = effective_inherit
            record.request.inherit_stage_outputs = effective_inherit_stages
            if record.draft_state is not None:
                record.draft_state.inherit_intermediates = effective_inherit
                record.draft_state.inherit_stage_outputs = list(effective_inherit_stages)

            reusable_stages = self._resolve_reusable_stage_prefix(record)
            if reusable_stages and get_stage_index(reusable_stages[-1]) >= get_stage_index(effective_end_stage):
                no_work_message = (
                    f"Continue skipped because stage {effective_end_stage.value} "
                    "is already available in this task."
                )
                record.updated_at = utc_now()
                record.recent_logs.append(no_work_message)
                record.recent_logs = record.recent_logs[-24:]
                record.progress.last_heartbeat_at = record.updated_at
                record.progress.last_heartbeat_message = no_work_message
                record.progress.message = no_work_message
                self._persist_task_unlocked(record, force=True)
                return self._build_snapshot(record)

            pause_event.clear()
            cancel_event.clear()

        if not effective_inherit:
            clear_task_directory(task_id)

        with self._lock:
            record = self._require_task_unlocked(task_id)
            continue_message = (
                (
                    f"Task continue requested to stage {effective_end_stage.value} "
                    f"using inherited stages: {', '.join(stage.value for stage in reusable_stages) or 'none'}."
                )
                if effective_inherit
                else f"Task restart requested to stage {effective_end_stage.value} without inherited intermediates."
            )
            record.status = TaskStatus.QUEUED
            record.error = None
            record.updated_at = utc_now()
            if not effective_inherit:
                record.result = None
                record.last_completed_stage = None
                record.completed_stages = []
                record.progress = TaskProgress(
                    stage=PIPELINE_STAGE_SEQUENCE[0],
                    total_units=1,
                    message=continue_message,
                )
            else:
                restart_stage = self._resolve_start_stage(record)
                reusable_stages = self._resolve_reusable_stage_prefix(record)
                record.completed_stages = list(reusable_stages)
                record.last_completed_stage = reusable_stages[-1] if reusable_stages else None
                if restart_stage is not None:
                    self._reset_result_from_stage_unlocked(record, restart_stage)
                record.progress = TaskProgress(
                    stage=restart_stage or PIPELINE_STAGE_SEQUENCE[0],
                    total_units=1,
                    message=continue_message,
                )
            record.recent_logs.append(continue_message)
            record.recent_logs = record.recent_logs[-24:]
            record.progress.last_heartbeat_at = record.updated_at
            record.progress.last_heartbeat_message = continue_message
            self._persist_task_unlocked(record, force=True)

        task_logger.info(
            "task_id=%s | continue_requested | end_stage=%s | inherit=%s",
            task_id,
            effective_end_stage.value,
            effective_inherit,
        )
        self._submit_task(task_id)
        return self.get_task(task_id)

    def rerun_task(self, task_id: str) -> RiverTaskSnapshot | None:
        """Reset a task and schedule it to recompute from scratch."""

        with self._lock:
            record = self._tasks.get(task_id)
            pause_event = self._pause_events.get(task_id)
            cancel_event = self._cancel_events.get(task_id)
            if record is None or pause_event is None or cancel_event is None:
                return None

            if record.request is None:
                return self._build_snapshot(record)

            if record.status in {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.PAUSING}:
                return self._build_snapshot(record)

            pause_event.clear()
            cancel_event.clear()

        clear_task_directory(task_id)

        with self._lock:
            record = self._require_task_unlocked(task_id)
            rerun_message = "Task rerun requested from the beginning."
            record.status = TaskStatus.QUEUED
            record.progress = TaskProgress(stage=PIPELINE_STAGE_SEQUENCE[0], total_units=1, message=rerun_message)
            record.updated_at = utc_now()
            record.error = None
            record.result = None
            record.last_completed_stage = None
            record.completed_stages = []
            if record.request is not None:
                record.draft_state = DraftTaskState(
                    input_path=record.request.input_path,
                    mask_path=record.request.mask_path,
                    output_path=record.request.output_path,
                    config=record.request.config.model_copy(deep=True),
                    inherit_intermediates=record.request.inherit_intermediates,
                    inherit_stage_outputs=list(record.request.inherit_stage_outputs or []),
                )
            record.recent_logs.append(rerun_message)
            record.recent_logs = record.recent_logs[-24:]
            record.progress.last_heartbeat_at = record.updated_at
            record.progress.last_heartbeat_message = rerun_message
            self._persist_task_unlocked(record, force=True)

        task_logger.info("task_id=%s | rerun_requested", task_id)
        self._submit_task(task_id)
        return self.get_task(task_id)

    def cancel_task(self, task_id: str) -> RiverTaskSnapshot | None:
        """Mark a task for cancellation and return the latest snapshot."""

        with self._lock:
            record = self._tasks.get(task_id)
            cancel_event = self._cancel_events.get(task_id)
            if record is None or cancel_event is None:
                return None

            cancel_event.set()
            if record.status in {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.PAUSING, TaskStatus.PAUSED}:
                record.status = TaskStatus.CANCELED
                cancel_message = "Cancellation requested."
                record.updated_at = utc_now()
                record.recent_logs.append(cancel_message)
                record.recent_logs = record.recent_logs[-24:]
                record.progress.last_heartbeat_at = record.updated_at
                record.progress.last_heartbeat_message = cancel_message
                record.progress.message = cancel_message
                self._persist_task_unlocked(record, force=True)

        task_logger.warning("task_id=%s | cancel_requested", task_id)
        return self.get_task(task_id)

    def _load_persisted_tasks(self) -> None:
        """Load persisted task records so the frontend can reconnect to them."""

        for record in load_task_records():
            if record.status in {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.PAUSING}:
                restore_message = "Task was restored after backend restart. Resume to continue from completed stages."
                record.status = TaskStatus.PAUSED
                record.updated_at = utc_now()
                record.recent_logs.append(restore_message)
                record.recent_logs = record.recent_logs[-24:]
                record.progress.last_heartbeat_at = record.updated_at
                record.progress.last_heartbeat_message = restore_message
                record.progress.message = restore_message

            self._tasks[record.task_id] = record
            self._cancel_events[record.task_id] = Event()
            self._pause_events[record.task_id] = Event()
            self._last_persist_at[record.task_id] = 0.0
            self._persist_task_unlocked(record, force=True)

    def _submit_task(self, task_id: str) -> None:
        """Queue one task worker and its queue-state monitor."""

        self._executor.submit(self._run_task, task_id)
        Thread(
            target=self._monitor_queued_task,
            args=(task_id,),
            daemon=True,
            name=f"river-queue-monitor-{task_id[:8]}",
        ).start()

    def _run_task(self, task_id: str) -> None:
        """Execute the pipeline for one task inside the background worker."""

        reporter = InMemoryProgressReporter(self, task_id)
        with self._lock:
            if task_id in self._active_workers:
                return

            task = self._require_task_unlocked(task_id)
            if task.request is None:
                task_logger.warning("task_id=%s | skipped_start_without_request", task_id)
                return
            self._active_workers.add(task_id)
            existing_result = task.result.model_copy(deep=True) if task.result else None
            effective_request = task.request.model_copy(deep=True)
            effective_request.start_stage = self._resolve_start_stage(task)
            start_message = (
                "Task resumed from the last completed stage."
                if task.last_completed_stage is not None and effective_request.inherit_intermediates
                else "Task started."
            )
            task.status = TaskStatus.RUNNING
            if task.result is None:
                task.result = prepare_initial_result(task_id, effective_request)
            task.error = None
            task.updated_at = utc_now()
            task.recent_logs.append(start_message)
            task.recent_logs = task.recent_logs[-24:]
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = start_message
            task.progress.message = start_message
            self._persist_task_unlocked(task, force=True)

        task_logger.info("task_id=%s | started", task_id)
        try:
            result = self._pipeline.run(
                task_id,
                effective_request,
                reporter,
                existing_result=existing_result,
            )
            if self._is_canceled(task_id):
                task_logger.warning("task_id=%s | stopped_after_cancel", task_id)
                return

            with self._lock:
                task = self._require_task_unlocked(task_id)
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.updated_at = utc_now()
                task.recent_logs.append("Task completed successfully.")
                task.recent_logs = task.recent_logs[-24:]
                task.progress.last_heartbeat_at = task.updated_at
                task.progress.last_heartbeat_message = "Task completed successfully."
                task.progress.message = "Task completed successfully."
                self._persist_task_unlocked(task, force=True)
            task_logger.info("task_id=%s | completed", task_id)
        except TaskPauseRequested:
            with self._lock:
                task = self._require_task_unlocked(task_id)
                pause_message = "Task paused at a safe point. Resume will continue from completed stages."
                task.status = TaskStatus.PAUSED
                task.updated_at = utc_now()
                task.recent_logs.append(pause_message)
                task.recent_logs = task.recent_logs[-24:]
                task.progress.last_heartbeat_at = task.updated_at
                task.progress.last_heartbeat_message = pause_message
                task.progress.message = pause_message
                self._persist_task_unlocked(task, force=True)
            task_logger.info("task_id=%s | paused", task_id)
        except Exception as exc:
            with self._lock:
                task = self._require_task_unlocked(task_id)
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                task.updated_at = utc_now()
                task.recent_logs.append(f"Task failed: {exc}")
                task.recent_logs = task.recent_logs[-24:]
                task.progress.last_heartbeat_at = task.updated_at
                task.progress.last_heartbeat_message = f"Task failed: {exc}"
                task.progress.message = f"Task failed: {exc}"
                self._persist_task_unlocked(task, force=True)
            task_logger.exception("task_id=%s | failed", task_id)
        finally:
            with self._lock:
                self._active_workers.discard(task_id)

    def _resolve_start_stage(self, task: RiverTaskRecord) -> PipelineStage | None:
        """Resolve the effective start stage for one execution attempt."""

        if task.request is None:
            return None

        if not task.request.inherit_intermediates:
            return task.request.start_stage

        reusable_stages = self._resolve_reusable_stage_prefix(task)
        if not reusable_stages:
            return task.request.start_stage

        next_stage = get_next_stage(reusable_stages[-1])
        if next_stage is None:
            return reusable_stages[-1]

        return next_stage

    def _normalize_stage_selection(self, stages: list[PipelineStage] | None) -> list[PipelineStage]:
        """Return de-duplicated stage selections in pipeline order."""

        if not stages:
            return []

        selected = set(stages)
        return [stage for stage in PIPELINE_STAGE_SEQUENCE if stage in selected]

    def _resolve_effective_inherit_stages(
        self,
        task: RiverTaskRecord,
        inherit_intermediates: bool,
        inherit_stage_outputs: list[PipelineStage] | None,
    ) -> list[PipelineStage]:
        """Resolve which completed stages are allowed to be reused for the next run."""

        if not inherit_intermediates:
            return []

        if inherit_stage_outputs is None:
            requested_stages = task.request.inherit_stage_outputs
            if requested_stages is None:
                return list(task.completed_stages)
            return self._normalize_stage_selection(requested_stages)

        return self._normalize_stage_selection(inherit_stage_outputs)

    def _resolve_reusable_stage_prefix(self, task: RiverTaskRecord) -> list[PipelineStage]:
        """Return the contiguous completed prefix that the next run is allowed to reuse."""

        if task.request is None:
            return []

        if not task.request.inherit_intermediates:
            return []

        completed_stage_set = set(task.completed_stages)
        allowed_stage_set = set(
            task.completed_stages
            if task.request.inherit_stage_outputs is None
            else task.request.inherit_stage_outputs
        )
        reusable_stages: list[PipelineStage] = []

        for stage in PIPELINE_STAGE_SEQUENCE:
            if stage not in completed_stage_set or stage not in allowed_stage_set:
                break
            reusable_stages.append(stage)

        return reusable_stages

    def _reset_result_from_stage_unlocked(self, task: RiverTaskRecord, start_stage: PipelineStage) -> None:
        """Drop stale artifact metadata at and after the restart stage while keeping reusable outputs."""

        if task.result is None:
            return

        reset_from_index = get_stage_index(start_stage)
        for artifact_key, artifact in list(task.result.artifacts.items()):
            if get_stage_index(artifact.stage) < reset_from_index:
                continue

            task.result.artifacts[artifact_key] = ArtifactRecord(
                key=artifact.key,
                label=artifact.label,
                stage=artifact.stage,
                status=ArtifactStatus.PENDING,
            )

        task.result.input_preview = task.result.artifacts.get("input_preview", ArtifactRecord(key="input_preview", label="", stage=PipelineStage.IO)).preview_path
        task.result.auto_mask = task.result.artifacts.get("auto_mask", ArtifactRecord(key="auto_mask", label="", stage=PipelineStage.PREPROCESS)).preview_path
        task.result.terrain_preprocessed = task.result.artifacts.get("terrain_preprocessed", ArtifactRecord(key="terrain_preprocessed", label="", stage=PipelineStage.PREPROCESS)).preview_path
        task.result.flow_direction = task.result.artifacts.get("flow_direction", ArtifactRecord(key="flow_direction", label="", stage=PipelineStage.FLOW_DIRECTION)).preview_path
        task.result.flow_accumulation = task.result.artifacts.get("flow_accumulation", ArtifactRecord(key="flow_accumulation", label="", stage=PipelineStage.FLOW_ACCUMULATION)).preview_path
        task.result.channel_mask = task.result.artifacts.get("channel_mask", ArtifactRecord(key="channel_mask", label="", stage=PipelineStage.CHANNEL_EXTRACT)).preview_path

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
            task.result.auto_mask = task.result.artifacts["auto_mask"].preview_path
            task.result.terrain_preprocessed = task.result.artifacts["terrain_preprocessed"].preview_path
            task.result.flow_direction = task.result.artifacts["flow_direction"].preview_path
            task.result.flow_accumulation = task.result.artifacts["flow_accumulation"].preview_path
            task.result.channel_mask = task.result.artifacts["channel_mask"].preview_path
            task.updated_at = utc_now()
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = f"Artifact ready: {artifact.label}"
            self._persist_task_unlocked(task, force=True)

        task_logger.info(
            "task_id=%s | artifact=%s | status=%s | preview=%s",
            task_id,
            artifact.key,
            artifact.status.value,
            artifact.preview_path,
        )

    def _append_log(self, task_id: str, message: str) -> None:
        """Append a bounded log line to keep the snapshot light."""

        with self._lock:
            task = self._require_task_unlocked(task_id)
            task.recent_logs.append(message)
            task.recent_logs = task.recent_logs[-24:]
            task.updated_at = utc_now()
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = message
            self._persist_task_unlocked(task)

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
            task.progress.parallel_work = None
            task.updated_at = utc_now()
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = message
            self._persist_task_unlocked(task)

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
            force_persist = (
                processed == total
                or processed == 1
                or processed % max(1, total // 16) == 0
            )
            self._persist_task_unlocked(task, force=force_persist)

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
            task.progress.parallel_work = None
            task.updated_at = utc_now()
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = task.progress.message
            if task.progress.stage not in task.completed_stages:
                task.completed_stages.append(task.progress.stage)
                task.completed_stages.sort(key=PIPELINE_STAGE_SEQUENCE.index)
            task.last_completed_stage = task.progress.stage
            self._persist_task_unlocked(task, force=True)

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
            self._persist_task_unlocked(task)

    def _set_parallel_work(
        self,
        task_id: str,
        label: str,
        strategy: str,
        chunks: list[tuple[str, str, int]],
    ) -> None:
        """Publish one new structured parallel-work snapshot for the current stage."""

        with self._lock:
            task = self._require_task_unlocked(task_id)
            normalized_chunks = [
                ParallelChunkProgress(
                    chunk_id=chunk_id,
                    label=chunk_label,
                    status=ParallelChunkStatus.PENDING,
                    processed_units=0,
                    total_units=max(1, total_units),
                    percent=0.0,
                    detail="Waiting for a worker.",
                )
                for chunk_id, chunk_label, total_units in chunks
            ]
            task.progress.parallel_work = ParallelWorkProgress(
                label=label,
                strategy=strategy,
                processed_units=0,
                total_units=max(1, sum(chunk.total_units for chunk in normalized_chunks)),
                completed_chunks=0,
                total_chunks=len(normalized_chunks),
                chunks=normalized_chunks,
            )
            task.updated_at = utc_now()
            self._persist_task_unlocked(task, force=True)

    def _update_parallel_chunk(
        self,
        task_id: str,
        chunk_id: str,
        status: str,
        processed_units: int,
        total_units: int,
        detail: str = "",
    ) -> None:
        """Update one chunk inside the active structured parallel-work snapshot."""

        with self._lock:
            task = self._require_task_unlocked(task_id)
            parallel_work = task.progress.parallel_work
            if parallel_work is None:
                return

            target_chunk = next(
                (chunk for chunk in parallel_work.chunks if chunk.chunk_id == chunk_id),
                None,
            )
            if target_chunk is None:
                return

            safe_total_units = max(1, total_units)
            safe_processed_units = min(safe_total_units, max(0, processed_units))
            target_chunk.status = ParallelChunkStatus(status)
            target_chunk.total_units = safe_total_units
            target_chunk.processed_units = safe_processed_units
            target_chunk.percent = round((safe_processed_units / safe_total_units) * 100.0, 2)
            target_chunk.detail = detail

            parallel_work.total_units = max(1, sum(chunk.total_units for chunk in parallel_work.chunks))
            parallel_work.processed_units = min(
                parallel_work.total_units,
                sum(chunk.processed_units for chunk in parallel_work.chunks),
            )
            parallel_work.completed_chunks = sum(
                1 for chunk in parallel_work.chunks if chunk.status == ParallelChunkStatus.COMPLETED
            )
            parallel_work.total_chunks = len(parallel_work.chunks)
            task.updated_at = utc_now()
            self._persist_task_unlocked(
                task,
                force=(
                    target_chunk.status == ParallelChunkStatus.COMPLETED
                    or safe_processed_units == 0
                    or safe_processed_units == safe_total_units
                ),
            )

    def _clear_parallel_work(self, task_id: str) -> None:
        """Clear any structured parallel-work snapshot for the active stage."""

        with self._lock:
            task = self._require_task_unlocked(task_id)
            if task.progress.parallel_work is None:
                return

            task.progress.parallel_work = None
            task.updated_at = utc_now()
            self._persist_task_unlocked(task)

    def _checkpoint_task_control(self, task_id: str) -> None:
        """Raise at the next safe point when a pause was requested."""

        if self._is_canceled(task_id):
            return

        with self._lock:
            task = self._require_task_unlocked(task_id)
            pause_event = self._pause_events.get(task_id)
            if pause_event is None or not pause_event.is_set():
                return

            task.status = TaskStatus.PAUSED
            pause_message = "Pause acknowledged. The current stage will be rerun on resume."
            task.updated_at = utc_now()
            task.recent_logs.append(pause_message)
            task.recent_logs = task.recent_logs[-24:]
            task.progress.last_heartbeat_at = task.updated_at
            task.progress.last_heartbeat_message = pause_message
            task.progress.message = pause_message
            self._persist_task_unlocked(task, force=True)

        raise TaskPauseRequested(pause_message)

    def _is_canceled(self, task_id: str) -> bool:
        """Return whether a cancel event was raised for the task."""

        with self._lock:
            cancel_event = self._cancel_events.get(task_id)
            return bool(cancel_event and cancel_event.is_set())

    def _build_snapshot(self, record: RiverTaskRecord) -> RiverTaskSnapshot:
        """Convert a mutable task record into one API-safe snapshot."""

        return RiverTaskSnapshot(
            task_id=record.task_id,
            name=record.name,
            status=record.status,
            draft_state=record.draft_state.model_copy(deep=True) if record.draft_state else None,
            progress=record.progress.model_copy(deep=True),
            created_at=record.created_at,
            updated_at=record.updated_at,
            result=record.result.model_copy(deep=True) if record.result else None,
            error=record.error,
            recent_logs=list(record.recent_logs),
            last_completed_stage=record.last_completed_stage,
            completed_stages=list(record.completed_stages),
        )

    def _persist_task_unlocked(self, task: RiverTaskRecord, force: bool = False) -> None:
        """Persist a task record with light throttling for hot progress loops."""

        now = monotonic()
        last_persisted_at = self._last_persist_at.get(task.task_id, 0.0)
        if not force and now - last_persisted_at < 0.5:
            return

        save_task_record(task)
        self._last_persist_at[task.task_id] = now

    def _require_task_unlocked(self, task_id: str) -> RiverTaskRecord:
        """
        Fetch a task record while the caller already owns the lock.

        The split helper avoids nested lock acquisition and keeps the rest of
        the task-runner methods shallow.
        """

        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} does not exist.")

        return task
