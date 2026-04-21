from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import DraftTaskState, PIPELINE_STAGE_SEQUENCE, PipelineResult, PipelineStage, RiverTaskRequest, TaskStatus
from app.task_runner import InMemoryTaskRunner


class FakePipeline:
    """Small fake pipeline used to exercise the task runner state machine."""

    def run(self, task_id, request, reporter, existing_result=None):  # type: ignore[no-untyped-def]
        del task_id
        result = PipelineResult() if existing_result is None else existing_result
        start_stage = request.start_stage or PIPELINE_STAGE_SEQUENCE[0]
        end_stage = request.end_stage or PipelineStage.PREPROCESS
        start_index = PIPELINE_STAGE_SEQUENCE.index(start_stage)
        end_index = PIPELINE_STAGE_SEQUENCE.index(end_stage)
        for stage in PIPELINE_STAGE_SEQUENCE[start_index : end_index + 1]:
            reporter.begin_stage(stage, 12, f"fake {stage.value}")
            for step in range(12):
                time.sleep(0.01)
                reporter.advance(1, f"fake {stage.value} step {step + 1}/12")
            reporter.complete_stage(f"fake {stage.value} complete")
        return result


class TaskRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.load_patch = patch('app.task_runner.load_task_records', return_value=[])
        self.save_patch = patch('app.task_runner.save_task_record')
        self.clear_patch = patch('app.task_runner.clear_task_directory')
        self.delete_patch = patch('app.task_runner.delete_task_directory')
        self.load_patch.start()
        self.save_patch.start()
        self.clear_patch.start()
        self.delete_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.addCleanup(self.save_patch.stop)
        self.addCleanup(self.clear_patch.stop)
        self.addCleanup(self.delete_patch.stop)

    def _build_request(self) -> RiverTaskRequest:
        return RiverTaskRequest(
            input_path='data/input/example-height-map.pgm',
            output_path='data/output/test-channel.png',
            end_stage=PipelineStage.PREPROCESS,
        )

    def _wait_for_status(self, runner: InMemoryTaskRunner, task_id: str, expected: set[TaskStatus], timeout: float = 3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            snapshot = runner.get_task(task_id)
            if snapshot is not None and snapshot.status in expected:
                return snapshot
            time.sleep(0.02)
        self.fail(f"Task {task_id} did not reach {expected} within {timeout}s.")

    def test_task_runner_lists_completed_tasks(self) -> None:
        runner = InMemoryTaskRunner()
        runner._pipeline = FakePipeline()

        task_id = runner.create_task(self._build_request())
        snapshot = self._wait_for_status(runner, task_id, {TaskStatus.COMPLETED})

        self.assertEqual(snapshot.last_completed_stage, PipelineStage.PREPROCESS)
        self.assertIn(PipelineStage.IO, snapshot.completed_stages)
        self.assertIn(PipelineStage.PREPROCESS, snapshot.completed_stages)
        self.assertEqual(runner.list_tasks()[0].task_id, task_id)

    def test_task_draft_can_be_started_renamed_and_deleted(self) -> None:
        runner = InMemoryTaskRunner()
        runner._pipeline = FakePipeline()

        draft_snapshot = runner.create_draft_task("Primary Basin")
        self.assertEqual(draft_snapshot.status, TaskStatus.DRAFT)
        self.assertEqual(draft_snapshot.name, "Primary Basin")

        updated_draft = runner.update_draft_state(
            draft_snapshot.task_id,
            DraftTaskState(
                input_path='data/input/example-height-map.pgm',
                mask_path='data/input/uploads/masks/example-mask.png',
                output_path='data/output/draft-channel.png',
            ),
        )
        self.assertIsNotNone(updated_draft)
        self.assertEqual(updated_draft.draft_state.input_path, 'data/input/example-height-map.pgm')
        self.assertEqual(updated_draft.draft_state.output_path, 'data/output/draft-channel.png')

        renamed_snapshot = runner.rename_task(draft_snapshot.task_id, "Main Basin")
        self.assertIsNotNone(renamed_snapshot)
        self.assertEqual(renamed_snapshot.name, "Main Basin")

        started_snapshot = runner.start_task(draft_snapshot.task_id, self._build_request())
        self.assertIsNotNone(started_snapshot)
        self.assertIn(started_snapshot.status, {TaskStatus.QUEUED, TaskStatus.RUNNING})

        completed_snapshot = self._wait_for_status(runner, draft_snapshot.task_id, {TaskStatus.COMPLETED})
        self.assertEqual(completed_snapshot.name, "Main Basin")

        deleted_snapshot = runner.delete_task(draft_snapshot.task_id)
        self.assertIsNotNone(deleted_snapshot)
        self.assertIsNone(runner.get_task(draft_snapshot.task_id))

    def test_task_can_pause_and_resume_from_last_completed_stage(self) -> None:
        runner = InMemoryTaskRunner()
        runner._pipeline = FakePipeline()

        task_id = runner.create_task(self._build_request())
        time.sleep(0.05)
        paused_snapshot = runner.pause_task(task_id)
        self.assertIsNotNone(paused_snapshot)

        paused_snapshot = self._wait_for_status(runner, task_id, {TaskStatus.PAUSED})
        self.assertEqual(paused_snapshot.status, TaskStatus.PAUSED)

        resumed_snapshot = runner.resume_task(task_id)
        self.assertIsNotNone(resumed_snapshot)
        completed_snapshot = self._wait_for_status(runner, task_id, {TaskStatus.COMPLETED})

        self.assertEqual(completed_snapshot.status, TaskStatus.COMPLETED)
        self.assertEqual(completed_snapshot.last_completed_stage, PipelineStage.PREPROCESS)
        self.assertEqual(completed_snapshot.completed_stages, [PipelineStage.IO, PipelineStage.PREPROCESS])

    def test_task_can_continue_to_later_stage_with_same_task_id(self) -> None:
        runner = InMemoryTaskRunner()
        runner._pipeline = FakePipeline()

        task_id = runner.create_task(self._build_request())
        completed_snapshot = self._wait_for_status(runner, task_id, {TaskStatus.COMPLETED})
        self.assertEqual(completed_snapshot.last_completed_stage, PipelineStage.PREPROCESS)

        continued_snapshot = runner.continue_task(
            task_id,
            end_stage=PipelineStage.FLOW_DIRECTION,
            inherit_intermediates=True,
        )
        self.assertIsNotNone(continued_snapshot)
        completed_snapshot = self._wait_for_status(runner, task_id, {TaskStatus.COMPLETED})

        self.assertEqual(completed_snapshot.task_id, task_id)
        self.assertEqual(completed_snapshot.last_completed_stage, PipelineStage.FLOW_DIRECTION)
        self.assertEqual(
            completed_snapshot.completed_stages,
            [PipelineStage.IO, PipelineStage.PREPROCESS, PipelineStage.FLOW_DIRECTION],
        )
        self.assertIn(PipelineStage.PREPROCESS, completed_snapshot.completed_stages)
        self.assertIn(PipelineStage.FLOW_DIRECTION, completed_snapshot.completed_stages)

    def test_continue_without_inheritance_restarts_stage_history(self) -> None:
        runner = InMemoryTaskRunner()
        runner._pipeline = FakePipeline()

        task_id = runner.create_task(self._build_request())
        completed_snapshot = self._wait_for_status(runner, task_id, {TaskStatus.COMPLETED})
        self.assertEqual(completed_snapshot.last_completed_stage, PipelineStage.PREPROCESS)

        continued_snapshot = runner.continue_task(
            task_id,
            end_stage=PipelineStage.FLOW_DIRECTION,
            inherit_intermediates=False,
        )
        self.assertIsNotNone(continued_snapshot)
        completed_snapshot = self._wait_for_status(runner, task_id, {TaskStatus.COMPLETED})

        self.assertEqual(completed_snapshot.last_completed_stage, PipelineStage.FLOW_DIRECTION)
        self.assertEqual(
            completed_snapshot.completed_stages,
            [PipelineStage.IO, PipelineStage.PREPROCESS, PipelineStage.FLOW_DIRECTION],
        )

    def test_continue_skips_when_requested_stage_already_exists(self) -> None:
        runner = InMemoryTaskRunner()
        runner._pipeline = FakePipeline()

        task_id = runner.create_task(self._build_request())
        completed_snapshot = self._wait_for_status(runner, task_id, {TaskStatus.COMPLETED})
        self.assertEqual(completed_snapshot.last_completed_stage, PipelineStage.PREPROCESS)

        skipped_snapshot = runner.continue_task(
            task_id,
            end_stage=PipelineStage.PREPROCESS,
            inherit_intermediates=True,
        )
        self.assertIsNotNone(skipped_snapshot)
        self.assertEqual(skipped_snapshot.status, TaskStatus.COMPLETED)
        self.assertEqual(skipped_snapshot.last_completed_stage, PipelineStage.PREPROCESS)
        self.assertIn("already available", skipped_snapshot.progress.message)

    def test_continue_with_partial_stage_reuse_restarts_from_first_disabled_stage(self) -> None:
        runner = InMemoryTaskRunner()
        runner._pipeline = FakePipeline()

        task_id = runner.create_task(
            RiverTaskRequest(
                input_path='data/input/example-height-map.pgm',
                output_path='data/output/test-channel.png',
                end_stage=PipelineStage.CHANNEL_EXTRACT,
            )
        )
        completed_snapshot = self._wait_for_status(runner, task_id, {TaskStatus.COMPLETED})
        self.assertEqual(completed_snapshot.last_completed_stage, PipelineStage.CHANNEL_EXTRACT)

        continued_snapshot = runner.continue_task(
            task_id,
            end_stage=PipelineStage.CHANNEL_EXTRACT,
            inherit_intermediates=True,
            inherit_stage_outputs=[PipelineStage.IO, PipelineStage.PREPROCESS],
        )
        self.assertIsNotNone(continued_snapshot)

        record = runner._tasks[task_id]
        self.assertEqual(record.request.inherit_stage_outputs, [PipelineStage.IO, PipelineStage.PREPROCESS])
        self.assertEqual(record.completed_stages, [PipelineStage.IO, PipelineStage.PREPROCESS])
        self.assertEqual(record.last_completed_stage, PipelineStage.PREPROCESS)
        self.assertEqual(runner._resolve_start_stage(record), PipelineStage.FLOW_DIRECTION)

        completed_snapshot = self._wait_for_status(runner, task_id, {TaskStatus.COMPLETED})
        self.assertEqual(completed_snapshot.last_completed_stage, PipelineStage.CHANNEL_EXTRACT)


if __name__ == '__main__':
    unittest.main()
