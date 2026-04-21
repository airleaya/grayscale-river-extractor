import type {
  ContinueTaskRequest,
  CreateDraftTaskRequest,
  CreateTaskRequest,
  DraftTaskState,
  DeleteUploadedFileRequest,
  RenameTaskRequest,
  RenameUploadedFileRequest,
  TaskSnapshot,
  UploadedFileInfo,
  UploadedFileKind,
} from './types'

const API_BASE_URL = 'http://127.0.0.1:8000'

async function readJsonOrThrow<T>(response: Response, failureMessage: string): Promise<T> {
  if (!response.ok) {
    throw new Error(`${failureMessage}: ${response.status}`)
  }

  return (await response.json()) as T
}

export async function createTask(payload: CreateTaskRequest): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/tasks`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  const data = await readJsonOrThrow<{ task_id: string }>(response, '创建任务失败')
  return data.task_id
}

export async function createDraftTask(payload: CreateDraftTaskRequest): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/draft`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  return readJsonOrThrow<TaskSnapshot>(response, '新建任务失败')
}

export async function startExistingTask(taskId: string, payload: CreateTaskRequest): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/start`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  return readJsonOrThrow<TaskSnapshot>(response, `启动任务 ${taskId} 失败`)
}

export async function updateDraftTaskState(taskId: string, payload: DraftTaskState): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/draft-state`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  return readJsonOrThrow<TaskSnapshot>(response, `保存草稿 ${taskId} 失败`)
}

export async function renameTask(taskId: string, payload: RenameTaskRequest): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/rename`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  return readJsonOrThrow<TaskSnapshot>(response, `重命名任务 ${taskId} 失败`)
}

export async function deleteTask(taskId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`, {
    method: 'DELETE',
  })

  await readJsonOrThrow<{ deleted: boolean }>(response, `删除任务 ${taskId} 失败`)
}

export async function getTask(taskId: string): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`)
  return readJsonOrThrow<TaskSnapshot>(response, `读取任务 ${taskId} 失败`)
}

export async function listTasks(): Promise<TaskSnapshot[]> {
  const response = await fetch(`${API_BASE_URL}/tasks`)
  return readJsonOrThrow<TaskSnapshot[]>(response, '读取任务列表失败')
}

export async function cancelTask(taskId: string): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/cancel`, {
    method: 'POST',
  })

  return readJsonOrThrow<TaskSnapshot>(response, `取消任务 ${taskId} 失败`)
}

export async function pauseTask(taskId: string): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/pause`, {
    method: 'POST',
  })

  return readJsonOrThrow<TaskSnapshot>(response, `暂停任务 ${taskId} 失败`)
}

export async function resumeTask(taskId: string): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/resume`, {
    method: 'POST',
  })

  return readJsonOrThrow<TaskSnapshot>(response, `继续任务 ${taskId} 失败`)
}

export async function continueTask(taskId: string, payload: ContinueTaskRequest): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/continue`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  return readJsonOrThrow<TaskSnapshot>(response, `继续任务 ${taskId} 失败`)
}

export async function rerunTask(taskId: string): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/rerun`, {
    method: 'POST',
  })

  return readJsonOrThrow<TaskSnapshot>(response, `重跑任务 ${taskId} 失败`)
}

export async function uploadInputFile(file: File, kind: UploadedFileKind): Promise<UploadedFileInfo> {
  const response = await fetch(`${API_BASE_URL}/files/upload?kind=${kind}`, {
    method: 'POST',
    headers: {
      'Content-Type': file.type || 'application/octet-stream',
      'X-Filename': encodeURIComponent(file.name),
    },
    body: await file.arrayBuffer(),
  })

  return readJsonOrThrow<UploadedFileInfo>(response, `${kind === 'input' ? '输入' : '遮罩'}上传失败`)
}

export async function listUploadedFiles(kind: UploadedFileKind): Promise<UploadedFileInfo[]> {
  const response = await fetch(`${API_BASE_URL}/files/uploaded?kind=${kind}`)
  return readJsonOrThrow<UploadedFileInfo[]>(response, `${kind === 'input' ? '输入' : '遮罩'}列表读取失败`)
}

export async function renameUploadedFile(payload: RenameUploadedFileRequest): Promise<UploadedFileInfo> {
  const response = await fetch(`${API_BASE_URL}/files/rename`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  return readJsonOrThrow<UploadedFileInfo>(response, '素材重命名失败')
}

export async function deleteUploadedFile(payload: DeleteUploadedFileRequest): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/files/delete`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  await readJsonOrThrow<{ deleted: boolean }>(response, '素材删除失败')
}
