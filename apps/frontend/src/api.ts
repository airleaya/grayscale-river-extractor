import type { CreateTaskRequest, TaskSnapshot, UploadedFileInfo } from './types'

const API_BASE_URL = 'http://127.0.0.1:8000'

/**
 * Keep backend calls in a dedicated module so UI components stay focused on
 * rendering and local interaction state instead of fetch wiring.
 */
export async function createTask(payload: CreateTaskRequest): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/tasks`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    throw new Error(`Failed to create task: ${response.status}`)
  }

  const data = (await response.json()) as { task_id: string }
  return data.task_id
}

export async function getTask(taskId: string): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`)

  if (!response.ok) {
    throw new Error(`Failed to load task ${taskId}: ${response.status}`)
  }

  return (await response.json()) as TaskSnapshot
}

export async function cancelTask(taskId: string): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/cancel`, {
    method: 'POST',
  })

  if (!response.ok) {
    throw new Error(`Failed to cancel task ${taskId}: ${response.status}`)
  }

  return (await response.json()) as TaskSnapshot
}

export async function uploadInputFile(file: File): Promise<UploadedFileInfo> {
  const response = await fetch(`${API_BASE_URL}/files/upload`, {
    method: 'POST',
    headers: {
      'Content-Type': file.type || 'application/octet-stream',
      'X-Filename': encodeURIComponent(file.name),
    },
    body: await file.arrayBuffer(),
  })

  if (!response.ok) {
    throw new Error(`文件上传失败: ${response.status}`)
  }

  return (await response.json()) as UploadedFileInfo
}

export async function listUploadedFiles(): Promise<UploadedFileInfo[]> {
  const response = await fetch(`${API_BASE_URL}/files/uploaded`)

  if (!response.ok) {
    throw new Error(`历史素材读取失败: ${response.status}`)
  }

  return (await response.json()) as UploadedFileInfo[]
}
