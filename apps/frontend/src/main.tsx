import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { TaskWorkbenchZh } from './TaskWorkbenchZh.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <TaskWorkbenchZh />
  </StrictMode>,
)
