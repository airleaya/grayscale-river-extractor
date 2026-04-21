# Grayscale River Extractor

Grayscale River Extractor 是一个“高度灰度图 -> 河道二值图”的前后端分离实验项目，目标是把大幅灰度栅格中的河道结构稳定提取出来，并让任务过程对用户尽可能可见、可控、可恢复。

当前项目已经具备：
- 中文任务工作台
- FastAPI 后端
- 持久化任务中心
- 草稿任务与自动保存
- 暂停 / 继续 / 重跑 / 分阶段启动
- 前后端分离运行，前端关闭不会杀掉后端任务
- 前端重开后重连已有任务并恢复进度监听
- 输入素材库与遮罩素材库
- 中间产物查看与多图层叠图查看
- 高频阶段反馈与任务日志

## 当前结构

```text
river/
  apps/
    backend/
      app/
        logging_utils.py
        main.py
        models.py
        pipeline.py
        raster_algorithms.py
        storage.py
        task_runner.py
      logs/
      requirements.txt
      tests/
    frontend/
      public/
      src/
        api.ts
        App.css
        App.cockpit.css
        ArtifactViewerTabs.tsx
        PixelRasterViewer.tsx
        TaskWorkbenchZh.tsx
        constants.ts
        main.tsx
        types.ts
      package.json
  data/
    input/
      uploads/
    masks/
      uploads/
    output/
      tasks/
  docs/
  scripts/
    start-backend.ps1
    start-dev.ps1
    start-frontend.ps1
  start.bat
```

## 运行方式

后端：

```powershell
.\scripts\start-backend.ps1
```

前端：

```powershell
.\scripts\start-frontend.ps1
```

一键启动：

```bat
start.bat
```

或者：

```powershell
.\scripts\start-dev.ps1
```

默认地址：
- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`
- 健康检查：`http://127.0.0.1:8000/health`

## 任务工作流

当前推荐流程：
1. 在右侧任务中心新建一个草稿任务。
2. 在左侧选择输入主图、用户遮罩和算法参数。
3. 草稿状态会自动保存到后端，刷新页面后仍可恢复。
4. 可以直接启动到最终阶段，也可以点某个阶段按钮只跑到该阶段。
5. 对历史任务可以按阶段选择是否继承已有产物，再继续后续阶段。
6. 运行中的任务支持暂停、继续、取消和重跑。

任务管理能力当前包括：
- 新建任务
- 草稿自动保存
- 任务重命名
- 任务删除
- 暂停 / 继续
- 继续运行到指定阶段
- 从头重跑

## 前端能力

当前前端工作台支持：
- 选择文件后立即上传
- 输入主图与用户遮罩分别管理
- 自动遮罩与用户遮罩分别查看
- 左侧参数区与右侧任务区联动锁定
- 任务运行时禁止会破坏一致性的左侧操作
- 任务状态、当前子步骤、最近活动、日志、阶段进度展示
- 图层叠加查看
- 每个阶段图层独立开关和透明度调节

图层叠加规则：
- 输入图始终在最底层
- 阶段越靠后越在上层
- 最终输出永远在最上层

## 当前算法说明

预处理阶段当前支持：
- 高低映射切换
- 可选平滑
- 可选填洼
- 单像素洼地直接用八邻域平均值填补
- 自动遮罩
- 用户遮罩参与后续计算
- 无出口平坡 / 封闭平坦区域修复

水文阶段当前支持：
- `D8` 单流向
- 流向阶段高频进度反馈
- 汇流累积阶段高频进度反馈
- 河道提取阶段高频进度反馈
- 基于累积阈值的河道提取
- 触边 / 触遮罩边缘后结束河道，避免沿边缘行走
- 河道长度阈值过滤，剔除短河道碎段

模块职责：
- `pipeline.py` 负责阶段编排、产物落盘和进度语义
- `raster_algorithms.py` 负责纯栅格算法、填洼、流向、汇流与河道后处理
- `task_runner.py` 负责持久化任务生命周期、暂停继续、阶段继承和前后端解耦
- `storage.py` 负责任务记录、任务目录和上传素材存储

## 数据目录

- 输入素材：`data/input/uploads/`
- 遮罩素材：`data/masks/uploads/`
- 任务目录：`data/output/tasks/<task_id>/`
- 最终输出：`data/output/`

每个任务目录会保存：
- 任务元数据
- 阶段产物
- 预览图
- 运行日志所需的阶段反馈信息

## API 概览

任务相关：
- `POST /tasks/draft`
- `GET /tasks`
- `GET /tasks/{task_id}`
- `POST /tasks/{task_id}/draft-state`
- `POST /tasks/{task_id}/start`
- `POST /tasks/{task_id}/pause`
- `POST /tasks/{task_id}/resume`
- `POST /tasks/{task_id}/continue`
- `POST /tasks/{task_id}/rerun`
- `POST /tasks/{task_id}/rename`
- `DELETE /tasks/{task_id}`
- `POST /tasks/{task_id}/cancel`

素材相关：
- `POST /files/upload?kind=input|mask`
- `GET /files/uploaded?kind=input|mask`
- `POST /files/rename`
- `POST /files/delete`

## 验证方式

后端测试：

```powershell
E:\codex\river\.condaenv\python.exe -m unittest discover -s apps/backend/tests -p "test_*.py"
```

前端构建：

```powershell
npm run build
```

建议在 `apps/frontend` 目录运行前端构建。

## 当前限制

- 当前仍以本地单机任务队列为主，还没有做分布式调度
- 大图算法虽然已有高频反馈，但极端输入下仍需要继续做性能优化
- 现阶段仍主要面向灰度高程风格图像，复杂多源遥感输入尚未系统适配
