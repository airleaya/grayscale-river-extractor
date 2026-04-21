# River

River 是一个“高度灰度图 -> 河道二值图”的前后端实验项目。

当前版本已经具备一条完整可运行的 MVP 算法链：
- 输入检测
- 预处理
- D8 流向
- 汇流累积
- 基于阈值的河道提取

项目同时具备：
- 中文前端任务工作台
- FastAPI 后端
- 异步任务执行与阶段进度反馈
- 按任务目录保存的中间产物
- 本地按天切割的日志系统
- 像素级图像查看器与中间产物标签页

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
    frontend/
      public/
      src/
        api.ts
        ArtifactViewerTabs.tsx
        constants.ts
        PixelRasterViewer.tsx
        TaskWorkbenchZh.tsx
        index.css
        main.tsx
        types.ts
      package.json
  data/
    input/
      example-height-map.pgm
      uploads/
    output/
      tasks/
  docs/
    problem-forest.md
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

## 主要数据目录

- 示例输入：`data/input/example-height-map.pgm`
- 用户上传输入：`data/input/uploads/`
- 任务产物：`data/output/tasks/<task_id>/`
- 最终输出：`data/output/`

## 当前算法说明

预处理阶段当前支持：
- 高低映射切换
- 可选平滑
- 可选局部填洼
- 无出口平坡 / 封闭平坦盆地修复
- 可选 `nodata` 保留

真实水文阶段当前支持：
- `D8` 单流向
- 有出口平坡导流
- 基于拓扑传播的汇流累积
- 基于累积阈值的二值河道提取

模块职责：
- `pipeline.py` 负责阶段编排、产物落盘和进度语义
- `raster_algorithms.py` 负责纯栅格算法与预览图生成

## 当前验证基线

后端算法层已经补了一组小尺寸单元测试，当前主要覆盖：
- 严格下降时是否选择最陡下游方向
- 有真实出口的狭长平坡是否能被连续导流
- 平坡加权参数关闭后是否会回退到无流向行为

运行方式：

```powershell
E:\codex\river\.condaenv\python.exe -m unittest discover -s apps/backend/tests -p "test_*.py"
```

## 当前限制

- 任务系统仍为内存态，重启后不会保留历史状态
- 河道提取当前是最小阈值法，尚未加入后处理
- 前端步骤入口已预留，但大部分高级参数面板仍未展开
