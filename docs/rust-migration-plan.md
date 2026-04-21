# Rust 迁移草案

## 目标

在不推翻当前前后端架构的前提下，把最耗时的栅格计算内核逐步迁移到 Rust。

迁移后的总体结构保持为：

- 前端：继续负责参数配置、任务启动、进度展示、图像查看
- Python 后端：继续负责 API、任务系统、文件读写、阶段编排、产物落盘
- Rust 计算内核：负责重计算函数，尽量只做“数组进、数组出”

这条路线的核心原则是：

1. 先保留 Python 工程骨架
2. 再逐步替换算法热点
3. 迁移过程中保持 Python / Rust 双实现并存
4. 保留回退开关和一致性验证能力

## 为什么现在适合迁 Rust

当前项目已经具备比较适合迁移的边界：

- `pipeline.py` 负责阶段编排、进度、产物和元数据
- `raster_algorithms.py` 负责纯算法

也就是说，Python 侧已经把“调度逻辑”和“计算逻辑”拆开了。这样 Rust 不需要接管整个后端，只需要接管 `raster_algorithms.py` 里的热点函数。

## 推荐技术栈

第一阶段建议使用：

- Rust stable
- `PyO3`
- `maturin`
- `numpy` 绑定

原因：

- 能直接给 Python 暴露模块
- 适配当前 FastAPI / Python 架构
- 迁移路径平滑
- 结果验证方便

## 建议目录结构

建议新增：

```text
crates/
  river-kernel/
    Cargo.toml
    src/
      lib.rs
      d8.rs
      accumulation.rs
      sink_fill.rs
      mask.rs
      types.rs
```

Python 侧保持：

```text
apps/backend/app/
  pipeline.py
  raster_algorithms.py
  rust_bridge.py
```

其中：

- `rust_bridge.py` 负责统一调用 Rust 扩展模块
- `raster_algorithms.py` 负责保留 Python 备份实现和切换逻辑

## 第一阶段迁移范围

### 优先级 1：严格 D8

最先迁移：

- 严格下降方向计算
- 不含平坡补流向
- 输入：`height_array`, `valid_mask`
- 输出：`direction_array`

这是第一步最合适的切入点，因为：

- 逻辑独立
- 输入输出清晰
- 容易和 Python 结果逐像素对比
- 性能收益明显

### 优先级 2：汇流累积

迁移内容：

- 下游索引构建
- 入度统计
- 拓扑传播累积

这一步同样是大规模遍历，非常适合 Rust。

### 优先级 3：局部填洼与封闭平坡修复

迁移内容：

- 局部填洼
- 封闭平坡区域扫描
- 区域修复与微坡度施加

这一步复杂度更高，但收益也很大。

### 优先级 4：自动 Mask

迁移内容：

- 梯度分析
- 局部方差
- 边界连通
- 小连通域过滤

优先级比前三项略低，但很适合后续补齐。

## 暂时不要迁的部分

第一阶段不建议迁移：

- FastAPI 路由
- 任务系统
- 日志系统
- 产物管理
- 元数据写入
- 前端交互

这些部分用 Python 继续维护更高效。

## Python / Rust 交互设计

建议 Python 侧定义统一桥接函数，例如：

```python
def compute_d8_flow_directions_bridge(
    height_array: np.ndarray,
    valid_mask: np.ndarray,
    use_rust: bool,
    **kwargs: object,
) -> np.ndarray:
    ...
```

内部逻辑：

- `use_rust=False`：调用现有 Python 实现
- `use_rust=True`：调用 Rust 扩展实现

这样可以做到：

- A/B 对比
- 随时回退
- 逐阶段迁移

## 进度反馈策略

迁 Rust 时必须保留“任务没有卡死”的体验。

建议策略：

### 第一阶段

Rust 负责纯计算，Python 负责阶段级进度。

也就是：

- 进入严格 D8 阶段：Python 汇报阶段开始
- Rust 完成一整块 / 一整步后返回
- Python 汇报阶段完成

这种方式实现最简单，但阶段内细粒度进度较少。

### 第二阶段

Rust 支持分块回调或周期性进度回报。

例如：

- 每处理 N 行调用一次 Python 回调
- 或每处理一个 tile 返回一次已完成计数

这样才能保留现在前端那种“当前子步骤持续跳动”的体验。

## 一致性验证策略

每迁一个函数，都必须做双实现对比。

建议验证方式：

1. 小尺寸单元测试
   - 人工构造数组
   - 逐元素比较 Python / Rust 结果

2. 中尺寸回归样例
   - 跑一批真实高程图
   - 比较方向图、累积图、河道图差异

3. 性能基准
   - 记录 Python / Rust 的耗时
   - 记录峰值内存

## Rust 第一版接口建议

建议先暴露这些函数：

```rust
fn compute_strict_d8(
    height: PyReadonlyArray2<f32>,
    valid_mask: PyReadonlyArray2<bool>,
) -> Py<PyArray2<i8>>;

fn compute_flow_accumulation(
    direction: PyReadonlyArray2<i8>,
    valid_mask: PyReadonlyArray2<bool>,
) -> Py<PyArray2<f32>>;
```

注意：

- Python 侧的平坡补流向先保留
- 先不要一开始把加权平坡逻辑一起搬过去

这样能最大限度降低第一步风险。

## 推荐实施顺序

### 第 1 步

创建 `crates/river-kernel` 并接通 `PyO3 + maturin`

目标：

- 能在 Python 里 `import river_kernel`

### 第 2 步

实现 Rust 版严格 D8

目标：

- 和 Python 版严格 D8 对齐
- 先不替换平坡逻辑

### 第 3 步

在 Python 里加入 `use_rust_kernel` 开关

目标：

- 可切换 Python / Rust 实现

### 第 4 步

实现 Rust 版汇流累积

### 第 5 步

评估是否迁移：

- 局部填洼
- 封闭平坡修复
- 自动 Mask

## 风险

主要风险有：

1. Rust / Python 数组布局不一致
2. `bool` mask 与 `uint8` mask 转换不一致
3. 回调进度实现不当，导致前端体感仍像卡死
4. 平坡和加权流向逻辑过早迁移，导致调试困难

所以第一阶段必须坚持：

- 先迁严格 D8
- 再迁汇流累积
- 保留 Python 参考实现

## 当前建议

下一步最稳的动作是：

1. 新建 `crates/river-kernel`
2. 接通 `PyO3`
3. 先实现 Rust 版严格 D8
4. 在 Python 中保留可切换开关

这是最小、最稳、收益也足够明显的第一步。
