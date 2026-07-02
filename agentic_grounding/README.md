# Agentic zero-shot 3D grounding

这个目录是在 MVGGT 仓库中独立实现的一条零样本 3DVG 研究链路。它不修改训练代码，也不要求先把所有外部模型塞进同一环境。

核心原则是：**VLM 负责把语言编译成查询图、选择工具和核验语义；距离、方位、跨视角合并由可复现的几何工具计算。** 不让 VLM 直接目测三维距离。

## 数据流

```text
referring expression
  -> QueryCompiler -> QuerySpec(target, anchors, predicates, frames)
multi-view RGB
  -> MVGGT / VGGT / VGGT-Omega -> depth, cameras, shared point maps
  -> SAM3 concept masks OR GroundingDINO boxes + SAM masks
  -> LIFT(mask) -> per-view object point clouds
  -> cross-view association -> persistent Object Registry (O001, O002, ...)
  -> FactorGraphSolver + MEASURE
  -> deterministic planner or VLM tool planner
  -> VERIFY / STOP -> selected Object ID and its masks
```

`left/right/front/behind` 必须使用指定参考视角；`above/below/on` 使用重力轴；`near/closest_to` 使用共享三维坐标。关系定义集中在 `spatial/relations.py`，便于消融。

## 目录职责

- `query/`: Query Compiler 与严格 JSON schema。
- `geometry/`: 当前 MVGGT、官方 VGGT、官方 VGGT-Omega 后端。
- `perception/`: SAM3、SAM3 HTTP、GroundingDINO、经典 SAM box prompt。
- `fusion/`: mask lifting、跨视角实例关联、持久 Object ID。
- `spatial/`: 坐标系、确定性关系函数、查询图求解。
- `visualization/`: BEV、Set-of-Mark 和候选 crop cards。
- `tools/`: `LOOK / PROPAGATE / MEASURE / COMPARE / VERIFY`。
- `agent/`: 可解释规则 planner 与 VLM planner。
- `evaluation/`: mIoU、Oracle@K、跨视角关联指标。

## 先准备哪些外部仓库和权重

这些适配器只读本地路径，不自动下载权重。

### 1. 三维重建，任选一个

- 当前仓库的 MVGGT：无需额外 clone，`MVGGTBackend` 接受已经构造和加载权重的模型。
- [官方 VGGT](https://github.com/facebookresearch/vggt)：clone 到仓库外或 `third_party/`，执行 `pip install -e <vggt-path>`，再自行下载 `facebook/VGGT-1B` 到本地。
- [官方 VGGT-Omega](https://github.com/facebookresearch/vggt-omega)：申请 Hugging Face checkpoint 权限后，clone 并 `pip install -e <vggt-omega-path>`，下载 `VGGT-Omega-1B-512` 的 `model.pt`。

VGGT/VGGT-Omega 的尺度通常不是绝对米制。本实现对 `closest/farthest` 没有影响；固定米阈值的 `near` 应先做场景尺度归一化或按物体/场景对角线定义阈值。

### 2. 二维候选，任选一条

**推荐首轮实验：GroundingDINO + 经典 SAM。** 它容易在当前环境中跑通：

- 将 [GroundingDINO 的 Hugging Face 模型](https://huggingface.co/IDEA-Research/grounding-dino-base) 下载到本地，交给 `HuggingFaceGroundingDINO`。
- clone [Segment Anything](https://github.com/facebookresearch/segment-anything)，安装包并下载对应 `vit_h/vit_l/vit_b` checkpoint，交给 `SAMBoxSegmenter`。

**SAM3 路线：独立环境。** [官方 SAM3](https://github.com/facebookresearch/sam3) 当前依赖比本仓库更新的 Python/PyTorch，建议单独建环境并运行：

```powershell
python -m agentic_grounding.services.sam3_server --host 127.0.0.1 --port 8765
```

主环境使用 `HTTPConceptSegmenter("http://127.0.0.1:8765/segment")`。如果两个环境不共享本仓库，把 `agentic_grounding/` 以 editable package 暴露给 SAM3 环境，或复制 service 与 adapter 文件。

## 离线 Query Compiler

输入是 annotation record 的 JSON list；默认从 `description` 读取文本。服务只需兼容 OpenAI Chat Completions，例如 vLLM/SGLang 或云端 VLM：

```powershell
python -m agentic_grounding.scripts.compile_queries `
  --input data/annotations.json `
  --output cache/compiled_queries.json `
  --base-url http://127.0.0.1:8000 `
  --model your-local-vlm
```

脚本每 20 条写盘，并按 query ID 从已有输出续跑。输出中的 `compiled_query` 可在所有后续实验复用，避免 Query Compiler 成为在线开销和随机变量。

一个目标查询会被编译为：

```json
{
  "target": {"category": "chair", "attributes": ["wooden", "brown"]},
  "anchors": [
    {"anchor_id": "a0", "category": "table", "attributes": []},
    {"anchor_id": "a1", "category": "window", "attributes": []}
  ],
  "predicates": [
    {"op": "left_of", "subject": "target", "object": "a0", "frame": "view_dependent", "confidence": 1.0, "hard": true, "reference_view": 2},
    {"op": "closest_to", "subject": "target", "object": "a1", "frame": "world", "confidence": 1.0, "hard": true, "reference_view": null}
  ]
}
```

## 最小构建示例

```python
from agentic_grounding.geometry import VGGTOmegaBackend
from agentic_grounding.perception import HuggingFaceGroundingDINO, SAMBoxSegmenter
from agentic_grounding.pipeline import RegistryBuilder

geometry = VGGTOmegaBackend("ckpts/VGGT-Omega-1B-512/model.pt")
detector = HuggingFaceGroundingDINO("ckpts/grounding-dino-base")
segmenter = SAMBoxSegmenter("ckpts/sam_vit_h_4b8939.pth", model_type="vit_h")
builder = RegistryBuilder(
    geometry_backend=geometry,
    detector=detector,
    box_segmenter=segmenter,
)
artifacts = builder.build(image_paths, compiled_query)
print(artifacts.registry.to_summary())
```

然后用 `SpatialRelationEngine(artifacts.geometry)` 和 `FactorGraphSolver` 得到候选排序。第一版建议先用 `DeterministicPlanner` 做无 LLM-agent 的强基线，再换 `VLMToolPlanner`；这样能分别测出 Query Compiler、工具规划和视觉语义核验的贡献。

## 与 TAB 的关系

不建议直接在 TAB 上重写主干：TAB 更适合作为 RGB-D/顺序观测设置下的对照和工具设计参考，而这里的关键变量是无深度多视图重建、跨视角 Object ID 与显式坐标系。可以复用它的 agent loop、记忆形式或提示词思想，但保持本目录的数据层和几何层独立，消融会更清楚。

## 推荐的首轮实验顺序

1. 用少量 ScanNet/ScanRefer 场景缓存重建结果，人工检查相机和点云方向。
2. 跑 GroundingDINO+SAM，测单视角 mask recall 与跨视角 association precision/recall。
3. 用真值解析 query 验证 `MEASURE + FactorGraphSolver`，再接 Query Compiler。
4. 报告 deterministic planner，再报告 VLM planner；额外给 Oracle@K，区分候选召回问题和推理问题。
5. 最后替换 VGGT/MVGGT/Omega 与 SAM/SAM3，做模块化消融。
