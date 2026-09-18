# ppt-agentic-rag

基于 **LangGraph** 的多模态 PPT 文档问答系统。离线把 PPT/PDF 解析入库，在线用
自省驱动的迭代检索生成带引用的回答。**全程离线推理**（不调用任何在线大模型 API），
跑在一张 8GB 显存的笔记本显卡 + 16GB 内存上。

> **架构定位**：LangGraph **workflow**（Routing + Evaluator-optimizer），
> 不是 ReAct Agent。自省机制对应 **Self-RAG 的 reflection tokens，但不微调模型**
> —— 用图节点把 `[Retrieve]` / `[IsREL]` / `[IsSUP]` 三个判定外置实现。
> 完整设计见 `docs/DESIGN.md`。

---

## 一、两条流水线（物理隔离）

```
离线入库（吃 CPU）    上传 → 版本登记 → MinerU 解析 → 语义切分 → 批量 embedding → 写 Qdrant + PG
在线问答（吃 GPU）    提问 → 检索决策 → 混合检索 → 评分 → [低分 HITL 打断] → 生成 → 自省 → [带新词重检]
```

两条线**物理隔离**是核心取舍：入库吃 CPU 不占显存，所以文档解析期间问答完全不受影响。

**检索链路**：BM25 稀疏 + 向量稠密各取 top6 → Qdrant 服务端 RRF 融合 → top8
→ BGE-reranker 精排 → **top3 进 prompt**。

**四重防死循环**：总检索上限 4 ｜ 自省轮次上限 2 ｜ 墙钟超时 120s ｜ LangGraph 递归上限 30。

---

## 二、三条硬约束（决定了架构，不是建议）

| 约束 | 数字 | 后果 |
|---|---|---|
| **显存** | 8GB 卡，实测可用约 6.9GiB（清理桌面后） | 单模型驻留 + VLM ⇄ reranker 强制互斥；生成态峰值 6.3GB |
| **内存** | 16GB 宿主，WSL2 分配 11GB | 容器限额合计 11.4GB，**不能保活模型**，换出策略必须是 `discard` |
| **磁盘** | 一切落 E 盘；模型走 ext4 | 放 `/mnt/*` 会走 9P 桥（吞吐仅 5-6%），破坏 6-10s 权重重载假设 |

> ⚠️ **跑之前先清桌面 GPU 占用。** 实测桌面应用（Edge / 动态壁纸 / 音频增强等）
> 会占掉 0.8-1.7GiB。`nvidia-smi` 显示可用 **< 6.5GiB** 时按本文末的清单清理一遍。
> 这是唯一一条「不做就跑不起来」的运行约束，且它**不是**硬件属性，取决于你当下开了什么。

---

## 三、从哪里执行（2026-09-17 实测更正）

> ⚠️ 本节此前写的是「必须从 WSL 内跑，Windows 侧会失败」。**该断言已被实测推翻** ——
> 真正的门槛是 **Docker Desktop 的 WSL Integration 开关**，不是 Windows 本身。

项目代码与模型权重都放在 **WSL 发行版内部的 ext4**（`~/projects/ppt-agentic-rag`），
物理上仍在 E 盘（`ext4.vhdx`），C 盘零占用。
**这一条不受下面更正影响，仍是既定决策。**

### 三种执行方式（实测结论）

| 方式 | 前置条件 | 结论 |
|---|---|---|
| **WSL 内** | 无 | ✅ **推荐** —— 路径原生，零转发 |
| **Windows 侧**（UNC） | 开 WSL Integration | ✅ **可行** —— 相对路径解析成 `\\wsl.localhost\...`，挂载与构建均实测通过 |
| Windows 侧 + `/mnt/e` 路径 | 无 | ❌ 走 9P 桥，吞吐仅原生 ext4 的 5~6% |

```bash
# ✅ 推荐
cd ~/projects/ppt-agentic-rag && docker compose up -d
```

未开 WSL Integration 时从 Windows 侧跑，报错为：

```
accessing specified distro mount service:
stat /run/guest-services/distro-services/ubuntu-22-04.sock: no such file or directory
```

开启方式：**Docker Desktop → Settings → Resources → WSL Integration → 勾选 `Ubuntu-22.04`**。

### ⚠️ 性能口径（诚实标注，别过度解读）

**已测**（本机实测）：

- 100MB 连续写 —— UNC 侧 3.9GB/s vs 容器内层 1.2GB/s
- 512×64KB 小文件写 —— UNC 净 197ms vs 内层净 266ms

两项**均未测出 9P 惩罚**，UNC 侧甚至略优。

**未测**：GB 级顺序读（模型权重加载约 2.4GB / embedder 约 0.9GB）。
这才是 9P 最可能显形的场景。在补测之前，**不要**用上面两行断言"Windows 侧一样快"。

验证自己在正确的文件系统上：

```bash
df -T .        # Type 必须是 ext4；显示 9p 就是走错路径了
```


---

## 四、快速开始

```bash
# 0. 前置：WSL2 至少 11GB 内存（改 %USERPROFILE%\.wslconfig 后 wsl --shutdown）
#    确认 Docker Desktop 已启动

cd ~/projects/ppt-agentic-rag

# 1. 生成 .env 并改掉里面的 CHANGE_ME（尤其是 POSTGRES_PASSWORD / JWT_SECRET）
make env && ${EDITOR:-vi} .env

# 2. 构建并启动（首次约 25-45 分钟，主要吃在 MinerU 镜像上）
make up

# 3. 验收
make ps          # 全绿 healthy
make health      # 看各依赖状态
make gpu         # 显存基线：容器启动前后增量应 < 100MB（四个服务都走 CPU）

# 4. 建向量库集合
make init-qdrant

# 5. 下载模型权重（约 12GB，走 ModelScope；不下载也能起服务，只是问答不可用）
make models
```

前端（S4）：`docker compose --profile frontend up -d` → <http://localhost:8080>

---

## 五、目录结构（关键部分）

```
configs/config.yaml              ★ 全量超参唯一来源（含容器资源限额）
configs/prompts/*.txt            外置 Prompt 模板 —— 改 prompt 不用重建镜像
deploy/postgres/init/*.sql       建表 SQL（首次启动自动执行）
deploy/{backend,model-service,mineru,frontend}/Dockerfile
docker-compose.yml               默认（MinerU 走 CPU）
docker-compose.gpu.yml           叠加覆盖（MinerU 走 GPU）
backend/app/core/lock_factory.py ★ 锁层级自动推导（device → inproc/redis）
backend/app/main.py              FastAPI 入口（启动期做锁层级 fail-fast 自检）
backend/model_service/manager.py ★ 显存 LRU 换出（进程级卸载）
```

---

## 六、MinerU 的 CPU / GPU 双模式

**默认 CPU**：不占显存、不持锁、与在线问答真并行。50 页解析约 4-12 分钟。

**GPU 是可选的加速开关**：改一处 + 叠加一个文件即可切换。

```bash
# ① 改配置（只改一处）
#    .env → MINERU_DEVICE=cuda    或    configs/config.yaml → ingest.mineru.device: cuda

# ② 叠加启动
make up-gpu

# ③ 切回
make up
```

> ⚠️ **锁层级会自动跟随，不要手工配两处。** `lock_backend: auto` 会读 `device` 推导：
> `cpu → inproc`（只护 model-service 容器内的 VLM⇄reranker）、
> `cuda → redis`（跨容器锁，MinerU 重新参与显存竞争）。
> 配错的代价是**随机 OOM**（只在解析与问答恰好撞上时炸），所以启动期会 fail-fast 自检，
> 不匹配直接拒绝启动 —— 宁可起不来，也不要带着错的锁跑。

> ⚠️ **GPU 模式的用户可见代价**：从切到 GPU 那一刻起，上传文档后的 1-3 分钟内
> 所有问答请求会排队等锁。省下的是后台异步时间，付出的是交互路径上的阻塞
> —— 这正是它不做默认值的原因。

---

## 七、已经踩过的坑（改代码前先读）

1. **RRF 融合分与阈值不是一个量纲。** RRF 分只有 0.016~0.033，而 `score_threshold=0.35`
   是 rerank 的 sigmoid 分。混用会导致 HITL 要么永不触发、要么永远触发。
2. **「磁盘一份」≠「显存一份」。** `torch.load` 是进程级行为，两个容器共用同一份权重文件
   仍会各占一份显存 —— 这才是拆 `model-service` 的真正理由。
3. **`del + empty_cache()` 不释放 CUDA context。** context 占 0.6-0.9GB，只在进程退出时
   归还驱动。所以卸载必须是**进程级 kill**（`unload_mode: process`）。
4. **`/mnt/c` 与 `/mnt/e` 是同一条 9P 桥**，都慢。项目和模型都不能放那儿。
5. **`\\wsl.localhost` 是 UNC 路径，Windows Search 不索引它**；但别从它往 C 盘复制文件。
6. **不要把这个项目称作 ReAct Agent** —— 检索决策节点没有回边，`ToolNode`/`bind_tools`
   全项目 0 命中。它是单次规划 + 反思检索。
7. **"可用显存"是会漂移的量**，取决于你当下开了什么。任何余量结论都要标注桌面状态。
8. **改 `deploy/postgres/init/*.sql` 后必须 `make down-v` 重建卷** —— init 脚本只在
   数据卷为空时执行一次。

---

## 八、阶段进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| S0 | 设计文档（16 章 + 3 附录） | ✅ |
| **S1** | 骨架 + config + 建表 SQL + Dockerfile + compose | ✅ |
| S2 | 离线入库流水线（MinerU / model-service / chunker / Celery / 版本 API） | ⬜ |
| S3 | 在线问答 Agent（State / 7 节点 / 条件边 / SSE） | ⬜ |
| S4 | Vue3 前端（4 页面 / 8 组件 / useSSE） | ⬜ |
| S5 | 调优与文档（阈值标定 / 评测脚本） | ⬜ |

设计文档：`docs/DESIGN.md`（权威）｜ 一页纸骨架：`docs/FRAMEWORK.md`
