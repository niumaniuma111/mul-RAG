<script setup lang="ts">
/**
 * 引用卡片（DESIGN §13.2）：按 chunk_type 三种渲染
 *   text          → 灰色引用块
 *   table         → v-html 渲染 table_html（保留合并单元格）
 *   image_caption → 图片（/storage 静态代理）+ caption
 * 统一页脚：文件名 · 第 N 页 · v{version}
 */
import { computed } from 'vue'
import type { PropType } from 'vue'
import type { Citation } from '../types/api'

// ⚠️ 这里必须用**运行时形式**声明 props。
//    类型参数形式 `defineProps<{ citation: Citation }>()` 依赖 @vue/compiler-sfc
//    在构建期解析 TS 类型；当 import type 与 defineProps 之间夹着注释块时，
//    该解析会**静默失败**，宏被原样保留 → 产物里出现裸的 `defineProps` →
//    运行时报 `ReferenceError: defineProps is not defined`，整个组件白屏。
//    运行时形式不依赖构建期类型解析，彻底规避这一类问题。
const props = defineProps({
  citation: { type: Object as PropType<Citation>, required: true },
})

const TYPE_LABELS: Record<string, string> = {
  text: '文本',
  table: '表格',
  image_caption: '图片',
}

const typeLabel = computed(() => TYPE_LABELS[props.citation.chunk_type || 'text'] || '片段')

/**
 * 把 image_path 归一成 nginx `/storage/` 下**真实可请求**的路径。
 *
 * 历史上存在三种形态：
 *   ① /storage/...                  → 已是公开路径，直接用
 *   ② /app/storage/...              → 容器内绝对路径（后端写入侧的标准形态），
 *                                     nginx 把宿主 ./storage 挂成 /storage 静态代理，
 *                                     剥掉 /app/storage 前缀即可命中
 *   ③ images/x.jpg                  → MinerU 原始相对路径（旧数据/越界回退），
 *                                     用 citation 自带的 doc_id + version 补全
 *
 * 🔴 三种都不匹配时必须返回 ''（配合 v-if 不渲染 el-image）。
 *    绝不能把不可用的路径透给浏览器 —— nginx 的 SPA fallback 会把 404 兜成
 *    index.html 并返回 **200 + text/html**，<img> 解码失败后才走 error 插槽，
 *    显示成「图片不可预览」，看起来像"格式不支持"，实际是路径错了，极难排查。
 */
const imageUrl = computed(() => {
  const p = (props.citation.image_path || '').trim()
  if (!p) return ''
  if (p.startsWith('/storage/')) return p
  if (p.startsWith('/app/storage/')) return p.replace('/app/storage/', '/storage/')
  const { doc_id: docId, version } = props.citation
  if (docId && version) {
    // 落盘约定：storage/parsed/{doc_id}/v{version}/images/xxx.jpg
    return `/storage/parsed/${docId}/v${version}/${p.replace(/^\.?\//, '')}`
  }
  return ''
})

const scorePct = computed(() =>
  props.citation.rerank_score != null ? Math.round(props.citation.rerank_score * 100) : null,
)
</script>

<template>
  <div :id="`citation-${citation.index}`" class="cite-card">
    <div class="cite-head">
      <el-tag size="small" effect="plain" type="info">{{ typeLabel }}</el-tag>
      <span class="cite-idx">[{{ citation.index }}]</span>
      <span v-if="scorePct != null" class="score">相关度 {{ scorePct }}%</span>
    </div>

    <!-- text -->
    <blockquote v-if="citation.chunk_type !== 'table' && citation.chunk_type !== 'image_caption'" class="snippet">
      {{ citation.snippet }}
    </blockquote>

    <!-- table：后端已存 HTML 原文，v-html 渲染保留合并单元格 -->
    <div
      v-else-if="citation.chunk_type === 'table' && citation.table_html"
      class="table-wrap"
      v-html="citation.table_html"
    />
    <blockquote v-else-if="citation.chunk_type === 'table'" class="snippet">{{ citation.snippet }}</blockquote>

    <!-- image -->
    <template v-else-if="citation.chunk_type === 'image_caption'">
      <el-image
        v-if="imageUrl"
        :src="imageUrl"
        :preview-src-list="[imageUrl]"
        fit="contain"
        class="cite-img"
        hide-on-click-modal
      >
        <template #error>
          <div class="img-fallback">图片不可预览</div>
        </template>
      </el-image>
      <blockquote class="snippet">{{ citation.snippet }}</blockquote>
    </template>

    <!-- 统一页脚：文件名 · 页码 · 版本 -->
    <div class="cite-foot">
      {{ citation.file_name || '未知文件' }}
      <template v-if="citation.page_num != null"> · 第 {{ citation.page_num + 1 }} 页</template>
      <template v-if="citation.version != null"> · v{{ citation.version }}</template>
    </div>
  </div>
</template>

<style scoped>
.cite-card {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 8px 12px;
  background: #fff;
  font-size: 13px;
}
.cite-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.cite-idx {
  font-weight: 600;
  color: var(--el-color-primary);
}
.score {
  margin-left: auto;
  font-size: 12px;
  color: #909399;
}
.snippet {
  margin: 0;
  padding: 8px 10px;
  background: #f5f7fa;
  border-left: 3px solid var(--el-color-primary-light-5);
  border-radius: 4px;
  color: #606266;
  line-height: 1.6;
  max-height: 140px;
  overflow-y: auto;
  white-space: pre-wrap;
}
.table-wrap {
  max-height: 240px;
  overflow: auto;
  border: 1px solid #ebeef5;
  border-radius: 4px;
}
/* 后端表格是裸 <table> HTML，样式在此补齐 */
.table-wrap :deep(table) {
  border-collapse: collapse;
  width: 100%;
  font-size: 12px;
}
.table-wrap :deep(td),
.table-wrap :deep(th) {
  border: 1px solid #dcdfe6;
  padding: 4px 8px;
}
.table-wrap :deep(th) {
  background: #f5f7fa;
}
.cite-img {
  max-height: 180px;
  border-radius: 4px;
}
.img-fallback {
  width: 160px;
  height: 90px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f7fa;
  color: #c0c4cc;
  font-size: 12px;
  border-radius: 4px;
}
.cite-foot {
  margin-top: 6px;
  font-size: 12px;
  color: #909399;
}
</style>
