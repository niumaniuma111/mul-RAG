<script setup lang="ts">
/**
 * 文档范围选择器：勾选哪些文档参与本轮问答检索。
 * 空 = 不传 doc_ids = 检索用户全部生效版本。
 */
import { computed } from 'vue'
import { useDocumentsStore } from '../stores/documents'

const docs = useDocumentsStore()

const list = computed(() => docs.readyDocs())
const summary = computed(() =>
  docs.selectedDocIds.length === 0 ? '全部文档' : `已选 ${docs.selectedDocIds.length} 篇`,
)
</script>

<template>
  <el-card shadow="never" class="panel">
    <template #header>
      <div class="head">
        <span>检索范围</span>
        <el-tag size="small" effect="plain">{{ summary }}</el-tag>
      </div>
    </template>
    <el-alert
      v-if="list.length === 0"
      type="info"
      :closable="false"
      title="暂无已解析完成的文档"
      description="先到「文档管理」上传并等待解析成功。"
    />
    <el-checkbox
      v-for="d in list"
      :key="d.doc_id"
      :model-value="docs.selectedDocIds.includes(d.doc_id)"
      class="doc-check"
      @change="docs.toggleSelect(d.doc_id)"
    >
      <span class="doc-name">{{ d.file_name }}</span>
      <el-tag size="small" type="info" effect="plain">v{{ d.active_version }}</el-tag>
    </el-checkbox>
    <el-button v-if="docs.selectedDocIds.length" link size="small" @click="docs.selectedDocIds = []">清空选择</el-button>
  </el-card>
</template>

<style scoped>
.panel {
  border-radius: 10px;
}
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.doc-check {
  display: flex;
  width: 100%;
  margin-bottom: 6px;
}
.doc-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 150px;
  display: inline-block;
  vertical-align: bottom;
}
</style>
