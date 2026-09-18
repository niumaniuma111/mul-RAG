<script setup lang="ts">
/**
 * 上传对话框：选文件 → POST /files/upload → 返回 task_id 交给父组件轮询。
 * 限制：pptx / ppt / pdf / png，≤200MB（与 nginx client_max_body_size 对齐）。
 * ⚠️ png 是**图片输入**：MinerU flash 档会把它当"整页图"转码成 jpg、不提取文字，
 *    内容完全依赖下游 VLM caption（见 configs/prompts/caption_system.txt）。
 *    所以一张 png 最终只产出 1 个 image_caption chunk，且入库比文档慢（要跑 VLM）。
 */
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import type { UploadFile } from 'element-plus'
import { uploadFile } from '../api/files'
import type { UploadResult } from '../types/api'

const emit = defineEmits<{ uploaded: [result: UploadResult] }>()

const visible = defineModel<boolean>({ default: false })
const uploading = ref(false)

async function onChange(uploadFileRef: UploadFile) {
  const raw = uploadFileRef.raw
  if (!raw) return
  if (raw.size > 200 * 1024 * 1024) {
    ElMessage.error('文件超过 200MB 上限')
    return
  }
  uploading.value = true
  try {
    const result = await uploadFile(raw)
    ElMessage.success(`上传成功：${result.is_new_doc ? '新文档' : '新版本'} v${result.version}，开始解析`)
    visible.value = false
    emit('uploaded', result)
  } catch (err) {
    ElMessage.error((err as Error).message)
  } finally {
    uploading.value = false
  }
}
</script>

<template>
  <el-dialog v-model="visible" title="上传文档" width="480px" align-center>
    <el-upload
      drag
      :auto-upload="false"
      :limit="1"
      accept=".pptx,.ppt,.pdf,.png"
      :on-change="onChange"
      :show-file-list="false"
      :disabled="uploading"
    >
      <el-icon :size="42" color="#c0c4cc"><UploadFilled /></el-icon>
      <div class="el-upload__text">拖拽文件到这里，或 <em>点击选择</em></div>
      <template #tip>
        <div class="el-upload__tip">支持 .pptx / .ppt / .pdf / .png，不超过 200MB；同名文件将自动产生新版本</div>
      </template>
    </el-upload>
    <div v-if="uploading" class="uploading">
      <el-progress :percentage="90" :indeterminate="true" :duration="3" status="warning" />
      <p>上传中，请勿关闭…</p>
    </div>
  </el-dialog>
</template>

<style scoped>
.uploading {
  margin-top: 12px;
  text-align: center;
  color: #909399;
  font-size: 12px;
}
</style>
