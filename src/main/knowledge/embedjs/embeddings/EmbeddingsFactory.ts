import type { BaseEmbeddings } from '@cherrystudio/embedjs-interfaces'
import { OllamaEmbeddings } from '@cherrystudio/embedjs-ollama'
import { OpenAiEmbeddings } from '@cherrystudio/embedjs-openai'
import type { ApiClient } from '@types'
import { net } from 'electron'

import { VoyageEmbeddings } from './VoyageEmbeddings'

export default class EmbeddingsFactory {
  /**
   * 创建 Embeddings 实例
   * @param embedApiClient - API 客户端配置
   * @param dimensions - 向量维度
   * @param sendDimensions - 是否在请求中发送 dimensions 参数（某些第三方 API 不支持）
   */
  static create({ embedApiClient, dimensions, sendDimensions = true }: { embedApiClient: ApiClient; dimensions?: number; sendDimensions?: boolean }): BaseEmbeddings {
    const batchSize = 10
    const { model, provider, apiKey, baseURL } = embedApiClient
    if (provider === 'voyageai') {
      return new VoyageEmbeddings({
        modelName: model,
        apiKey,
        outputDimension: dimensions,
        batchSize: 8
      })
    }
    if (provider === 'ollama') {
      return new OllamaEmbeddings({
        model: model,
        baseUrl: baseURL.replace(/\/api$/, ''),
        requestOptions: {
          // @ts-ignore expected
          'encoding-format': 'float'
        }
      })
    }
    // NOTE: Azure OpenAI 也走 OpenAIEmbeddings, baseURL是https://xxxx.openai.azure.com/openai/v1
    return new OpenAiEmbeddings({
      model,
      apiKey,
      dimensions: sendDimensions ? dimensions : undefined,  // 👈 条件传递
      batchSize,
      configuration: { baseURL, fetch: net.fetch as typeof fetch }
    })
  }
}
