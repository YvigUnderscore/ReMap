import type { CameraPose, JobLogEntry, JobStatus, ProcessingSettings, SettingsSchema } from './types'

export class RemapApi {
  constructor(
    private readonly baseUrl: string,
    private readonly apiKey: string,
  ) {}

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
        ...(init?.headers ?? {}),
      },
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || `HTTP ${response.status}`)
    }

    return (await response.json()) as T
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return response.json() as Promise<{ status: string; timestamp: string }>
  }

  settingsSchema() {
    return this.request<SettingsSchema>('/settings/schema')
  }

  async uploadDataset(file: File) {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${this.baseUrl}/upload`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.apiKey}` },
      body: formData,
    })

    if (!response.ok) {
      throw new Error(await response.text())
    }

    return response.json() as Promise<{ dataset_id: string }>
  }

  startProcessing(payload: {
    dataset_id: string
    settings: ProcessingSettings
    input_colorspace?: string
    output_colorspace?: string
  }) {
    return this.request<{ job_id: string; status: string; message: string }>('/process', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  }

  status(jobId: string) {
    return this.request<JobStatus>(`/jobs/${jobId}/status`)
  }

  logs(jobId: string) {
    return this.request<{ job_id: string; log: JobLogEntry[] }>(`/jobs/${jobId}/logs`)
  }

  jobs() {
    return this.request<{ jobs: JobStatus[] }>('/jobs')
  }

  cancel(jobId: string) {
    return this.request<{ job_id: string; status: string }>(`/jobs/${jobId}/cancel`, {
      method: 'POST',
    })
  }

  visualizerCameras(jobId: string) {
    return this.request<{ job_id: string; cameras: CameraPose[] }>(`/jobs/${jobId}/visualizer/cameras`)
  }

  async downloadResult(jobId: string) {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}/result`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    })
    if (!response.ok) {
      throw new Error(await response.text())
    }
    return response.blob()
  }
}
