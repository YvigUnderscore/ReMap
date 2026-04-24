export type InputMode = 'video' | 'images' | 'rescan'

export interface ProcessingSettings {
  fps: number
  feature_type: string
  matcher_type: string
  max_keypoints: number
  camera_model: string
  mapper_type: string
  stray_approach: string
  pairing_mode: string
  num_threads: number
  stray_confidence: number
  stray_depth_subsample: number
  stray_gen_pointcloud: boolean
  input_colorspace?: string
  output_colorspace?: string
}

export interface SettingsSchema {
  input_modes: string[]
  defaults: ProcessingSettings
  options: Record<string, string[] | number[]>
  supported_colorspaces: string[]
}

export interface RuntimeInfo {
  platform: string
  isDev: boolean
  running: boolean
  port: number
  apiKey: string
  startedAt: string
  baseUrl: string
}

export interface JobStatus {
  job_id: string
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled'
  progress: number
  current_step: string
  created_at: string
  updated_at: string
  error?: string | null
  settings?: Record<string, unknown>
}

export interface JobLogEntry {
  time: string
  message: string
}

export interface CameraPose {
  image_id: number
  name: string
  world_from_cam: number[][]
}
