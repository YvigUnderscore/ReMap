export type InputMode = "video" | "images" | "rescan";

export interface ProcessingJobRequest {
  input_mode: InputMode;
  input_paths: string[];
  output_path: string;
  fps_extract: number;
  force_16bit: boolean;
  camera_model: string;
  feature_type: string;
  matcher_type: string;
  max_keypoints: number;
  pairing_mode: string;
  mapper_type: string;
  stray_approach: string;
  stray_confidence: number;
  stray_depth_subsample: number;
  stray_gen_pointcloud: boolean;
  color_enabled: boolean;
  color_source: string;
  color_dest: string;
  detected_color_profile: string;
  ocio_path: string;
  ocio_in_cs: string;
  ocio_out_cs: string;
  keep_srgb_png: boolean;
  use_acescg_exr: boolean;
  num_workers: number;
  server_port: number;
  server_api_key: string;
  label: string;
  skip_existing: boolean;
  quality_sweep: boolean;
  sweep_sample_frames: number;
  exclude_blurry: boolean;
  exclude_black: boolean;
  blur_threshold: number;
  black_threshold: number;
  ram_limit_percent: number;
  gpu_vram_limit_percent: number;
}

export interface JobLogEvent {
  id: number;
  timestamp: string;
  level: string;
  message: string;
}

export interface JobSummary {
  job_id: string;
  status: string;
  progress: number;
  current_step: string;
  created_at: string;
  updated_at: string;
  label: string;
  output_path: string;
  input_mode: InputMode;
  error?: string | null;
  queue_position?: number | null;
  progress_note: string;
  eta_seconds?: number | null;
}

export interface JobDetail extends JobSummary {
  request: ProcessingJobRequest;
  logs: JobLogEvent[];
}

export interface JobArtifactItem {
  path: string;
  name: string;
  kind: string;
  exists: boolean;
  size?: number | null;
  modified_at?: number | null;
  extension: string;
  previewable: boolean;
  samples?: JobArtifactItem[];
}

export interface JobArtifacts {
  job_id: string;
  input_paths: JobArtifactItem[];
  output_path: JobArtifactItem;
  frame_dirs: JobArtifactItem[];
  frames: JobArtifactItem[];
  exrs: JobArtifactItem[];
  reconstruction: JobArtifactItem[];
  latest_outputs: JobArtifactItem[];
}

export interface ServerConfig {
  host: string;
  port: number;
  api_key: string;
  auto_start: boolean;
  output_dir: string;
}

export interface AppSettings {
  schema_version: number;
  theme: string;
  defaults: ProcessingJobRequest;
  server: ServerConfig;
  recent_inputs: string[];
  recent_outputs: string[];
  notifications_enabled: boolean;
  system_notifications: boolean;
  cache_enabled: boolean;
  cache_max_size_gb: number;
  ram_limit_percent: number;
  gpu_vram_limit_percent: number;
  blur_threshold: number;
  black_threshold: number;
}

export interface SystemCapabilities {
  cpu_count: number;
  glomap_available: boolean;
  ffmpeg_available: boolean;
  ffprobe_available: boolean;
  openimageio_available: boolean;
  torch_available: boolean;
  cuda_available: boolean;
  loma_available: boolean;
  triton_available: boolean;
  torch_device_name: string;
  torch_compute_capability: string;
  python_version: string;
  platform: string;
  ocio_env: string;
}

export interface OptionsPayload {
  color_sources: string[];
  color_destinations: string[];
  ocio_configs: string[];
  ocio_spaces: string[];
  default_ocio_config: string;
  features: string[];
  matchers: string[];
  pairing_modes: string[];
  camera_models: string[];
  mapper_types: string[];
}

export interface CapabilitiesPayload extends SystemCapabilities, OptionsPayload {}

export interface ProbeItem {
  path: string;
  name: string;
  kind: string;
  duration?: number;
  native_fps: number;
  total_frames: number;
  estimated_frames: number;
  color_profile?: string;
  step?: number;
  valid: boolean;
}

export interface ProbeResponse {
  input_mode: InputMode;
  items: ProbeItem[];
  detected_color_profile: string;
  max_native_fps: number;
}

export interface ServerState {
  config: ServerConfig;
  running: boolean;
  log_path: string;
  local_ip?: string;
  connect_url?: string;
  health: {
    reachable: boolean;
    url: string;
    payload?: Record<string, unknown>;
    error?: string;
  };
  remote_jobs: Array<Record<string, unknown>>;
}

export interface AnalyticsRecentJob {
  job_id: string;
  label: string;
  status: string;
  progress: number;
  matcher_type?: string;
  feature_type?: string;
  input_mode: string;
  frames: number;
  features: number;
  matches: number;
  pairs: number;
  points3d?: number | null;
  output_size_mb: number;
  duration_seconds?: number | null;
  step_seconds: Record<string, number>;
  feature_ms_per_frame?: number | null;
  match_ms_per_pair?: number | null;
  updated_at: string;
}

export interface AnalyticsPayload {
  generated_at: string;
  system: Record<string, number | string | boolean | null>;
  gpu: Record<string, number | string | boolean | null>;
  maxima: Record<string, number | null>;
  jobs: {
    total: number;
    active: number;
    completed: number;
    failed: number;
    cancelled: number;
    by_status: Record<string, number>;
    recent: AnalyticsRecentJob[];
  };
  usage: {
    matchers: Record<string, number>;
    features: Record<string, number>;
    input_modes: Record<string, number>;
  };
  throughput: Record<string, number | null>;
}

export interface JobEstimateItem {
  input_mode: InputMode;
  items: ProbeItem[];
  frames: number;
  pairs: number;
  estimated_disk_bytes: number;
  estimated_seconds: number;
  confidence: string;
  history_samples: number;
  breakdown: Record<string, number>;
  warnings: string[];
}

export interface EstimatePayload {
  estimates: JobEstimateItem[];
  total_frames: number;
  total_pairs: number;
  total_disk_bytes: number;
  total_seconds: number;
  warnings: string[];
}

export interface CacheEntry {
  key: string;
  path: string;
  size: number;
  size_mb: number;
  modified_at: number;
  files: string[];
  request: Record<string, unknown>;
  created_at: string;
}

export interface CacheStatus {
  path: string;
  entries: CacheEntry[];
  total_size: number;
  total_size_mb: number;
}

export interface ReconstructionPreview {
  available: boolean;
  sparse_dir: string;
  cameras: Array<Record<string, unknown>>;
  images: Array<{
    image_id: number;
    name: string;
    camera_id: number;
    t: number[];
    q: number[];
    num_points2d: number;
  }>;
  points: {
    count: number;
    sample: Array<{ id: number; xyz: number[]; rgb: number[]; error: number }>;
    mean_error?: number | null;
  };
  stats: {
    camera_count: number;
    image_count: number;
    registered_images: number;
    point_count: number;
    mean_reprojection_error?: number | null;
  };
}

export interface DependencyPackage {
  name: string;
  installed: boolean;
  version: string;
  direct_url: string;
}

export interface DependencyTaskLog {
  timestamp: string;
  message: string;
}

export interface DependencyTaskState {
  running: boolean;
  action: string;
  status: string;
  started_at?: string | null;
  finished_at?: string | null;
  returncode?: number | null;
  message: string;
  log: DependencyTaskLog[];
}

export interface DependencyModelCacheItem {
  name: string;
  path: string;
  size_mb: number;
  modified_at: number;
}

export interface DependencyStatus {
  python: string;
  requirements: string;
  requirements_lock: string;
  loma_locked_url: string;
  packages: DependencyPackage[];
  models: {
    torch_hub_dir: string;
    torch_hub_checkpoints: DependencyModelCacheItem[];
    superglue_weights: DependencyModelCacheItem[];
    total_size_mb: number;
  };
  task: DependencyTaskState;
}
