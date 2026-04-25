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
}

export interface SystemCapabilities {
  cpu_count: number;
  glomap_available: boolean;
  ffmpeg_available: boolean;
  ffprobe_available: boolean;
  openimageio_available: boolean;
  torch_available: boolean;
  cuda_available: boolean;
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
  health: {
    reachable: boolean;
    url: string;
    payload?: Record<string, unknown>;
    error?: string;
  };
  remote_jobs: Array<Record<string, unknown>>;
}
