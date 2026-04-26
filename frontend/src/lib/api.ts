import type {
  AppSettings,
  AnalyticsPayload,
  CapabilitiesPayload,
  CacheStatus,
  DependencyStatus,
  EstimatePayload,
  JobArtifacts,
  JobDetail,
  ReconstructionPreview,
  JobSummary,
  OptionsPayload,
  ProcessingJobRequest,
  ProbeResponse,
  ServerState,
} from "./types";

const BASE_URL = import.meta.env.VITE_REMAP_BACKEND_URL ?? "http://127.0.0.1:8765";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(errorBody || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  getSettings: () => request<AppSettings>("/internal/v1/settings"),
  saveSettings: (payload: Partial<AppSettings>) =>
    request<AppSettings>("/internal/v1/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  getCapabilities: () =>
    request<CapabilitiesPayload>("/internal/v1/system/capabilities"),
  getAnalytics: () => request<AnalyticsPayload>("/internal/v1/analytics"),
  getDependencies: () => request<DependencyStatus>("/internal/v1/dependencies"),
  runDependencyAction: (action: string) =>
    request<DependencyStatus>("/internal/v1/dependencies/actions", {
      method: "POST",
      body: JSON.stringify({ action }),
    }),
  getOptions: (ocioPath?: string) =>
    request<OptionsPayload>(
      `/internal/v1/options${ocioPath ? `?ocioPath=${encodeURIComponent(ocioPath)}` : ""}`,
    ),
  probeInputs: (payload: Partial<ProcessingJobRequest>) =>
    request<ProbeResponse>("/internal/v1/probe", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  estimate: (payloads: Partial<ProcessingJobRequest>[]) =>
    request<EstimatePayload>("/internal/v1/estimate", {
      method: "POST",
      body: JSON.stringify({ requests: payloads }),
    }),
  getJobs: async () => {
    const data = await request<{ jobs: JobSummary[] }>("/internal/v1/jobs");
    return data.jobs;
  },
  getJob: (jobId: string) => request<JobDetail>(`/internal/v1/jobs/${jobId}`),
  createJob: (payload: Partial<ProcessingJobRequest>) =>
    request<JobDetail>("/internal/v1/jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  createJobsBatch: async (payloads: Partial<ProcessingJobRequest>[]) => {
    const data = await request<{ jobs: JobDetail[] }>("/internal/v1/jobs/batch", {
      method: "POST",
      body: JSON.stringify({ requests: payloads }),
    });
    return data.jobs;
  },
  cancelJob: (jobId: string) =>
    request<JobDetail>(`/internal/v1/jobs/${jobId}/cancel`, {
      method: "POST",
      body: JSON.stringify({}),
    }),
  deleteJob: (jobId: string) =>
    request<{ deleted: boolean; job_id: string }>(`/internal/v1/jobs/${jobId}`, {
      method: "DELETE",
    }),
  clearQueuedJobs: () =>
    request<{ removed: string[]; count: number }>("/internal/v1/jobs/queue", {
      method: "DELETE",
    }),
  getJobArtifacts: (jobId: string) =>
    request<JobArtifacts>(`/internal/v1/jobs/${jobId}/artifacts`),
  getJobReconstruction: (jobId: string) =>
    request<ReconstructionPreview>(`/internal/v1/jobs/${jobId}/reconstruction`),
  getCache: () => request<CacheStatus>("/internal/v1/cache"),
  clearCache: () =>
    request<{ cleared: boolean; removed_entries: number; removed_size: number }>("/internal/v1/cache", {
      method: "DELETE",
    }),
  fileUrl: (path: string) =>
    `${BASE_URL}/internal/v1/files?path=${encodeURIComponent(path)}`,
  pauseJob: (jobId: string) =>
    request<JobDetail>(`/internal/v1/jobs/${jobId}/pause`, {
      method: "POST",
      body: JSON.stringify({}),
    }),
  resumeJob: (jobId: string) =>
    request<JobDetail>(`/internal/v1/jobs/${jobId}/resume`, {
      method: "POST",
      body: JSON.stringify({}),
    }),
  getServerState: () => request<ServerState>("/internal/v1/server"),
  updateServerConfig: (payload: Partial<ServerState["config"]>) =>
    request<{ config: ServerState["config"]; state: ServerState }>("/internal/v1/server", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  serverAction: (action: "start" | "stop" | "refresh") =>
    request<ServerState>("/internal/v1/server", {
      method: "POST",
      body: JSON.stringify({ action }),
    }),
  jobLogsStreamUrl: (jobId: string) =>
    `${BASE_URL}/internal/v1/jobs/${jobId}/logs/stream`,
};
