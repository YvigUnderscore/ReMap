import { AnimatePresence, motion } from "framer-motion";
import { isPermissionGranted, requestPermission, sendNotification } from "@tauri-apps/plugin-notification";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./lib/api";
import type {
  AppSettings,
  CapabilitiesPayload,
  JobDetail,
  JobSummary,
  OptionsPayload,
  ProcessingJobRequest,
  ServerState,
} from "./lib/types";
import { AppShell, type AppView } from "./components/app-shell";
import { AnalyticsView } from "./components/views/analytics-view";
import { JobsView } from "./components/views/jobs-view";
import { NewJobWizard } from "./components/views/new-job-wizard";
import { ServerView } from "./components/views/server-view";
import { SettingsView } from "./components/views/settings-view";

const emptyDefaults: ProcessingJobRequest = {
  input_mode: "video",
  input_paths: [],
  output_path: "",
  fps_extract: 4,
  force_16bit: false,
  camera_model: "OPENCV",
  feature_type: "superpoint_aachen",
  matcher_type: "superpoint+lightglue",
  max_keypoints: 4096,
  pairing_mode: "Sequential (Video)",
  mapper_type: "COLMAP",
  stray_approach: "full_sfm",
  stray_confidence: 2,
  stray_depth_subsample: 2,
  stray_gen_pointcloud: true,
  color_enabled: false,
  color_source: "Auto-detect",
  color_dest: "ACEScg (EXR + sRGB PNG)",
  detected_color_profile: "",
  ocio_path: "",
  ocio_in_cs: "",
  ocio_out_cs: "",
  keep_srgb_png: true,
  use_acescg_exr: true,
  num_workers: 8,
  server_port: 5000,
  server_api_key: "",
  label: "",
  skip_existing: true,
  quality_sweep: false,
  sweep_sample_frames: 80,
  exclude_blurry: false,
  exclude_black: false,
  blur_threshold: 75,
  black_threshold: 0.08,
  ram_limit_percent: 90,
  gpu_vram_limit_percent: 92,
};

const emptyOptions: OptionsPayload = {
  color_sources: [],
  color_destinations: [],
  ocio_configs: [],
  ocio_spaces: [],
  default_ocio_config: "",
  features: [],
  matchers: [],
  pairing_modes: [],
  camera_models: [],
  mapper_types: [],
};

export default function App() {
  const [view, setView] = useState<AppView>("new-job");
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [capabilities, setCapabilities] = useState<CapabilitiesPayload | null>(null);
  const [options, setOptions] = useState<OptionsPayload>(emptyOptions);
  const [serverState, setServerState] = useState<ServerState | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [appError, setAppError] = useState<string>("");
  const [toasts, setToasts] = useState<Array<{ id: string; title: string; message: string; tone: string }>>([]);
  const seenStatusesRef = useRef<Map<string, string>>(new Map());
  const notificationsPrimedRef = useRef(false);

  async function loadInitial() {
    const jobsPromise = api.getJobs()
      .then((value) => {
        setJobs(value);
        if (!selectedJobId && value[0]) {
          setSelectedJobId(value[0].job_id);
        }
      })
      .catch((error) => setAppError(`Jobs: ${String(error)}`));

    const settingsPromise = api.getSettings()
      .then((value) => setSettings(value))
      .catch((error) => setAppError(`Settings: ${String(error)}`));

    const capabilitiesPromise = api.getCapabilities()
      .then((value) => {
        setCapabilities(value);
        setOptions({
          color_sources: value.color_sources,
          color_destinations: value.color_destinations,
          ocio_configs: value.ocio_configs,
          ocio_spaces: value.ocio_spaces,
          default_ocio_config: value.default_ocio_config,
          features: value.features,
          matchers: value.matchers,
          pairing_modes: value.pairing_modes,
          camera_models: value.camera_models,
          mapper_types: value.mapper_types,
        });
      })
      .catch((error) => setAppError(`Capabilities: ${String(error)}`));

    const serverPromise = api.getServerState()
      .then((value) => setServerState(value))
      .catch((error) => setAppError(`Server: ${String(error)}`));

    await Promise.allSettled([jobsPromise, settingsPromise, capabilitiesPromise, serverPromise]);
  }

  useEffect(() => {
    loadInitial().catch(console.error);
  }, []);

  function pushToast(title: string, message: string, tone: string) {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setToasts((current) => [...current, { id, title, message, tone }].slice(-5));
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 6500);
  }

  async function sendSystemNotification(title: string, body: string) {
    if (!settings?.system_notifications) {
      return;
    }
    try {
      let permissionGranted = await isPermissionGranted();
      if (!permissionGranted) {
        const permission = await requestPermission();
        permissionGranted = permission === "granted";
      }
      if (permissionGranted) {
        sendNotification({ title, body });
        return;
      }
    } catch {
      // Fall through to browser notifications in the web preview.
    }
    if (typeof Notification === "undefined") {
      return;
    }
    try {
      const permission = Notification.permission === "default"
        ? await Notification.requestPermission()
        : Notification.permission;
      if (permission === "granted") {
        new Notification(title, { body });
      }
    } catch {
      // Browser notifications are best-effort in the web preview.
    }
  }

  useEffect(() => {
    const next = new Map(seenStatusesRef.current);
    const terminal = new Set(["completed", "failed", "cancelled", "interrupted"]);
    for (const job of jobs) {
      const previous = seenStatusesRef.current.get(job.job_id);
      next.set(job.job_id, job.status);
      if (!notificationsPrimedRef.current || previous === job.status || !terminal.has(job.status)) {
        continue;
      }
      if (!settings?.notifications_enabled) {
        continue;
      }
      const title = job.status === "completed" ? "Job completed" : "Job stopped";
      const message = `${job.label || job.job_id}: ${job.status}`;
      pushToast(title, message, job.status === "completed" ? "success" : "danger");
      sendSystemNotification(title, message).catch(console.error);
    }
    seenStatusesRef.current = next;
    notificationsPrimedRef.current = true;
  }, [jobs, settings?.notifications_enabled, settings?.system_notifications]);

  useEffect(() => {
    const poll = window.setInterval(() => {
      api.getJobs()
        .then((jobsData) => {
          setJobs(jobsData);
          if (!selectedJobId && jobsData[0]) {
            setSelectedJobId(jobsData[0].job_id);
          }
        })
        .catch(console.error);
      api.getServerState().then(setServerState).catch(console.error);
    }, 3000);
    return () => window.clearInterval(poll);
  }, [selectedJobId]);

  async function createJobs(payloads: ProcessingJobRequest[]) {
    setSubmitting(true);
    setAppError("");
    try {
      const created = await api.createJobsBatch(payloads);
      const nextJobs = await api.getJobs();
      setJobs(nextJobs);
      if (created[0]) {
        setSelectedJobId(created[0].job_id);
      }
      setView("jobs");
    } catch (error) {
      setAppError(String(error));
    } finally {
      setSubmitting(false);
    }
  }

  async function cancelJob(jobId: string) {
    await api.cancelJob(jobId);
    const nextJobs = await api.getJobs();
    setJobs(nextJobs);
  }

  async function deleteJob(jobId: string) {
    await api.deleteJob(jobId);
    const nextJobs = await api.getJobs();
    setJobs(nextJobs);
    if (selectedJobId === jobId) {
      setSelectedJobId(nextJobs[0]?.job_id ?? null);
    }
  }

  async function clearQueuedJobs() {
    await api.clearQueuedJobs();
    const nextJobs = await api.getJobs();
    setJobs(nextJobs);
    if (selectedJobId && !nextJobs.some((job) => job.job_id === selectedJobId)) {
      setSelectedJobId(nextJobs[0]?.job_id ?? null);
    }
  }

  async function pauseJob(jobId: string) {
    await api.pauseJob(jobId);
    const nextJobs = await api.getJobs();
    setJobs(nextJobs);
  }

  async function resumeJob(jobId: string) {
    await api.resumeJob(jobId);
    const nextJobs = await api.getJobs();
    setJobs(nextJobs);
  }

  async function requeueJob(request: ProcessingJobRequest) {
    await createJobs([{ ...request }]);
  }

  async function saveServerConfig(payload: Partial<ServerState["config"]>) {
    const response = await api.updateServerConfig(payload);
    setServerState(response.state);
    const nextSettings = await api.getSettings();
    setSettings(nextSettings);
  }

  async function runServerAction(action: "start" | "stop" | "refresh") {
    const state = await api.serverAction(action);
    setServerState(state);
  }

  async function refreshOcioOptions(ocioPath?: string) {
    const nextOptions = await api.getOptions(ocioPath);
    setOptions((current) => ({
      ...current,
      ocio_configs: nextOptions.ocio_configs,
      ocio_spaces: nextOptions.ocio_spaces,
      default_ocio_config: nextOptions.default_ocio_config,
    }));
  }

  const defaults = useMemo(() => settings?.defaults ?? emptyDefaults, [settings]);

  return (
    <AppShell activeView={view} onViewChange={setView} jobs={jobs}>
      <AnimatePresence mode="wait">
        <motion.div
          key={view}
          initial={{ opacity: 0, y: 14, scale: 0.985 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -10, scale: 0.99 }}
          transition={{ duration: 0.24, ease: "easeOut" }}
          className="min-w-0"
        >
          {appError ? (
            <div className="mb-5 rounded-2xl border border-accent-red/30 bg-accent-red/10 px-4 py-3 text-sm text-rose-100">
              {appError}
            </div>
          ) : null}

          {view === "new-job" && (
            <NewJobWizard
              defaults={defaults}
              settings={settings}
              options={options}
              capabilities={capabilities}
              onSubmit={createJobs}
              onRefreshOcioOptions={refreshOcioOptions}
              submitting={submitting}
            />
          )}
          {view === "jobs" && (
            <JobsView
              jobs={jobs}
              selectedJobId={selectedJobId}
              onSelectJob={setSelectedJobId}
              onCancelJob={cancelJob}
              onDeleteJob={deleteJob}
              onClearQueuedJobs={clearQueuedJobs}
              onPauseJob={pauseJob}
              onResumeJob={resumeJob}
              onRequeueJob={requeueJob}
            />
          )}
          {view === "analytics" && <AnalyticsView />}
          {view === "server" && (
            <ServerView
              state={serverState}
              fallbackConfig={settings?.server ?? null}
              onSave={saveServerConfig}
              onAction={runServerAction}
            />
          )}
          {view === "settings" && (
            <SettingsView
              settings={settings}
              capabilities={capabilities}
              options={options}
              onSave={async (payload) => {
                const next = await api.saveSettings(payload);
                setSettings(next);
              }}
            />
          )}
        </motion.div>
      </AnimatePresence>
      <div className="fixed bottom-5 right-5 z-50 grid w-[min(360px,calc(100vw-2rem))] gap-3">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`rounded-2xl border px-4 py-3 shadow-glow backdrop-blur-xl ${
              toast.tone === "success"
                ? "border-accent-emerald/30 bg-accent-emerald/15"
                : "border-accent-red/30 bg-accent-red/15"
            }`}
          >
            <div className="font-medium text-white">{toast.title}</div>
            <div className="mt-1 text-sm text-slate-300">{toast.message}</div>
          </div>
        ))}
      </div>
    </AppShell>
  );
}
