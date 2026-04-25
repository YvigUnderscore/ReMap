import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
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
      const created: JobDetail[] = [];
      for (const payload of payloads) {
        const job = await api.createJob(payload);
        created.push(job);
      }
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
    </AppShell>
  );
}
