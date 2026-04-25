import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  Box,
  CheckCircle2,
  Clock3,
  Copy,
  Cpu,
  Eye,
  FileJson,
  Film,
  FolderOpen,
  Images,
  LoaderCircle,
  Pause,
  PauseCircle,
  Play,
  RefreshCw,
  RotateCcw,
  Square,
  Terminal,
  Trash2,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../../lib/api";
import { desktop } from "../../lib/desktop";
import type {
  JobArtifactItem,
  JobArtifacts,
  JobDetail,
  JobLogEvent,
  JobSummary,
  ProcessingJobRequest,
} from "../../lib/types";
import { Button } from "../ui/button";
import { Card, CardDescription, CardTitle } from "../ui/card";
import { Progress } from "../ui/progress";

const ACTIVE_STATUSES = new Set(["queued", "processing", "paused"]);
const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp"]);
const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"]);

function basename(path: string) {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}

function formatBytes(value?: number | null) {
  if (!value) {
    return "-";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let size = value / 1024;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[unit]}`;
}

function statusTone(status: string) {
  if (status === "completed") {
    return "border-accent-emerald/30 bg-accent-emerald/10 text-accent-emerald";
  }
  if (status === "failed" || status === "cancelled" || status === "interrupted") {
    return "border-accent-red/30 bg-accent-red/10 text-rose-200";
  }
  if (status === "paused") {
    return "border-accent-amber/30 bg-accent-amber/10 text-accent-amber";
  }
  return "border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan";
}

function JobProcessIcon({ job, large = false }: { job: Pick<JobSummary, "status" | "current_step">; large?: boolean }) {
  const step = job.current_step.toLowerCase();
  const size = large ? 24 : 18;
  const wrapper = large ? "h-12 w-12 rounded-2xl" : "h-9 w-9 rounded-xl";

  if (job.status === "completed") {
    return (
      <div className={`${wrapper} flex items-center justify-center bg-accent-emerald/12 text-accent-emerald`}>
        <CheckCircle2 size={size} />
      </div>
    );
  }
  if (job.status === "failed" || job.status === "interrupted") {
    return (
      <div className={`${wrapper} flex items-center justify-center bg-accent-red/12 text-rose-200`}>
        <AlertTriangle size={size} />
      </div>
    );
  }
  if (job.status === "cancelled") {
    return (
      <div className={`${wrapper} flex items-center justify-center bg-accent-red/12 text-rose-200`}>
        <XCircle size={size} />
      </div>
    );
  }
  if (job.status === "paused") {
    return (
      <div className={`${wrapper} flex items-center justify-center bg-accent-amber/12 text-accent-amber`}>
        <PauseCircle size={size} />
      </div>
    );
  }
  if (job.status === "queued") {
    return (
      <motion.div
        className={`${wrapper} flex items-center justify-center bg-white/6 text-slate-300`}
        animate={{ opacity: [0.55, 1, 0.55] }}
        transition={{ duration: 1.3, repeat: Infinity, ease: "easeInOut" }}
      >
        <Clock3 size={size} />
      </motion.div>
    );
  }

  const Icon = step.includes("extract") || step.includes("frame")
    ? Film
    : step.includes("feature") || step.includes("match")
      ? Cpu
      : step.includes("map") || step.includes("reconstruct") || step.includes("sparse")
        ? Box
        : LoaderCircle;

  return (
    <motion.div
      className={`${wrapper} relative flex items-center justify-center overflow-hidden bg-accent-cyan/12 text-accent-cyan`}
      animate={{ boxShadow: ["0 0 0 0 rgba(62,203,255,0.18)", "0 0 0 8px rgba(62,203,255,0)", "0 0 0 0 rgba(62,203,255,0)"] }}
      transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }}
    >
      <motion.div
        animate={{ rotate: Icon === LoaderCircle ? 360 : 0, y: Icon === Film ? [0, -2, 0] : 0 }}
        transition={{ duration: Icon === LoaderCircle ? 1.1 : 0.9, repeat: Infinity, ease: "linear" }}
      >
        <Icon size={size} />
      </motion.div>
    </motion.div>
  );
}

function ArtifactButton({ item, label }: { item: JobArtifactItem; label?: string }) {
  return (
    <Button
      variant="secondary"
      className="min-w-0 justify-start"
      disabled={!item.exists}
      onClick={() => desktop.revealPath(item.path)}
      title={item.path}
    >
      <FolderOpen size={16} className="mr-2 shrink-0" />
      <span className="truncate">{label ?? item.name}</span>
    </Button>
  );
}

function MediaPreview({ item }: { item: JobArtifactItem }) {
  if (IMAGE_EXTENSIONS.has(item.extension)) {
    return (
      <button
        className="group relative aspect-video overflow-hidden rounded-xl border border-white/8 bg-graphite-950/80"
        onClick={() => desktop.revealPath(item.path)}
        title={item.path}
      >
        <img
          src={api.fileUrl(item.path)}
          alt={item.name}
          className="h-full w-full object-cover transition duration-200 group-hover:scale-105"
          loading="lazy"
        />
        <div className="absolute inset-x-0 bottom-0 bg-graphite-950/78 px-2 py-1 text-left text-xs text-slate-200">
          <div className="truncate">{item.name}</div>
        </div>
      </button>
    );
  }
  if (VIDEO_EXTENSIONS.has(item.extension)) {
    return (
      <div className="overflow-hidden rounded-xl border border-white/8 bg-graphite-950/80">
        <video src={api.fileUrl(item.path)} controls className="aspect-video w-full bg-black" />
        <div className="truncate px-3 py-2 text-xs text-slate-300">{item.name}</div>
      </div>
    );
  }
  return null;
}

function FileList({ items, empty }: { items: JobArtifactItem[]; empty: string }) {
  if (!items.length) {
    return <div className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-3 text-sm text-slate-500">{empty}</div>;
  }
  return (
    <div className="space-y-2">
      {items.slice(0, 12).map((item) => (
        <button
          key={item.path}
          onClick={() => desktop.revealPath(item.path)}
          className="grid w-full min-w-0 grid-cols-[1fr_auto] gap-3 rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2 text-left transition hover:bg-white/[0.06]"
          title={item.path}
        >
          <span className="min-w-0 truncate text-sm text-slate-200">{item.name}</span>
          <span className="text-xs text-slate-500">{formatBytes(item.size)}</span>
        </button>
      ))}
    </div>
  );
}

export function JobsView({
  jobs,
  selectedJobId,
  onSelectJob,
  onCancelJob,
  onDeleteJob,
  onClearQueuedJobs,
  onPauseJob,
  onResumeJob,
  onRequeueJob,
}: {
  jobs: JobSummary[];
  selectedJobId: string | null;
  onSelectJob: (jobId: string) => void;
  onCancelJob: (jobId: string) => Promise<void>;
  onDeleteJob: (jobId: string) => Promise<void>;
  onClearQueuedJobs: () => Promise<void>;
  onPauseJob: (jobId: string) => Promise<void>;
  onResumeJob: (jobId: string) => Promise<void>;
  onRequeueJob: (request: ProcessingJobRequest) => Promise<void>;
}) {
  const [detail, setDetail] = useState<JobDetail | null>(null);
  const [liveLogs, setLiveLogs] = useState<JobLogEvent[]>([]);
  const [artifacts, setArtifacts] = useState<JobArtifacts | null>(null);
  const [showConsole, setShowConsole] = useState(false);
  const [showPayload, setShowPayload] = useState(false);

  const queuedCount = jobs.filter((job) => job.status === "queued").length;
  const clearableCount = jobs.filter((job) => job.status !== "processing").length;

  useEffect(() => {
    if (!selectedJobId) {
      setDetail(null);
      setLiveLogs([]);
      setArtifacts(null);
      return;
    }

    let disposed = false;
    const loadDetail = async () => {
      const result = await api.getJob(selectedJobId);
      if (!disposed) {
        setDetail(result);
        setLiveLogs(result.logs);
      }
    };
    const loadArtifacts = async () => {
      const result = await api.getJobArtifacts(selectedJobId);
      if (!disposed) {
        setArtifacts(result);
      }
    };

    loadDetail().catch(console.error);
    loadArtifacts().catch(console.error);

    const detailPoll = window.setInterval(() => {
      loadDetail().catch(console.error);
    }, 2500);
    const artifactPoll = window.setInterval(() => {
      loadArtifacts().catch(console.error);
    }, 5000);

    const source = new EventSource(api.jobLogsStreamUrl(selectedJobId));
    source.addEventListener("log", (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as JobLogEvent;
      setLiveLogs((current) =>
        current.some((item) => item.id === payload.id) ? current : [...current, payload],
      );
    });

    return () => {
      disposed = true;
      window.clearInterval(detailPoll);
      window.clearInterval(artifactPoll);
      source.close();
    };
  }, [selectedJobId]);

  const activeJob = useMemo(
    () => jobs.find((job) => job.job_id === selectedJobId) ?? jobs[0] ?? null,
    [jobs, selectedJobId],
  );
  const liveDetail = activeJob && detail?.job_id === activeJob.job_id ? { ...detail, ...activeJob } : detail;
  const inputFolders = artifacts?.input_paths ?? [];
  const previewMedia = [
    ...(artifacts?.input_paths.filter((item) => item.previewable) ?? []),
    ...(artifacts?.input_paths.flatMap((item) => item.samples ?? []) ?? []),
    ...(artifacts?.frames ?? []),
  ].slice(0, 9);

  return (
    <div className="grid h-full min-w-0 gap-4 xl:grid-cols-[minmax(300px,380px)_minmax(0,1fr)] xl:gap-6">
      <Card className="flex min-h-[420px] flex-col xl:min-h-[min(760px,calc(100vh-9rem))]">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <CardTitle>Queue</CardTitle>
            <CardDescription>{jobs.length} jobs - {queuedCount} queued</CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="secondary" onClick={onClearQueuedJobs} disabled={!clearableCount} title="Clear queue">
              <Trash2 size={16} />
            </Button>
            <Button variant="info" onClick={() => activeJob && onSelectJob(activeJob.job_id)} title="Refresh selection">
              <RefreshCw size={16} />
            </Button>
          </div>
        </div>
        <div className="space-y-3 overflow-auto pr-1">
          {jobs.map((job) => (
            <button
              key={job.job_id}
              onClick={() => onSelectJob(job.job_id)}
              className={`relative w-full overflow-hidden rounded-2xl border p-4 text-left transition ${
                activeJob?.job_id === job.job_id
                  ? "border-accent-cyan/40 bg-accent-cyan/10"
                  : "border-white/8 bg-white/[0.03] hover:bg-white/[0.06]"
              }`}
            >
              {job.status === "processing" ? (
                <motion.div
                  className="absolute inset-0 bg-accent-cyan/5"
                  animate={{ opacity: [0.15, 0.5, 0.15] }}
                  transition={{ duration: 1.35, repeat: Infinity, ease: "easeInOut" }}
                />
              ) : null}
              <div className="relative flex items-start gap-3">
                <JobProcessIcon job={job} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0 truncate font-medium text-white">{job.label || job.job_id}</div>
                    <div className={`shrink-0 rounded-full border px-2 py-1 text-[10px] uppercase tracking-[0.14em] ${statusTone(job.status)}`}>
                      {job.status}
                    </div>
                  </div>
                  <div className="mt-2 line-clamp-2 text-sm text-slate-400">
                    {job.progress_note || job.current_step}
                  </div>
                  <div className="mt-3 flex items-center gap-3">
                    <Progress value={job.progress} />
                    <div className="w-10 text-right text-xs font-semibold text-slate-300">{job.progress}%</div>
                  </div>
                  {job.queue_position ? (
                    <div className="mt-2 text-xs text-accent-cyan">Queue #{job.queue_position}</div>
                  ) : null}
                </div>
              </div>
            </button>
          ))}
          {!jobs.length ? (
            <div className="rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-8 text-center text-sm text-slate-500">
              No jobs yet.
            </div>
          ) : null}
        </div>
      </Card>

      <Card className="min-h-[420px] min-w-0 overflow-auto xl:min-h-[min(760px,calc(100vh-9rem))]">
        {liveDetail ? (
          <div className="space-y-5">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="flex min-w-0 items-start gap-4">
                <JobProcessIcon job={liveDetail} large />
                <div className="min-w-0">
                  <CardTitle className="truncate">{liveDetail.label || liveDetail.job_id}</CardTitle>
                  <CardDescription>
                    {liveDetail.input_mode} - {liveDetail.progress_note || liveDetail.current_step}
                  </CardDescription>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant={showConsole ? "info" : "secondary"} onClick={() => setShowConsole((value) => !value)}>
                  <Terminal size={16} className="mr-2" />
                  Console
                </Button>
                <Button variant={showPayload ? "info" : "secondary"} onClick={() => setShowPayload((value) => !value)}>
                  <FileJson size={16} className="mr-2" />
                  Request
                </Button>
                <Button variant="warning" onClick={() => onRequeueJob(liveDetail.request)}>
                  <RotateCcw size={16} className="mr-2" />
                  Requeue
                </Button>
                {liveDetail.status === "paused" ? (
                  <Button variant="success" onClick={() => onResumeJob(liveDetail.job_id)}>
                    <Play size={14} className="mr-2" />
                    Resume
                  </Button>
                ) : liveDetail.status === "processing" || liveDetail.status === "queued" ? (
                  <Button variant="warning" onClick={() => onPauseJob(liveDetail.job_id)}>
                    <Pause size={14} className="mr-2" />
                    Pause
                  </Button>
                ) : null}
                {ACTIVE_STATUSES.has(liveDetail.status) ? (
                  <Button variant="danger" onClick={() => onCancelJob(liveDetail.job_id)}>
                    <Square size={14} className="mr-2" />
                    Cancel
                  </Button>
                ) : null}
                <Button variant="danger" onClick={() => onDeleteJob(liveDetail.job_id)}>
                  <Trash2 size={14} className="mr-2" />
                  Delete
                </Button>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-4">
              <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                <div className="text-sm text-slate-500">Status</div>
                <div className="mt-2 text-xl font-semibold capitalize">{liveDetail.status}</div>
              </div>
              <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                <div className="text-sm text-slate-500">Progress</div>
                <div className="mt-2 text-xl font-semibold">{liveDetail.progress}%</div>
              </div>
              <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                <div className="text-sm text-slate-500">Queue</div>
                <div className="mt-2 text-xl font-semibold">{liveDetail.queue_position ?? "-"}</div>
              </div>
              <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                <div className="text-sm text-slate-500">Updated</div>
                <div className="mt-2 text-sm font-semibold text-slate-200">
                  {new Date(liveDetail.updated_at).toLocaleTimeString()}
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-white/8 bg-graphite-950/60 p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 text-sm font-medium text-slate-200">
                  <JobProcessIcon job={liveDetail} />
                  <span>{liveDetail.progress_note || liveDetail.current_step}</span>
                </div>
                <span className="text-sm font-semibold text-accent-cyan">{liveDetail.progress}%</span>
              </div>
              <Progress value={liveDetail.progress} />
            </div>

            <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
              <div className="space-y-4 rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-medium text-slate-300">Folders</div>
                  <Button variant="secondary" onClick={() => api.getJobArtifacts(liveDetail.job_id).then(setArtifacts).catch(console.error)}>
                    <RefreshCw size={16} />
                  </Button>
                </div>
                <div className="grid gap-2">
                  {artifacts ? <ArtifactButton item={artifacts.output_path} label="Output" /> : null}
                  {inputFolders.map((item, index) => (
                    <ArtifactButton key={item.path} item={item} label={`Input ${index + 1}: ${basename(item.path)}`} />
                  ))}
                </div>
                <div>
                  <div className="mb-2 text-sm font-medium text-slate-300">Reconstruction</div>
                  <FileList items={artifacts?.reconstruction ?? []} empty="No reconstruction output yet." />
                </div>
                <div>
                  <div className="mb-2 text-sm font-medium text-slate-300">EXR</div>
                  <FileList items={artifacts?.exrs ?? []} empty="No EXR files yet." />
                </div>
              </div>

              <div className="space-y-4 rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-slate-300">
                    <Images size={16} />
                    Media
                  </div>
                  <Button variant="secondary" onClick={() => liveDetail.output_path && desktop.revealPath(liveDetail.output_path)}>
                    <FolderOpen size={16} />
                  </Button>
                </div>
                {previewMedia.length ? (
                  <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
                    {previewMedia.map((item) => (
                      <MediaPreview key={item.path} item={item} />
                    ))}
                  </div>
                ) : (
                  <div className="flex min-h-32 items-center justify-center rounded-xl border border-white/8 bg-graphite-950/70 text-sm text-slate-500">
                    Preview waiting for video or frames.
                  </div>
                )}
                <div>
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-300">
                    <Eye size={16} />
                    Latest outputs
                  </div>
                  <FileList items={artifacts?.latest_outputs ?? []} empty="No files yet." />
                </div>
              </div>
            </div>

            <AnimatePresence>
              {showConsole ? (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="rounded-2xl border border-white/8 bg-graphite-950/90 p-4">
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <div className="text-sm font-medium text-slate-300">Console</div>
                      <Button
                        variant="secondary"
                        onClick={() =>
                          navigator.clipboard.writeText(
                            liveLogs.map((event) => `${event.timestamp} ${event.message}`).join("\n"),
                          )
                        }
                      >
                        <Copy size={16} className="mr-2" />
                        Copy
                      </Button>
                    </div>
                    <pre className="max-h-[360px] overflow-auto font-mono text-xs text-slate-300">
{liveLogs.map((event) => `[${event.timestamp}] ${event.message}`).join("\n")}
                    </pre>
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>

            <AnimatePresence>
              {showPayload ? (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                    <div className="mb-3 text-sm font-medium text-slate-300">Request payload</div>
                    <pre className="max-h-[360px] overflow-auto font-mono text-xs text-slate-300">
{JSON.stringify(liveDetail.request, null, 2)}
                    </pre>
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>
        ) : (
          <div className="flex h-full min-h-[360px] items-center justify-center text-slate-500">
            Select a job to inspect its live output.
          </div>
        )}
      </Card>
    </div>
  );
}
