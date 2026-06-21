import {
  AlertTriangle,
  ArrowRight,
  Check,
  Clock3,
  FileJson,
  FolderOpen,
  HardDrive,
  History,
  Plus,
  Sparkle,
  Trash2,
  Wand2,
} from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState, type SetStateAction } from "react";
import { api } from "../../lib/api";
import { desktop } from "../../lib/desktop";
import type {
  CapabilitiesPayload,
  AppSettings,
  EstimatePayload,
  OptionsPayload,
  ProbeItem,
  ProcessingJobRequest,
} from "../../lib/types";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Card, CardDescription, CardTitle } from "../ui/card";
import { Input } from "../ui/input";

const steps = ["Source", "Preparation", "Color", "Reconstruction", "Review & Run"] as const;
const ACESCG_OCIO_SPACE = "ACES - ACEScg";
const WIZARD_STORAGE_KEY = "remap:new-job-wizard";
const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"]);

type WizardSnapshot = {
  step?: number;
  draft?: ProcessingJobRequest;
};

function basename(path: string) {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}

function slugify(value: string) {
  return value.replace(/\.[^.]+$/, "").replace(/[^\w-]+/g, "_");
}

function extname(path: string) {
  const name = basename(path);
  const index = name.lastIndexOf(".");
  return index >= 0 ? name.slice(index).toLowerCase() : "";
}

function formatBytes(value?: number | null) {
  if (!value) {
    return "-";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDuration(seconds?: number | null) {
  if (!seconds) {
    return "-";
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes) {
    return `${minutes}m`;
  }
  return `${Math.max(1, Math.round(seconds))}s`;
}

function sliderMaxFromProbe(items: ProbeItem[], maxNativeFps: number) {
  if (maxNativeFps > 0) {
    return Math.max(1, Math.ceil(maxNativeFps));
  }
  const itemMax = Math.max(0, ...items.map((item) => item.native_fps || 0));
  return Math.max(1, Math.ceil(itemMax || 60));
}

function readWizardSnapshot(): WizardSnapshot | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(WIZARD_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as WizardSnapshot) : null;
  } catch {
    return null;
  }
}

function clampStep(value: number | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return 0;
  }
  return Math.min(steps.length - 1, Math.max(0, value));
}

function samplingStep(nativeFps: number, targetFps: number) {
  if (!targetFps || targetFps <= 0 || !nativeFps || nativeFps <= 0) {
    return 1;
  }
  return Math.max(1, Math.round(nativeFps / targetFps));
}

function estimateFrames(item: ProbeItem, fps: number, inputMode: ProcessingJobRequest["input_mode"]) {
  if (inputMode === "images") {
    return { estimated_frames: item.total_frames, step: item.step };
  }
  const step = samplingStep(item.native_fps || fps, fps);
  if (item.total_frames > 0) {
    return { estimated_frames: Math.ceil(item.total_frames / step), step };
  }
  if (item.duration && item.duration > 0) {
    return { estimated_frames: Math.max(0, Math.round(item.duration * fps)), step };
  }
  return { estimated_frames: 0, step };
}

export function NewJobWizard({
  defaults,
  settings,
  options,
  capabilities,
  onSubmit,
  onRefreshOcioOptions,
  submitting,
}: {
  defaults: ProcessingJobRequest;
  settings: AppSettings | null;
  options: OptionsPayload;
  capabilities?: CapabilitiesPayload | null;
  onSubmit: (payloads: ProcessingJobRequest[]) => Promise<void>;
  onRefreshOcioOptions: (ocioPath?: string) => Promise<void>;
  submitting: boolean;
}) {
  const savedSnapshotRef = useRef<WizardSnapshot | null>(readWizardSnapshot());
  const draftDirtyRef = useRef(false);
  const restoredSnapshotRef = useRef(false);
  const [step, setStep] = useState(() => clampStep(savedSnapshotRef.current?.step));
  const [draft, setDraft] = useState<ProcessingJobRequest>(defaults);
  const [probeItems, setProbeItems] = useState<ProbeItem[]>([]);
  const [maxNativeFps, setMaxNativeFps] = useState(60);
  const [probeLoading, setProbeLoading] = useState(false);
  const [showRequestPayload, setShowRequestPayload] = useState(false);
  const [estimate, setEstimate] = useState<EstimatePayload | null>(null);
  const [estimateLoading, setEstimateLoading] = useState(false);
  const [dropActive, setDropActive] = useState(false);

  useEffect(() => {
    if (restoredSnapshotRef.current || draftDirtyRef.current) {
      return;
    }
    const snapshot = savedSnapshotRef.current;
    if (snapshot?.draft) {
      setDraft({ ...defaults, ...snapshot.draft });
      setStep(clampStep(snapshot.step));
      restoredSnapshotRef.current = true;
      draftDirtyRef.current = true;
      return;
    }
    setDraft(defaults);
  }, [defaults]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(
      WIZARD_STORAGE_KEY,
      JSON.stringify({ step, draft }),
    );
  }, [step, draft]);

  useEffect(() => {
    if (!draft.input_paths.length) {
      setProbeItems([]);
      setMaxNativeFps(60);
      return;
    }
    let disposed = false;
    setProbeLoading(true);
    api
      .probeInputs({
        input_mode: draft.input_mode,
        input_paths: draft.input_paths,
      })
      .then((response) => {
        if (disposed) {
          return;
        }
        setProbeItems(response.items);
        setMaxNativeFps(sliderMaxFromProbe(response.items, response.max_native_fps));
        if (!draft.detected_color_profile && response.detected_color_profile) {
          setDraft((current) => ({
            ...current,
            detected_color_profile: response.detected_color_profile,
          }));
        }
      })
      .finally(() => {
        if (!disposed) {
          setProbeLoading(false);
        }
      });
    return () => {
      disposed = true;
    };
  }, [draft.input_mode, draft.input_paths]);

  useEffect(() => {
    if (!draft.ocio_path && options.default_ocio_config) {
      setDraft((current) => ({ ...current, ocio_path: options.default_ocio_config }));
    }
  }, [draft.ocio_path, options.default_ocio_config]);

  useEffect(() => {
    if (
      draft.color_dest === "ACEScg (EXR + sRGB PNG)" &&
      options.ocio_spaces.includes(ACESCG_OCIO_SPACE) &&
      draft.ocio_out_cs !== ACESCG_OCIO_SPACE
    ) {
      setDraft((current) => ({ ...current, ocio_out_cs: ACESCG_OCIO_SPACE }));
    }
  }, [draft.color_dest, draft.ocio_out_cs, options.ocio_spaces]);

  useEffect(() => {
    let cleanup: (() => void) | undefined;
    desktop.onFileDrop((paths) => {
      setDropActive(false);
      addDroppedPaths(paths);
    }).then((unlisten) => {
      cleanup = unlisten;
    });
    return () => cleanup?.();
  }, [draft.input_mode]);

  function setDraftFromUser(updater: SetStateAction<ProcessingJobRequest>) {
    draftDirtyRef.current = true;
    setDraft(updater);
  }

  function update<K extends keyof ProcessingJobRequest>(key: K, value: ProcessingJobRequest[K]) {
    setDraftFromUser((current) => ({ ...current, [key]: value }));
  }

  function applyInputMode(inputMode: ProcessingJobRequest["input_mode"]) {
    setDraftFromUser((current) => ({
      ...current,
      input_mode: inputMode,
      input_paths: [],
      ...(inputMode === "rescan"
        ? {
            color_enabled: true,
            color_source: "Auto-detect",
            color_dest: "ACEScg (EXR + sRGB PNG)",
            use_acescg_exr: true,
          }
        : {}),
    }));
  }

  function addInputPaths(paths: string[], mode?: ProcessingJobRequest["input_mode"]) {
    if (!paths.length) {
      return;
    }
    setDraftFromUser((current) => ({
      ...current,
      input_mode: mode ?? current.input_mode,
      input_paths: Array.from(new Set([...(mode && mode !== current.input_mode ? [] : current.input_paths), ...paths])),
    }));
  }

  function addDroppedPaths(paths: string[]) {
    const cleaned = paths.filter(Boolean);
    if (!cleaned.length) {
      return;
    }
    const videos = cleaned.filter((path) => VIDEO_EXTENSIONS.has(extname(path)));
    if (videos.length === cleaned.length) {
      addInputPaths(videos, "video");
      return;
    }
    addInputPaths(cleaned, draft.input_mode === "video" ? "images" : draft.input_mode);
  }

  async function pickVideos() {
    const picked = await desktop.pickVideoFiles();
    if (picked?.length) {
      addInputPaths(picked, "video");
    }
  }

  async function pickFolder(mode: "input" | "output" | "ocio" = "input") {
    if (mode === "output") {
      const picked = await desktop.pickDirectory();
      if (picked) {
        update("output_path", picked);
      }
      return;
    }
    if (mode === "ocio") {
      const picked = await desktop.pickOcioFile();
      if (picked) {
        update("ocio_path", picked);
        await onRefreshOcioOptions(picked);
      }
      return;
    }
    const many = await desktop.pickDirectories();
    const fallback = many?.length ? null : await desktop.pickDirectory();
    const picked = many?.length ? many : fallback ? [fallback] : [];
    if (!picked.length) {
      return;
    }
    addInputPaths(picked);
  }

  function removeInput(path: string) {
    setDraftFromUser((current) => ({
      ...current,
      input_paths: current.input_paths.filter((item) => item !== path),
    }));
  }

  function updateColorInput(value: string) {
    if (value.startsWith("ocio:")) {
      const space = value.slice("ocio:".length);
      setDraftFromUser((current) => ({
        ...current,
        color_enabled: true,
        color_source: "Auto-detect",
        color_dest: "Custom OCIO...",
        ocio_in_cs: space,
        ocio_out_cs:
          current.ocio_out_cs ||
          (options.ocio_spaces.includes(ACESCG_OCIO_SPACE)
            ? ACESCG_OCIO_SPACE
            : options.ocio_spaces[0] || ""),
      }));
      return;
    }
    const source = value.replace("builtin:", "") || "Auto-detect";
    setDraftFromUser((current) => ({
      ...current,
      color_source: source,
      ocio_in_cs: "",
    }));
  }

  function updateColorOutput(value: string) {
    if (value.startsWith("ocio:")) {
      const space = value.slice("ocio:".length);
      setDraftFromUser((current) => ({
        ...current,
        color_enabled: true,
        color_dest: "Custom OCIO...",
        ocio_out_cs: space,
        ocio_in_cs: current.ocio_in_cs || options.ocio_spaces[0] || "",
      }));
      return;
    }
    const destination = value.replace("builtin:", "") || "ACEScg (EXR + sRGB PNG)";
    setDraftFromUser((current) => ({
      ...current,
      color_dest: destination,
      ocio_out_cs:
        destination === "ACEScg (EXR + sRGB PNG)" &&
        options.ocio_spaces.includes(ACESCG_OCIO_SPACE)
          ? ACESCG_OCIO_SPACE
          : "",
    }));
  }

  const displayProbeItems = useMemo(
    () =>
      probeItems.map((item) => {
        return { ...item, ...estimateFrames(item, draft.fps_extract, draft.input_mode) };
      }),
    [draft.fps_extract, draft.input_mode, probeItems],
  );

  function buildRequests() {
    const items = displayProbeItems.length
      ? displayProbeItems.filter((item) => item.valid)
      : draft.input_paths.map((path) => ({
          path,
          name: basename(path),
          kind: draft.input_mode,
          native_fps: draft.fps_extract,
          total_frames: 0,
          estimated_frames: 0,
          color_profile: "",
          valid: true,
        }));
    const multi = items.length > 1;
    return items.map((item, index) => {
      const itemName = item.name || basename(item.path);
      const manualPrefix = draft.label.trim();
      const label = manualPrefix ? (multi ? `${manualPrefix} - ${itemName}` : manualPrefix) : itemName;
      const outputPath = multi
        ? `${draft.output_path}/${slugify(itemName || `dataset_${index + 1}`)}`
        : draft.output_path;
      return {
        ...draft,
        input_paths: [item.path],
        output_path: outputPath,
        color_enabled: draft.input_mode === "rescan" ? true : draft.color_enabled,
        color_source: draft.input_mode === "rescan" ? "Auto-detect" : draft.color_source,
        color_dest:
          draft.input_mode === "rescan" && !draft.use_acescg_exr
            ? "sRGB (Tone Mapped)"
            : draft.color_dest,
        fps_extract: draft.fps_extract,
        detected_color_profile:
          draft.detected_color_profile || item.color_profile || draft.detected_color_profile,
        label,
      };
    });
  }

  async function handleSubmit() {
    await onSubmit(buildRequests());
  }

  const workerMax = capabilities?.cpu_count ?? 16;
  const queuePreview = useMemo(() => buildRequests(), [displayProbeItems, draft]);
  const colorInputValue =
    draft.color_dest === "Custom OCIO..." && draft.ocio_in_cs
      ? `ocio:${draft.ocio_in_cs}`
      : `builtin:${draft.color_source || "Auto-detect"}`;
  const colorOutputValue =
    draft.color_dest === "Custom OCIO..."
      ? draft.ocio_out_cs
        ? `ocio:${draft.ocio_out_cs}`
        : ""
      : `builtin:${draft.color_dest}`;
  const builtinColorOutputs = options.color_destinations.filter(
    (option) => option !== "Custom OCIO...",
  );

  useEffect(() => {
    if (!queuePreview.length || !draft.output_path) {
      setEstimate(null);
      return;
    }
    const timer = window.setTimeout(() => {
      setEstimateLoading(true);
      api
        .estimate(queuePreview)
        .then(setEstimate)
        .catch(() => setEstimate(null))
        .finally(() => setEstimateLoading(false));
    }, 400);
    return () => window.clearTimeout(timer);
  }, [queuePreview, draft.output_path]);

  function buildQualitySweepRequests() {
    const baseRequests = queuePreview.length ? queuePreview : buildRequests();
    const presets = [
      { suffix: "fast", matcher_type: "superpoint+lightglue", max_keypoints: 2048, fpsFactor: 0.5 },
      { suffix: "balanced", matcher_type: draft.matcher_type, max_keypoints: Math.max(4096, draft.max_keypoints), fpsFactor: 0.75 },
      { suffix: "quality", matcher_type: draft.matcher_type.includes("loma") ? draft.matcher_type : "loma_b", max_keypoints: Math.max(8192, draft.max_keypoints), fpsFactor: 1 },
    ];
    return baseRequests.flatMap((request) =>
      presets.map((preset) => ({
        ...request,
        quality_sweep: true,
        sweep_sample_frames: draft.sweep_sample_frames,
        matcher_type: preset.matcher_type,
        max_keypoints: preset.max_keypoints,
        fps_extract: Math.max(0.5, Number((request.fps_extract * preset.fpsFactor).toFixed(1))),
        label: `${request.label || basename(request.input_paths[0])} - ${preset.suffix}`,
        output_path: `${request.output_path}_${preset.suffix}`,
      })),
    );
  }

  async function handleQualitySweep() {
    await onSubmit(buildQualitySweepRequests());
  }

  const recentInputs = settings?.recent_inputs ?? [];
  const recentOutputs = settings?.recent_outputs ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <Badge className="border-accent-cyan/30 bg-accent-cyan/8 text-accent-cyan">
            Guided Workflow
          </Badge>
          <h2 className="mt-3 font-display text-2xl font-semibold sm:text-3xl">
            Build a clean queue from your datasets.
          </h2>
          <p className="mt-2 max-w-2xl text-sm text-slate-400">
            Input/output browse is native, preparation is dataset-aware, and the queue runs jobs
            one after another.
          </p>
        </div>

        <Card className="w-full min-w-0 max-w-md bg-gradient-to-br from-accent-cyan/8 to-accent-blue/6">
          <div className="flex items-start gap-3">
            <Sparkle className="mt-1 text-accent-cyan" size={18} />
            <div>
              <CardTitle className="text-base">Detected machine profile</CardTitle>
              <CardDescription>
                {capabilities
                  ? `${capabilities.cpu_count} CPU threads · ${
                      capabilities.cuda_available ? "CUDA ready" : "CPU only"
                    } · ${capabilities.glomap_available ? "GLOMAP available" : "COLMAP fallback"}`
                  : "Loading capabilities..."}
              </CardDescription>
            </div>
          </div>
        </Card>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        {steps.map((label, index) => (
          <button
            key={label}
            onClick={() => setStep(index)}
            className={`min-w-0 rounded-2xl border px-4 py-3 text-left transition duration-200 hover:-translate-y-0.5 active:translate-y-0 ${
              index === step
                ? "border-accent-cyan/40 bg-accent-cyan/10"
                : "border-white/8 bg-white/[0.025] hover:bg-white/[0.05]"
            }`}
          >
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs uppercase tracking-[0.18em] text-slate-500">
                Step {index + 1}
              </span>
              {index < step && <Check size={16} className="text-accent-emerald" />}
            </div>
            <div className="font-medium">{label}</div>
          </button>
        ))}
      </div>

      <div className="grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(280px,0.75fr)] xl:gap-6">
        <Card className="min-w-0 overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.div
              key={step}
              initial={{ opacity: 0, x: 18 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -18 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
            >
          {step === 0 && (
            <div className="space-y-5">
              <div>
                <CardTitle>Source</CardTitle>
                <CardDescription>
                  Browse input datasets and choose where the processed outputs should land.
                </CardDescription>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <label className="space-y-2 text-sm text-slate-300">
                  Input Mode
                  <select
                    value={draft.input_mode}
                    onChange={(event) =>
                      applyInputMode(event.target.value as ProcessingJobRequest["input_mode"])
                    }
                    className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
                  >
                    <option value="video">Video</option>
                    <option value="images">Image Folder</option>
                    <option value="rescan">Rescan / LiDAR</option>
                  </select>
                </label>
                <label className="space-y-2 text-sm text-slate-300">
                  Job Label Prefix
                  <Input
                    value={draft.label}
                    onChange={(event) => update("label", event.target.value)}
                    placeholder="Auto: selected folder/video name"
                  />
                </label>
              </div>

              <div
                onDragEnter={(event) => {
                  event.preventDefault();
                  setDropActive(true);
                }}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDropActive(true);
                }}
                onDragLeave={() => setDropActive(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDropActive(false);
                  const paths = Array.from(event.dataTransfer.files)
                    .map((file) => (file as File & { path?: string }).path || file.name)
                    .filter(Boolean);
                  addDroppedPaths(paths);
                }}
                className={`rounded-3xl border border-dashed px-5 py-8 text-center transition ${
                  dropActive
                    ? "border-accent-cyan/60 bg-accent-cyan/10"
                    : "border-white/12 bg-white/[0.025]"
                }`}
              >
                <div className="text-sm font-medium text-white">Drop videos or dataset folders here</div>
                <div className="mt-1 text-xs text-slate-500">
                  Tauri drops keep full paths; browser fallback uses available file names.
                </div>
              </div>

              <div className="flex flex-wrap gap-3">
                {draft.input_mode === "video" ? (
                  <Button variant="info" onClick={pickVideos}>
                    <FolderOpen size={16} className="mr-2" />
                    Browse video files
                  </Button>
                ) : (
                  <Button variant="info" onClick={() => pickFolder("input")}>
                    <Plus size={16} className="mr-2" />
                    Add Dataset Folder(s)
                  </Button>
                )}
                {draft.input_mode !== "video" && draft.input_paths.length > 0 ? (
                  <Button
                    variant="danger"
                    onClick={() => setDraftFromUser((current) => ({ ...current, input_paths: [] }))}
                  >
                    <Trash2 size={16} className="mr-2" />
                    Clear list
                  </Button>
                ) : null}
              </div>

              <div className="space-y-3">
                {draft.input_paths.length ? (
                  draft.input_paths.map((path) => (
                    <div
                      key={path}
                      className="flex min-w-0 flex-col gap-3 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3 sm:flex-row sm:items-center sm:justify-between"
                    >
                      <div className="min-w-0">
                        <div className="truncate font-medium text-white">{basename(path)}</div>
                        <div className="truncate text-xs text-slate-500">{path}</div>
                      </div>
                      <Button variant="ghost" onClick={() => removeInput(path)}>
                        Remove
                      </Button>
                    </div>
                  ))
                ) : (
                  <div className="rounded-2xl border border-dashed border-white/10 px-4 py-6 text-sm text-slate-500">
                    No dataset selected yet.
                  </div>
                )}
              </div>

              <label className="block space-y-2 text-sm text-slate-300">
                Output Root Folder
                <div className="flex gap-3">
                  <Input value={draft.output_path} readOnly placeholder="Choose an output folder" />
                  <Button variant="info" onClick={() => pickFolder("output")}>
                    Browse
                  </Button>
                </div>
              </label>

              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                  <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-300">
                    <History size={16} />
                    Recent inputs
                  </div>
                  <div className="space-y-2">
                    {recentInputs.slice(0, 5).map((path) => (
                      <button
                        key={path}
                        onClick={() => addInputPaths([path])}
                        className="w-full truncate rounded-xl bg-graphite-950/70 px-3 py-2 text-left text-xs text-slate-300 hover:bg-white/8"
                        title={path}
                      >
                        {basename(path)}
                      </button>
                    ))}
                    {!recentInputs.length ? <div className="text-xs text-slate-500">No recent input yet.</div> : null}
                  </div>
                </div>
                <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                  <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-300">
                    <History size={16} />
                    Recent outputs
                  </div>
                  <div className="space-y-2">
                    {recentOutputs.slice(0, 5).map((path) => (
                      <button
                        key={path}
                        onClick={() => update("output_path", path)}
                        className="w-full truncate rounded-xl bg-graphite-950/70 px-3 py-2 text-left text-xs text-slate-300 hover:bg-white/8"
                        title={path}
                      >
                        {basename(path) || path}
                      </button>
                    ))}
                    {!recentOutputs.length ? <div className="text-xs text-slate-500">No recent output yet.</div> : null}
                  </div>
                </div>
              </div>
            </div>
          )}

          {step === 1 && (
            <div className="space-y-5">
              <div>
                <CardTitle>Preparation</CardTitle>
                <CardDescription>
                  Set one extraction FPS for every queued dataset.
                </CardDescription>
              </div>

              {draft.input_mode !== "images" ? (
                <div className="rounded-3xl border border-white/8 bg-white/[0.03] p-4">
                  <div className="mb-2 flex items-center justify-between text-sm text-slate-300">
                    <span>Global FPS</span>
                    <span>{draft.fps_extract.toFixed(1)} FPS</span>
                  </div>
                  <input
                    type="range"
                    min={0.5}
                    max={maxNativeFps}
                    step={0.5}
                    value={draft.fps_extract}
                    onChange={(event) => update("fps_extract", Number(event.target.value))}
                    className="w-full accent-accent-cyan"
                  />
                  <div className="mt-2 text-xs text-slate-500">
                    Max detected FPS: {maxNativeFps.toFixed(1)}
                  </div>
                </div>
              ) : null}

              <div className="grid gap-4 md:grid-cols-2">
                <label className="flex items-center justify-between gap-4 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3 text-sm text-slate-200">
                  <span>
                    <span className="block font-medium text-white">Reuse existing checkpoints/cache</span>
                    <span className="mt-1 block text-xs text-slate-500">Skip extraction/features/matches when manifests are valid.</span>
                  </span>
                  <input
                    type="checkbox"
                    checked={draft.skip_existing}
                    onChange={(event) => update("skip_existing", event.target.checked)}
                    className="h-5 w-5 accent-accent-cyan"
                  />
                </label>
                <label className="flex items-center justify-between gap-4 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3 text-sm text-slate-200">
                  <span>
                    <span className="block font-medium text-white">Reject blurry frames</span>
                    <span className="mt-1 block text-xs text-slate-500">Moves rejected frames to `_rejected_frames`.</span>
                  </span>
                  <input
                    type="checkbox"
                    checked={draft.exclude_blurry}
                    onChange={(event) => update("exclude_blurry", event.target.checked)}
                    className="h-5 w-5 accent-accent-cyan"
                  />
                </label>
                <label className="flex items-center justify-between gap-4 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3 text-sm text-slate-200">
                  <span>
                    <span className="block font-medium text-white">Reject black frames</span>
                    <span className="mt-1 block text-xs text-slate-500">Useful for accidental lens covers or dead frames.</span>
                  </span>
                  <input
                    type="checkbox"
                    checked={draft.exclude_black}
                    onChange={(event) => update("exclude_black", event.target.checked)}
                    className="h-5 w-5 accent-accent-cyan"
                  />
                </label>
                <label className="space-y-2 text-sm text-slate-300">
                  Sweep sample frames
                  <Input
                    type="number"
                    min={20}
                    max={500}
                    value={draft.sweep_sample_frames}
                    onChange={(event) => update("sweep_sample_frames", Number(event.target.value))}
                  />
                </label>
              </div>

              <div className="space-y-4">
                {probeLoading && displayProbeItems.length ? (
                  <div className="text-xs text-accent-blue">Refreshing metadata...</div>
                ) : null}
                {displayProbeItems.length ? (
                  displayProbeItems.map((item) => (
                    <div
                      key={item.path}
                      className="rounded-3xl border border-white/8 bg-white/[0.03] p-4"
                    >
                      <div className="min-w-0">
                        <div className="min-w-0">
                          <div className="font-medium text-white">{item.name}</div>
                          <div className="mt-1 text-xs text-slate-500">{item.path}</div>
                          <div className="mt-3 grid gap-3 md:grid-cols-4">
                            <div>
                              <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
                                Total frames
                              </div>
                              <div className="mt-1 text-lg font-semibold text-white">
                                {item.total_frames}
                              </div>
                            </div>
                            <div>
                              <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
                                Native FPS
                              </div>
                              <div className="mt-1 text-lg font-semibold text-white">
                                {item.native_fps ? item.native_fps.toFixed(2) : "N/A"}
                              </div>
                            </div>
                            <div>
                              <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
                                Estimated frames
                              </div>
                              <div className="mt-1 text-lg font-semibold text-white">
                                {item.estimated_frames}
                              </div>
                            </div>
                            <div>
                              <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
                                Sampling step
                              </div>
                              <div className="mt-1 text-lg font-semibold text-white">
                                {item.step ?? 1}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-sm text-slate-500">
                    {probeLoading
                      ? "Reading dataset metadata..."
                      : "Select datasets in Source first to see frame estimates here."}
                  </div>
                )}
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-5">
              <div>
                <CardTitle>Color</CardTitle>
                <CardDescription>
                  Enable color conversion, then choose one input and one output. Input stays on
                  auto-detect unless you need an explicit OCIO colorspace.
                </CardDescription>
              </div>

              <label className="flex items-center justify-between gap-4 rounded-3xl border border-white/8 bg-white/[0.03] px-5 py-4 text-sm text-slate-200">
                <span>
                  <span className="block font-medium text-white">Enable color conversion</span>
                  <span className="mt-1 block text-xs text-slate-500">
                    Kept separate from input/output so the color pipeline can stay off cleanly.
                  </span>
                </span>
                <input
                  type="checkbox"
                  checked={draft.color_enabled}
                  onChange={(event) => update("color_enabled", event.target.checked)}
                  className="h-5 w-5 accent-accent-cyan"
                />
              </label>

              {draft.input_mode === "rescan" ? (
                <label className="flex items-center justify-between gap-4 rounded-3xl border border-white/8 bg-white/[0.03] px-5 py-4 text-sm text-slate-200">
                  <span>
                    <span className="block font-medium text-white">
                      Generate ACEScg EXR dataset output
                    </span>
                    <span className="mt-1 block text-xs text-slate-500">
                      On: final images are ACES - ACEScg EXR and images.bin is patched. Off: PNG-only pipeline.
                    </span>
                  </span>
                  <input
                    type="checkbox"
                    checked={draft.use_acescg_exr}
                    onChange={(event) => update("use_acescg_exr", event.target.checked)}
                    className="h-5 w-5 accent-accent-cyan"
                  />
                </label>
              ) : null}

              <div className="grid gap-4 md:grid-cols-[1fr_auto]">
                <label className="space-y-2 text-sm text-slate-300">
                  OCIO Config
                  <select
                    value={draft.ocio_path}
                    onChange={async (event) => {
                      update("ocio_path", event.target.value);
                      await onRefreshOcioOptions(event.target.value);
                    }}
                    className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
                  >
                    <option value="">No config detected</option>
                    {options.ocio_configs.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="flex items-end">
                  <Button variant="info" onClick={() => pickFolder("ocio")}>
                    Browse OCIO
                  </Button>
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <label className="space-y-2 text-sm text-slate-300">
                  Input
                  <select
                    value={colorInputValue}
                    onChange={(event) => updateColorInput(event.target.value)}
                    className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
                  >
                    <option value="builtin:Auto-detect">Auto-detect</option>
                    {options.color_sources
                      .filter((option) => option !== "Auto-detect")
                      .map((option) => (
                        <option key={option} value={`builtin:${option}`}>
                          {option}
                        </option>
                      ))}
                    {options.ocio_spaces.length ? (
                      <option disabled>-- OCIO spaces --</option>
                    ) : null}
                    {options.ocio_spaces.map((option) => (
                      <option key={option} value={`ocio:${option}`}>
                        {option}
                      </option>
                    ))}
                  </select>
                  {draft.color_dest === "Custom OCIO..." && !draft.ocio_in_cs ? (
                    <span className="text-xs text-amber-200">
                      Choose an OCIO input space before running a custom OCIO output.
                    </span>
                  ) : null}
                </label>
                <label className="space-y-2 text-sm text-slate-300">
                  Output
                  <select
                    value={colorOutputValue}
                    onChange={(event) => updateColorOutput(event.target.value)}
                    className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
                  >
                    <option value="" disabled>
                      Select OCIO output
                    </option>
                    {builtinColorOutputs.map((option) => (
                      <option key={option} value={`builtin:${option}`}>
                        {option}
                      </option>
                    ))}
                    {options.ocio_spaces.length ? (
                      <option disabled>-- OCIO spaces --</option>
                    ) : null}
                    {options.ocio_spaces.map((option) => (
                      <option key={option} value={`ocio:${option}`}>
                        {option}
                      </option>
                    ))}
                  </select>
                  {draft.color_dest === "ACEScg (EXR + sRGB PNG)" &&
                  options.ocio_spaces.includes(ACESCG_OCIO_SPACE) ? (
                    <span className="text-xs text-slate-500">
                      ACEScg is resolved as "{ACESCG_OCIO_SPACE}" in the OCIO config.
                    </span>
                  ) : null}
                </label>
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="space-y-5">
              <div>
                <CardTitle>Reconstruction</CardTitle>
                <CardDescription>
                  Reconstruction settings are now dropdown-driven and bounded by detected machine
                  capabilities.
                </CardDescription>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <label className="space-y-2 text-sm text-slate-300">
                  Features
                  <select
                    value={draft.feature_type}
                    onChange={(event) => update("feature_type", event.target.value)}
                    className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
                  >
                    {options.features.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-2 text-sm text-slate-300">
                  Matcher
                  <select
                    value={draft.matcher_type}
                    onChange={(event) => update("matcher_type", event.target.value)}
                    className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
                  >
                    {options.matchers.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-2 text-sm text-slate-300">
                  Pair Strategy
                  <select
                    value={draft.pairing_mode}
                    onChange={(event) => update("pairing_mode", event.target.value)}
                    className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
                  >
                    {options.pairing_modes.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-2 text-sm text-slate-300">
                  Mapper
                  <select
                    value={draft.mapper_type}
                    onChange={(event) => update("mapper_type", event.target.value)}
                    className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
                  >
                    {options.mapper_types.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-2 text-sm text-slate-300">
                  Camera Model
                  <select
                    value={draft.camera_model}
                    onChange={(event) => update("camera_model", event.target.value)}
                    className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
                  >
                    {options.camera_models.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-2 text-sm text-slate-300">
                  Max Keypoints
                  <Input
                    type="number"
                    min={256}
                    step={256}
                    value={draft.max_keypoints}
                    onChange={(event) => update("max_keypoints", Number(event.target.value))}
                  />
                </label>
              </div>

              <div className="rounded-3xl border border-white/8 bg-white/[0.03] p-4">
                <div className="mb-2 flex items-center justify-between text-sm text-slate-300">
                  <span>Worker Threads</span>
                  <span>{draft.num_workers}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={workerMax}
                  step={1}
                  value={draft.num_workers}
                  onChange={(event) => update("num_workers", Number(event.target.value))}
                  className="w-full accent-accent-cyan"
                />
                <div className="mt-2 text-xs text-slate-500">
                  Bound to detected CPU core count ({workerMax}).
                </div>
              </div>
            </div>
          )}

          {step === 4 && (
            <div className="space-y-5">
              <div>
                <CardTitle>Review & Queue</CardTitle>
                <CardDescription>
                  Selected datasets will be queued and executed one after another.
                </CardDescription>
              </div>

              <div className="grid gap-4 md:grid-cols-4">
                <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                  <div className="text-sm text-slate-400">Queued jobs</div>
                  <div className="mt-2 text-2xl font-semibold text-white">{queuePreview.length}</div>
                </div>
                <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                  <div className="text-sm text-slate-400">Frames / pairs</div>
                  <div className="mt-2 text-lg font-semibold text-white">
                    {estimateLoading ? "..." : `${estimate?.total_frames ?? 0} / ${estimate?.total_pairs ?? 0}`}
                  </div>
                </div>
                <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                  <div className="flex items-center gap-2 text-sm text-slate-400">
                    <HardDrive size={15} />
                    Disk
                  </div>
                  <div className="mt-2 text-lg font-semibold text-white">{formatBytes(estimate?.total_disk_bytes)}</div>
                </div>
                <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                  <div className="flex items-center gap-2 text-sm text-slate-400">
                    <Clock3 size={15} />
                    Duration
                  </div>
                  <div className="mt-2 text-lg font-semibold text-white">{formatDuration(estimate?.total_seconds)}</div>
                </div>
              </div>

              {estimate?.warnings.length ? (
                <div className="space-y-2 rounded-2xl border border-accent-amber/25 bg-accent-amber/10 p-4">
                  {Array.from(new Set(estimate.warnings)).map((warning) => (
                    <div key={warning} className="flex items-start gap-2 text-sm text-amber-100">
                      <AlertTriangle size={15} className="mt-0.5 shrink-0" />
                      <span>{warning}</span>
                    </div>
                  ))}
                </div>
              ) : null}

              <div className="space-y-3">
                {queuePreview.map((item) => (
                  <div
                    key={`${item.input_paths[0]}-${item.output_path}`}
                    className="rounded-2xl border border-white/8 bg-graphite-950/70 px-4 py-3"
                  >
                    <div className="font-medium text-white">{item.label || basename(item.input_paths[0])}</div>
                    <div className="mt-1 text-xs text-slate-500">{item.output_path}</div>
                    <div className="mt-2 text-sm text-slate-300">
                      FPS: {item.fps_extract.toFixed(1)} · Mapper: {item.mapper_type}
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex flex-wrap gap-3">
                <Button variant="success" onClick={handleSubmit} disabled={submitting || !queuePreview.length}>
                  {submitting ? "Queueing..." : `Queue ${queuePreview.length} job(s)`}
                  <ArrowRight className="ml-2" size={16} />
                </Button>
                <Button variant="info" onClick={handleQualitySweep} disabled={submitting || !queuePreview.length}>
                  <Wand2 className="mr-2" size={16} />
                  Queue quality sweep
                </Button>
              </div>
            </div>
          )}
            </motion.div>
          </AnimatePresence>
        </Card>

        <Card className="min-w-0 space-y-4">
          <div className="flex items-start justify-between gap-3">
            <div>
            <CardTitle>Live Summary</CardTitle>
            <CardDescription>
              {queuePreview.length} job(s) ready.
            </CardDescription>
            </div>
            <Button
              variant={showRequestPayload ? "info" : "secondary"}
              onClick={() => setShowRequestPayload((value) => !value)}
            >
              <FileJson size={16} className="mr-2" />
              Request
            </Button>
          </div>
          {showRequestPayload ? (
            <pre className="max-h-[min(560px,50vh)] overflow-auto rounded-2xl bg-graphite-950/90 p-4 font-mono text-xs text-slate-300">
{JSON.stringify(queuePreview, null, 2)}
            </pre>
          ) : null}
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-slate-500">Estimate</div>
              <div className="mt-2 text-sm text-slate-300">
                {estimateLoading
                  ? "Refreshing..."
                  : `${estimate?.total_frames ?? 0} frames, ${formatDuration(estimate?.total_seconds)}`}
              </div>
            </div>
            <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-slate-500">Confidence</div>
              <div className="mt-2 text-sm capitalize text-slate-300">
                {estimate?.estimates[0]?.confidence ?? "fallback"}
              </div>
            </div>
          </div>
        </Card>
      </div>

      <div className="flex items-center justify-between rounded-3xl border border-white/8 bg-white/[0.03] px-5 py-4">
        <div className="text-sm text-slate-400">
          Step {step + 1} of {steps.length}
        </div>
        <div className="flex gap-3">
          <Button variant="warning" disabled={step === 0} onClick={() => setStep((value) => value - 1)}>
            Previous
          </Button>
          <Button
            variant="success"
            onClick={() => setStep((value) => Math.min(steps.length - 1, value + 1))}
            disabled={step === steps.length - 1}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}
