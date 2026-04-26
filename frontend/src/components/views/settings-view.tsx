import { useEffect, useRef, useState } from "react";
import { Database, Download, PackageCheck, RefreshCw, Trash2, Wrench } from "lucide-react";
import { api } from "../../lib/api";
import type { AppSettings, CacheStatus, CapabilitiesPayload, DependencyStatus, OptionsPayload } from "../../lib/types";
import { Button } from "../ui/button";
import { Card, CardDescription, CardTitle } from "../ui/card";
import { Input } from "../ui/input";

const themeOptions = [
  { value: "graphite", label: "Graphite" },
  { value: "midnight", label: "Midnight" },
  { value: "copper", label: "Copper" },
  { value: "forest", label: "Forest" },
  { value: "mono", label: "Monochrome" },
];

const SETTINGS_DRAFT_KEY = "remap:settings-draft";

function formatMb(value?: number | null) {
  if (!value) {
    return "-";
  }
  if (value >= 1024) {
    return `${(value / 1024).toFixed(1)} GB`;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} MB`;
}

function readSettingsDraft(): AppSettings | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(SETTINGS_DRAFT_KEY);
    return raw ? (JSON.parse(raw) as AppSettings) : null;
  } catch {
    return null;
  }
}

export function SettingsView({
  settings,
  capabilities,
  options,
  onSave,
}: {
  settings: AppSettings | null;
  capabilities: CapabilitiesPayload | null;
  options: OptionsPayload;
  onSave: (payload: Partial<AppSettings>) => Promise<void>;
}) {
  const restoredDraftRef = useRef(false);
  const [draft, setDraft] = useState<AppSettings | null>(() => readSettingsDraft() ?? settings);
  const [dependencies, setDependencies] = useState<DependencyStatus | null>(null);
  const [cache, setCache] = useState<CacheStatus | null>(null);
  const [dependencyError, setDependencyError] = useState("");

  useEffect(() => {
    if (!settings) {
      return;
    }
    if (!restoredDraftRef.current) {
      setDraft(readSettingsDraft() ?? settings);
      restoredDraftRef.current = true;
      return;
    }
    setDraft((current) => current ?? settings);
  }, [settings]);

  useEffect(() => {
    if (typeof window === "undefined" || !draft) {
      return;
    }
    window.localStorage.setItem(SETTINGS_DRAFT_KEY, JSON.stringify(draft));
  }, [draft]);

  async function refreshDependencies() {
    try {
      setDependencies(await api.getDependencies());
      setDependencyError("");
    } catch (error) {
      setDependencyError(String(error));
    }
  }

  async function refreshCache() {
    setCache(await api.getCache());
  }

  async function clearCache() {
    await api.clearCache();
    await refreshCache();
  }

  async function runDependencyAction(action: string) {
    try {
      setDependencies(await api.runDependencyAction(action));
      setDependencyError("");
    } catch (error) {
      setDependencyError(String(error));
    }
  }

  useEffect(() => {
    refreshDependencies().catch(console.error);
    refreshCache().catch(console.error);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (dependencies?.task.running) {
        refreshDependencies().catch(console.error);
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [dependencies?.task.running]);

  if (!draft) {
    return <div className="text-slate-500">Loading settings…</div>;
  }

  const workerMax = capabilities?.cpu_count ?? 16;
  const workerValue = Math.min(Math.max(1, draft.defaults.num_workers), workerMax);
  const mapperOptions = options.mapper_types.length
    ? options.mapper_types
    : capabilities?.glomap_available
      ? ["GLOMAP", "COLMAP"]
      : ["COLMAP", "GLOMAP"];
  const dependencyRunning = Boolean(dependencies?.task.running);
  const requiredPackages = dependencies?.packages ?? [];
  const installedCount = requiredPackages.filter((pkg) => pkg.installed).length;

  function saveDraft() {
    return onSave({
      ...draft,
      defaults: { ...draft.defaults, num_workers: workerValue },
    });
  }

  return (
    <div className="grid min-w-0 gap-4 xl:grid-cols-[minmax(280px,0.9fr)_minmax(0,1.1fr)] xl:gap-6">
      <Card className="space-y-4">
        <div>
          <CardTitle>Desktop Defaults</CardTitle>
          <CardDescription>
            Saved defaults are written to `backend_state/settings.json`, which is ignored by git.
          </CardDescription>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="space-y-2 text-sm text-slate-300">
            Theme
            <select
              value={draft.theme}
              onChange={(event) => setDraft({ ...draft, theme: event.target.value })}
              className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
            >
              {themeOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <div className="space-y-2 text-sm text-slate-300">
            <div className="flex items-center justify-between">
              <span>Default Workers</span>
              <span className="text-accent-cyan">{workerValue}</span>
            </div>
            <input
              type="range"
              min={1}
              max={workerMax}
              step={1}
              value={workerValue}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  defaults: { ...draft.defaults, num_workers: Number(event.target.value) },
                })
              }
              className="w-full accent-accent-cyan"
            />
            <div className="text-xs text-slate-500">
              CPU capacity detected: {workerMax} thread(s).
            </div>
          </div>
          <label className="space-y-2 text-sm text-slate-300">
            Default Mapper
            <select
              value={draft.defaults.mapper_type}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  defaults: { ...draft.defaults, mapper_type: event.target.value },
                })
              }
              className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
            >
              {mapperOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            {capabilities?.glomap_available ? (
              <span className="text-xs text-slate-500">
                GLOMAP detected and available as the preferred default.
              </span>
            ) : (
              <span className="text-xs text-slate-500">
                GLOMAP not detected, COLMAP remains the safe default.
              </span>
            )}
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            Default Camera Model
            <select
              value={draft.defaults.camera_model}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  defaults: { ...draft.defaults, camera_model: event.target.value },
                })
              }
              className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
            >
              {options.camera_models.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="space-y-2 text-sm text-slate-300">
            Default OCIO Config
            <select
              value={draft.defaults.ocio_path}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  defaults: { ...draft.defaults, ocio_path: event.target.value },
                })
              }
              className="h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-white outline-none"
            >
              <option value="">None</option>
              {options.ocio_configs.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            Default API Key
            <Input
              value={draft.server.api_key}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  server: { ...draft.server, api_key: event.target.value },
                })
              }
            />
          </label>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="flex items-center justify-between gap-4 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3 text-sm text-slate-200">
            <span>
              <span className="block font-medium text-white">In-app notifications</span>
              <span className="mt-1 block text-xs text-slate-500">Show toasts when jobs finish or fail.</span>
            </span>
            <input
              type="checkbox"
              checked={draft.notifications_enabled}
              onChange={(event) => setDraft({ ...draft, notifications_enabled: event.target.checked })}
              className="h-5 w-5 accent-accent-cyan"
            />
          </label>
          <label className="flex items-center justify-between gap-4 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3 text-sm text-slate-200">
            <span>
              <span className="block font-medium text-white">System notifications</span>
              <span className="mt-1 block text-xs text-slate-500">Request OS/browser notification permission.</span>
            </span>
            <input
              type="checkbox"
              checked={draft.system_notifications}
              onChange={(event) => setDraft({ ...draft, system_notifications: event.target.checked })}
              className="h-5 w-5 accent-accent-cyan"
            />
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            RAM warning limit (%)
            <Input
              type="number"
              value={draft.ram_limit_percent}
              onChange={(event) => setDraft({ ...draft, ram_limit_percent: Number(event.target.value), defaults: { ...draft.defaults, ram_limit_percent: Number(event.target.value) } })}
            />
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            GPU VRAM warning limit (%)
            <Input
              type="number"
              value={draft.gpu_vram_limit_percent}
              onChange={(event) => setDraft({ ...draft, gpu_vram_limit_percent: Number(event.target.value), defaults: { ...draft.defaults, gpu_vram_limit_percent: Number(event.target.value) } })}
            />
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            Blur threshold
            <Input
              type="number"
              value={draft.blur_threshold}
              onChange={(event) => setDraft({ ...draft, blur_threshold: Number(event.target.value), defaults: { ...draft.defaults, blur_threshold: Number(event.target.value) } })}
            />
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            Black frame threshold
            <Input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={draft.black_threshold}
              onChange={(event) => setDraft({ ...draft, black_threshold: Number(event.target.value), defaults: { ...draft.defaults, black_threshold: Number(event.target.value) } })}
            />
          </label>
        </div>

        <Button variant="success" onClick={saveDraft}>Save Defaults</Button>
      </Card>

      <Card className="space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <CardTitle>Pipeline Cache</CardTitle>
            <CardDescription>
              Global features/pairs/matches cache used by checkpointed reruns.
            </CardDescription>
          </div>
          <Button variant="secondary" onClick={() => refreshCache().catch(console.error)}>
            <RefreshCw size={16} />
          </Button>
        </div>
        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
            <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
              <Database size={16} className="text-accent-cyan" />
              Entries
            </div>
            <div className="text-2xl font-semibold text-white">{cache?.entries.length ?? 0}</div>
          </div>
          <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
            <div className="text-sm text-slate-400">Size</div>
            <div className="mt-2 text-2xl font-semibold text-white">{formatMb(cache?.total_size_mb)}</div>
          </div>
          <label className="space-y-2 rounded-2xl border border-white/8 bg-white/[0.03] p-4 text-sm text-slate-300">
            Max GB
            <Input
              type="number"
              value={draft.cache_max_size_gb}
              onChange={(event) => setDraft({ ...draft, cache_max_size_gb: Number(event.target.value) })}
            />
          </label>
        </div>
        <label className="flex items-center justify-between gap-4 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3 text-sm text-slate-200">
          <span>
            <span className="block font-medium text-white">Enable global cache</span>
            <span className="mt-1 block text-xs text-slate-500">{cache?.path ?? "backend_state/cache"}</span>
          </span>
          <input
            type="checkbox"
            checked={draft.cache_enabled}
            onChange={(event) => setDraft({ ...draft, cache_enabled: event.target.checked })}
            className="h-5 w-5 accent-accent-cyan"
          />
        </label>
        <div className="flex flex-wrap gap-2">
          <Button variant="danger" onClick={() => clearCache().catch(console.error)} disabled={!cache?.entries.length}>
            <Trash2 size={16} className="mr-2" />
            Clear cache
          </Button>
        </div>
        <div className="max-h-[220px] space-y-2 overflow-auto pr-1">
          {(cache?.entries ?? []).slice(0, 8).map((entry) => (
            <div key={entry.key} className="grid grid-cols-[1fr_auto] gap-3 rounded-xl bg-graphite-950/60 px-3 py-2 text-sm" title={entry.path}>
              <span className="truncate text-slate-200">{entry.files.join(", ") || entry.key}</span>
              <span className="text-xs text-slate-500">{formatMb(entry.size_mb)}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card className="space-y-5 xl:col-span-2">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <CardTitle>Dependencies & Models</CardTitle>
            <CardDescription>
              Stable package baseline, LoMa/LightGlue model cache, and one-click repair actions.
            </CardDescription>
          </div>
          <Button variant="secondary" onClick={() => refreshDependencies().catch(console.error)} title="Refresh dependency status">
            <RefreshCw size={16} />
          </Button>
        </div>

        {dependencyError ? (
          <div className="rounded-2xl border border-accent-red/30 bg-accent-red/10 px-4 py-3 text-sm text-rose-100">
            {dependencyError}
          </div>
        ) : null}

        <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
          <div className="space-y-4">
            <div className="grid gap-3 md:grid-cols-3">
              <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
                  <PackageCheck size={16} className="text-accent-emerald" />
                  Packages
                </div>
                <div className="text-2xl font-semibold text-white">
                  {installedCount}/{requiredPackages.length || "-"}
                </div>
              </div>
              <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
                  <Download size={16} className="text-accent-blue" />
                  Model Cache
                </div>
                <div className="text-2xl font-semibold text-white">
                  {formatMb(dependencies?.models.total_size_mb)}
                </div>
              </div>
              <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
                  <Wrench size={16} className="text-accent-amber" />
                  Task
                </div>
                <div className="truncate text-lg font-semibold capitalize text-white">
                  {dependencies?.task.status ?? "idle"}
                </div>
              </div>
            </div>

            <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
              <Button variant="success" disabled={dependencyRunning} onClick={() => runDependencyAction("install_packages")}>
                Install Stable
              </Button>
              <Button variant="warning" disabled={dependencyRunning} onClick={() => runDependencyAction("update_packages")}>
                Update Stable
              </Button>
              <Button variant="info" disabled={dependencyRunning} onClick={() => runDependencyAction("download_core_models")}>
                Core Models
              </Button>
              <Button variant="secondary" disabled={dependencyRunning} onClick={() => runDependencyAction("download_loma_b")}>
                LoMa-B
              </Button>
              <Button variant="secondary" disabled={dependencyRunning} onClick={() => runDependencyAction("download_loma_g")}>
                LoMa-G
              </Button>
              <Button variant="secondary" disabled={dependencyRunning} onClick={() => runDependencyAction("download_all_models")}>
                All Models
              </Button>
            </div>

            <div className="rounded-2xl border border-white/8 bg-graphite-950/70 p-4">
              <div className="mb-2 text-sm font-medium text-slate-300">Pinned Baseline</div>
              <div className="space-y-1 text-xs text-slate-500">
                <div className="truncate">Python: {dependencies?.python ?? "-"}</div>
                <div className="truncate">Requirements: {dependencies?.requirements ?? "-"}</div>
                <div className="truncate">Lock: {dependencies?.requirements_lock ?? "-"}</div>
                <div className="truncate">LoMa: {dependencies?.loma_locked_url ?? "-"}</div>
              </div>
            </div>
          </div>

          <div className="grid min-w-0 gap-4 lg:grid-cols-2">
            <div className="min-w-0 rounded-2xl border border-white/8 bg-white/[0.03] p-4">
              <div className="mb-3 text-sm font-medium text-slate-300">Installed Packages</div>
              <div className="max-h-[300px] space-y-2 overflow-auto pr-1">
                {requiredPackages.map((pkg) => (
                  <div key={pkg.name} className="grid grid-cols-[1fr_auto] gap-3 rounded-xl bg-graphite-950/60 px-3 py-2 text-sm">
                    <span className={pkg.installed ? "truncate text-slate-200" : "truncate text-rose-200"}>{pkg.name}</span>
                    <span className="text-xs text-slate-500">{pkg.version || "missing"}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="min-w-0 rounded-2xl border border-white/8 bg-white/[0.03] p-4">
              <div className="mb-3 text-sm font-medium text-slate-300">Weights Cache</div>
              <div className="max-h-[300px] space-y-2 overflow-auto pr-1">
                {[...(dependencies?.models.superglue_weights ?? []), ...(dependencies?.models.torch_hub_checkpoints ?? [])].map((item) => (
                  <div key={item.path} className="grid grid-cols-[1fr_auto] gap-3 rounded-xl bg-graphite-950/60 px-3 py-2 text-sm" title={item.path}>
                    <span className="truncate text-slate-200">{item.name}</span>
                    <span className="text-xs text-slate-500">{formatMb(item.size_mb)}</span>
                  </div>
                ))}
                {!dependencies?.models.superglue_weights.length && !dependencies?.models.torch_hub_checkpoints.length ? (
                  <div className="rounded-xl border border-white/8 bg-graphite-950/60 px-3 py-3 text-sm text-slate-500">
                    No cached model weights found yet.
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>

        <pre className="max-h-[260px] overflow-auto rounded-2xl bg-graphite-950/90 p-4 font-mono text-xs text-slate-300">
{(dependencies?.task.log ?? []).map((event) => `[${event.timestamp}] ${event.message}`).join("\n") || "No dependency task running."}
        </pre>
      </Card>

      <Card>
        <CardTitle>System Capabilities</CardTitle>
        <CardDescription>
          Loaded from the internal backend and used to clamp sliders and available options.
        </CardDescription>
        <pre className="mt-4 max-h-[560px] overflow-auto rounded-2xl bg-graphite-950/90 p-4 font-mono text-xs text-slate-300">
{JSON.stringify(capabilities, null, 2)}
        </pre>
      </Card>
    </div>
  );
}
