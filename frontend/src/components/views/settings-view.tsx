import { useEffect, useRef, useState } from "react";
import type { AppSettings, CapabilitiesPayload, OptionsPayload } from "../../lib/types";
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

        <Button variant="success" onClick={saveDraft}>Save Defaults</Button>
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
