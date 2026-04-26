import { Copy, FolderOpen, KeyRound, Play, QrCode, Radio, RefreshCw, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import { desktop } from "../../lib/desktop";
import type { ServerConfig, ServerState } from "../../lib/types";
import { Button } from "../ui/button";
import { Card, CardDescription, CardTitle } from "../ui/card";
import { Input } from "../ui/input";

export function ServerView({
  state,
  fallbackConfig,
  onSave,
  onAction,
}: {
  state: ServerState | null;
  fallbackConfig: ServerConfig | null;
  onSave: (payload: Partial<ServerState["config"]>) => Promise<void>;
  onAction: (action: "start" | "stop" | "refresh") => Promise<void>;
}) {
  const [draft, setDraft] = useState<ServerConfig | null>(state?.config ?? fallbackConfig);

  useEffect(() => {
    setDraft(state?.config ?? fallbackConfig);
  }, [state, fallbackConfig]);

  if (!draft) {
    return <div className="text-slate-500">Preparing API server controls…</div>;
  }

  async function pickOutputDir() {
    const picked = await desktop.pickDirectory();
    if (picked) {
      setDraft((current) => (current ? { ...current, output_dir: picked } : current));
    }
  }
  const connectBaseUrl = state?.connect_url ?? `http://127.0.0.1:${draft.port}`;
  const rescanUrl = `remap://connect?baseUrl=${encodeURIComponent(connectBaseUrl)}&apiKey=${encodeURIComponent(draft.api_key)}`;

  return (
    <div className="grid min-w-0 gap-4 xl:grid-cols-[minmax(280px,0.8fr)_minmax(0,1.2fr)] xl:gap-6">
      <Card className="space-y-4">
        <div>
          <CardTitle>API Server</CardTitle>
          <CardDescription>
            Configure the ReScan-compatible `/api/v1` server, save the key, then launch it from here.
          </CardDescription>
        </div>

        <label className="space-y-2 text-sm text-slate-300">
          Host
          <Input
            value={draft.host}
            onChange={(event) => setDraft({ ...draft, host: event.target.value })}
          />
        </label>
        <label className="space-y-2 text-sm text-slate-300">
          Port
          <Input
            type="number"
            value={draft.port}
            onChange={(event) => setDraft({ ...draft, port: Number(event.target.value) })}
          />
        </label>
        <label className="space-y-2 text-sm text-slate-300">
          API Key
          <Input
            value={draft.api_key}
            onChange={(event) => setDraft({ ...draft, api_key: event.target.value })}
          />
        </label>
        <label className="space-y-2 text-sm text-slate-300">
          Output Folder
          <div className="flex gap-3">
            <Input
              value={draft.output_dir}
              readOnly
              placeholder="Default: system temp folder"
            />
            <Button variant="info" onClick={pickOutputDir}>
              <FolderOpen size={16} className="mr-2" />
              Browse
            </Button>
          </div>
        </label>

        <div className="flex flex-wrap gap-3">
          <Button variant="success" onClick={() => onSave(draft)}>
            <KeyRound size={16} className="mr-2" />
            Save Config
          </Button>
          <Button variant="info" onClick={() => onAction("refresh")}>
            <RefreshCw size={16} className="mr-2" />
            Refresh
          </Button>
          {(state?.running ?? false) ? (
            <Button variant="danger" onClick={() => onAction("stop")}>
              <Square size={14} className="mr-2" />
              Stop Server
            </Button>
          ) : (
            <Button variant="success" onClick={() => onAction("start")}>
              <Play size={14} className="mr-2" />
              Start Server
            </Button>
          )}
        </div>
      </Card>

      <div className="space-y-6">
        <Card>
          <div className="flex items-start gap-3">
            <div className="rounded-2xl bg-white/6 p-3 text-accent-cyan">
              <Radio size={18} />
            </div>
            <div>
              <CardTitle>Server State</CardTitle>
              <CardDescription>
                {state
                  ? state.running
                    ? "Remote server process is running."
                    : "Remote server is currently stopped."
                  : "State unavailable until the backend responds."}
              </CardDescription>
            </div>
          </div>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
              <div className="text-sm text-slate-500">Health URL</div>
              <div className="mt-2 break-all text-sm text-slate-300">
                {state?.health.url ?? `http://127.0.0.1:${draft.port}/api/v1/health`}
              </div>
            </div>
            <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
              <div className="text-sm text-slate-500">Reachable</div>
              <div className="mt-2 text-lg font-semibold">
                {state?.health.reachable ? "Yes" : "No"}
              </div>
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-start gap-3">
            <div className="rounded-2xl bg-white/6 p-3 text-accent-cyan">
              <QrCode size={18} />
            </div>
            <div>
              <CardTitle>ReScan Quick Connect</CardTitle>
              <CardDescription>
                Scan from ReScan or copy the local URL and API key manually.
              </CardDescription>
            </div>
          </div>
          <div className="mt-4 grid gap-4 md:grid-cols-[auto_1fr]">
            <div className="rounded-2xl bg-white p-3">
              <QRCodeSVG value={rescanUrl} size={156} />
            </div>
            <div className="space-y-3">
              <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                <div className="text-sm text-slate-500">Local URL</div>
                <div className="mt-2 break-all text-sm text-slate-200">{connectBaseUrl}</div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant="secondary" onClick={() => navigator.clipboard.writeText(connectBaseUrl)}>
                  <Copy size={16} className="mr-2" />
                  Copy URL
                </Button>
                <Button variant="secondary" onClick={() => navigator.clipboard.writeText(draft.api_key)}>
                  <Copy size={16} className="mr-2" />
                  Copy key
                </Button>
              </div>
            </div>
          </div>
        </Card>

        <Card>
          <CardTitle>Remote Jobs</CardTitle>
          <CardDescription>
            When the server is reachable, this mirrors `/api/v1/jobs`.
          </CardDescription>
          <pre className="mt-4 max-h-[420px] overflow-auto rounded-2xl bg-graphite-950/90 p-4 font-mono text-xs text-slate-300">
{JSON.stringify(state?.remote_jobs ?? [], null, 2)}
          </pre>
        </Card>
      </div>
    </div>
  );
}
