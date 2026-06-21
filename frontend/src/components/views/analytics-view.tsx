import {
  Activity,
  BarChart3,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  MemoryStick,
  RefreshCw,
  Thermometer,
  Timer,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../../lib/api";
import type { AnalyticsPayload, AnalyticsRecentJob } from "../../lib/types";
import { Button } from "../ui/button";
import { Card, CardDescription, CardTitle } from "../ui/card";
import { Progress } from "../ui/progress";

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatNumber(value: number | null | undefined, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-";
  }
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatMs(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "-";
  }
  if (value >= 1000) {
    return `${formatNumber(value / 1000, 2)} s`;
  }
  return `${formatNumber(value, value >= 10 ? 0 : 1)} ms`;
}

function formatDuration(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "-";
  }
  if (value < 60) {
    return `${formatNumber(value, 1)} s`;
  }
  if (value < 3600) {
    return `${formatNumber(value / 60, 1)} min`;
  }
  return `${formatNumber(value / 3600, 1)} h`;
}

function topEntries(values: Record<string, number>, limit = 5) {
  return Object.entries(values)
    .filter(([name]) => name)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit);
}

function Metric({
  icon: Icon,
  label,
  value,
  detail,
  tone = "text-accent-cyan",
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  detail?: string;
  tone?: string;
}) {
  return (
    <div className="min-w-0 rounded-2xl border border-white/8 bg-white/[0.035] p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="truncate text-sm text-slate-500">{label}</div>
        <Icon size={18} className={tone} />
      </div>
      <div className="truncate text-2xl font-semibold text-white">{value}</div>
      {detail ? <div className="mt-2 truncate text-xs text-slate-400">{detail}</div> : null}
    </div>
  );
}

function UsageBars({ title, values }: { title: string; values: Record<string, number> }) {
  const entries = topEntries(values);
  const max = Math.max(...entries.map(([, value]) => value), 1);
  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.035] p-4">
      <div className="mb-4 flex items-center gap-2 text-sm font-medium text-slate-200">
        <BarChart3 size={16} className="text-accent-blue" />
        {title}
      </div>
      <div className="space-y-3">
        {entries.length ? entries.map(([name, value]) => (
          <div key={name} className="min-w-0">
            <div className="mb-1 flex items-center justify-between gap-3 text-xs">
              <span className="truncate text-slate-300">{name}</span>
              <span className="shrink-0 text-slate-500">{value}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-white/8">
              <div className="h-full rounded-full bg-accent-blue" style={{ width: `${(value / max) * 100}%` }} />
            </div>
          </div>
        )) : (
          <div className="rounded-xl border border-white/8 bg-graphite-950/60 px-3 py-3 text-sm text-slate-500">
            No data yet.
          </div>
        )}
      </div>
    </div>
  );
}

function StepStrip({ job }: { job: AnalyticsRecentJob }) {
  const entries = Object.entries(job.step_seconds ?? {}).filter(([, value]) => value > 0);
  const max = Math.max(...entries.map(([, value]) => value), 1);
  if (!entries.length) {
    return <span className="text-xs text-slate-600">-</span>;
  }
  return (
    <div className="flex min-w-[160px] max-w-[260px] gap-1">
      {entries.slice(0, 6).map(([name, value]) => (
        <div
          key={name}
          className="h-2 rounded-full bg-accent-cyan"
          title={`${name}: ${formatDuration(value)}`}
          style={{ width: `${Math.max(10, (value / max) * 70)}px`, opacity: 0.38 + (value / max) * 0.55 }}
        />
      ))}
    </div>
  );
}

export function AnalyticsView() {
  const [analytics, setAnalytics] = useState<AnalyticsPayload | null>(null);
  const [error, setError] = useState("");

  async function refresh() {
    try {
      setAnalytics(await api.getAnalytics());
      setError("");
    } catch (err) {
      setError(String(err));
    }
  }

  useEffect(() => {
    refresh().catch(console.error);
    const timer = window.setInterval(() => refresh().catch(console.error), 2000);
    return () => window.clearInterval(timer);
  }, []);

  const rows = useMemo(() => analytics?.jobs.recent ?? [], [analytics]);
  const cpu = numberValue(analytics?.system.cpu_percent);
  const ram = numberValue(analytics?.system.ram_percent);
  const disk = numberValue(analytics?.system.disk_percent);
  const gpuMemory = numberValue(analytics?.gpu.memory_used_mb ?? analytics?.gpu.torch_reserved_mb);
  const gpuTotal = numberValue(analytics?.gpu.memory_total_mb);
  const gpuTemp = numberValue(analytics?.gpu.temperature_c);
  const maxTemp = numberValue(analytics?.maxima.gpu_temperature_c);

  return (
    <div className="grid min-w-0 gap-4 xl:gap-6">
      <Card>
        <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
          <div>
            <CardTitle>Analytics</CardTitle>
            <CardDescription>
              {analytics ? `Updated ${new Date(analytics.generated_at).toLocaleTimeString()}` : "Waiting for backend data"}
            </CardDescription>
          </div>
          <Button variant="secondary" onClick={() => refresh().catch(console.error)} title="Refresh">
            <RefreshCw size={16} />
          </Button>
        </div>
        {error ? (
          <div className="mb-4 rounded-2xl border border-accent-red/30 bg-accent-red/10 px-4 py-3 text-sm text-rose-100">
            {error}
          </div>
        ) : null}

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <Metric icon={Cpu} label="CPU" value={`${formatNumber(cpu, 0)}%`} detail={`Peak ${formatNumber(numberValue(analytics?.maxima.cpu_percent), 0)}%`} />
          <Metric icon={MemoryStick} label="RAM" value={`${formatNumber(ram, 0)}%`} detail={`${formatNumber(numberValue(analytics?.system.ram_used_gb), 1)} / ${formatNumber(numberValue(analytics?.system.ram_total_gb), 1)} GB`} tone="text-accent-emerald" />
          <Metric icon={Zap} label="GPU VRAM" value={gpuTotal ? `${formatNumber(gpuMemory, 0)} / ${formatNumber(gpuTotal, 0)} MB` : `${formatNumber(gpuMemory, 0)} MB`} detail={`${analytics?.gpu.name ?? "No CUDA GPU"} ${analytics?.gpu.capability ?? ""}`} tone="text-accent-amber" />
          <Metric icon={Thermometer} label="GPU Temp" value={`${formatNumber(gpuTemp, 0)} C`} detail={`Max ${formatNumber(maxTemp, 0)} C`} tone="text-rose-200" />
          <Metric icon={HardDrive} label="Storage" value={`${formatNumber(disk, 0)}%`} detail={`${formatNumber(numberValue(analytics?.system.disk_used_gb), 1)} / ${formatNumber(numberValue(analytics?.system.disk_total_gb), 1)} GB`} />
          <Metric icon={Activity} label="Jobs" value={`${analytics?.jobs.active ?? 0} active`} detail={`${analytics?.jobs.completed ?? 0} completed - ${analytics?.jobs.failed ?? 0} failed`} tone="text-accent-emerald" />
          <Metric icon={Database} label="Features" value={formatNumber(numberValue(analytics?.throughput.features_observed), 0)} detail={`${formatNumber(numberValue(analytics?.throughput.frames_observed), 0)} frames observed`} tone="text-accent-blue" />
          <Metric icon={Timer} label="Speed" value={formatMs(numberValue(analytics?.throughput.avg_feature_ms_per_frame))} detail={`${formatMs(numberValue(analytics?.throughput.avg_match_ms_per_pair))} / pair`} tone="text-accent-cyan" />
        </div>
      </Card>

      <div className="grid gap-4 xl:grid-cols-3">
        <UsageBars title="Matchers" values={analytics?.usage.matchers ?? {}} />
        <UsageBars title="Features" values={analytics?.usage.features ?? {}} />
        <UsageBars title="Inputs" values={analytics?.usage.input_modes ?? {}} />
      </div>

      <Card>
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <CardTitle>Recent Jobs</CardTitle>
            <CardDescription>{rows.length} sampled outputs</CardDescription>
          </div>
          <Gauge size={20} className="text-accent-cyan" />
        </div>
        <div className="overflow-auto">
          <table className="w-full min-w-[900px] border-separate border-spacing-y-2 text-left text-sm">
            <thead className="text-xs uppercase tracking-[0.12em] text-slate-500">
              <tr>
                <th className="px-3 py-2 font-medium">Job</th>
                <th className="px-3 py-2 font-medium">Matcher</th>
                <th className="px-3 py-2 font-medium">Frames</th>
                <th className="px-3 py-2 font-medium">Features</th>
                <th className="px-3 py-2 font-medium">Matches</th>
                <th className="px-3 py-2 font-medium">Points</th>
                <th className="px-3 py-2 font-medium">Timing</th>
                <th className="px-3 py-2 font-medium">Size</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((job) => (
                <tr key={job.job_id} className="bg-white/[0.035] text-slate-200">
                  <td className="rounded-l-2xl px-3 py-3">
                    <div className="max-w-[220px] truncate font-medium text-white">{job.label || job.job_id}</div>
                    <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
                      <span>{job.status}</span>
                      <div className="w-[90px]">
                        <Progress value={job.progress} />
                      </div>
                    </div>
                  </td>
                  <td className="px-3 py-3">{job.matcher_type || "-"}</td>
                  <td className="px-3 py-3">{formatNumber(job.frames)}</td>
                  <td className="px-3 py-3">{formatNumber(job.features)}</td>
                  <td className="px-3 py-3">{formatNumber(job.matches)}</td>
                  <td className="px-3 py-3">{formatNumber(job.points3d)}</td>
                  <td className="px-3 py-3">
                    <div className="space-y-1">
                      <StepStrip job={job} />
                      <div className="text-xs text-slate-500">
                        {formatDuration(job.duration_seconds)} - {formatMs(job.feature_ms_per_frame)} / frame
                      </div>
                    </div>
                  </td>
                  <td className="rounded-r-2xl px-3 py-3">{formatNumber(job.output_size_mb, 1)} MB</td>
                </tr>
              ))}
              {!rows.length ? (
                <tr>
                  <td colSpan={8} className="rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-8 text-center text-slate-500">
                    No jobs sampled yet.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
