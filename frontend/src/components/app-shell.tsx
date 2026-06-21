import { motion } from "framer-motion";
import { BarChart3, Cog, FolderKanban, Radio, Sparkles } from "lucide-react";
import type { PropsWithChildren } from "react";
import { Badge } from "./ui/badge";
import type { JobSummary } from "../lib/types";
import { cn } from "../lib/utils";

export type AppView = "new-job" | "jobs" | "analytics" | "server" | "settings";

const navItems: Array<{
  id: AppView;
  label: string;
  caption: string;
  icon: typeof Sparkles;
  tone: string;
}> = [
  { id: "new-job", label: "New Job", caption: "Wizard-driven creation", icon: Sparkles, tone: "text-accent-cyan" },
  { id: "jobs", label: "Jobs", caption: "Live progress and logs", icon: FolderKanban, tone: "text-accent-emerald" },
  { id: "analytics", label: "Analytics", caption: "Hardware and matching stats", icon: BarChart3, tone: "text-accent-blue" },
  { id: "server", label: "API Server", caption: "Remote ReScan bridge", icon: Radio, tone: "text-accent-blue" },
  { id: "settings", label: "Settings", caption: "Defaults and capabilities", icon: Cog, tone: "text-accent-amber" },
];

export function AppShell({
  activeView,
  onViewChange,
  jobs = [],
  children,
}: PropsWithChildren<{
  activeView: AppView;
  onViewChange: (view: AppView) => void;
  jobs?: JobSummary[];
}>) {
  const liveJobs = jobs.filter((job) => ["processing", "queued", "paused"].includes(job.status));
  const processingJobs = jobs.filter((job) => job.status === "processing");
  const jobsProgress = liveJobs.length
    ? Math.round(liveJobs.reduce((total, job) => total + job.progress, 0) / liveJobs.length)
    : 0;

  return (
    <div className="min-h-screen px-3 py-3 text-white sm:px-4 sm:py-4 xl:px-6 xl:py-6">
      <div className="grid min-h-[calc(100vh-1.5rem)] min-w-0 gap-4 lg:min-h-[calc(100vh-2rem)] lg:grid-cols-[minmax(220px,280px)_minmax(0,1fr)] xl:min-h-[calc(100vh-3rem)] xl:gap-6">
        <aside className="min-w-0 rounded-[24px] border border-white/8 bg-graphite-900/78 p-4 shadow-glow backdrop-blur-xl xl:rounded-[32px] xl:p-5">
          <div className="mb-4 grid items-center gap-4 sm:grid-cols-[auto_1fr] lg:mb-8 lg:block">
            <div>
              <Badge className="mb-3 border-accent-cyan/30 bg-accent-cyan/8 text-accent-cyan lg:mb-4">
                ReMap Desktop
              </Badge>
              <img
                src="/ReMap_logo.png"
                alt="ReMap"
                className="h-auto w-32 max-w-full sm:w-40 lg:mb-5 lg:w-full lg:max-w-[180px]"
              />
            </div>
            <div className="min-w-0">
              <h1 className="font-display text-xl font-semibold leading-tight sm:text-2xl lg:text-3xl">
                Creative pipeline control for Gaussian Splatting.
              </h1>
              <p className="mt-2 text-sm text-slate-400 lg:mt-3">
                New desktop shell powered by React, Tauri, and a headless Python backend.
              </p>
            </div>
          </div>

          <nav className="grid gap-2 sm:grid-cols-2 lg:grid-cols-1">
            {navItems.map(({ id, label, caption, icon: Icon, tone }) => {
              const isJobs = id === "jobs";
              const jobsActive = isJobs && liveJobs.length > 0;
              const shownCaption = jobsActive
                ? `${processingJobs.length || liveJobs.length} active - ${jobsProgress}%`
                : caption;

              return (
              <button
                key={id}
                onClick={() => onViewChange(id)}
                className={cn(
                  "group relative w-full min-w-0 overflow-hidden rounded-2xl border px-3 py-3 text-left transition duration-200 hover:-translate-y-0.5 active:translate-y-0 xl:px-4 xl:py-4",
                  activeView === id
                    ? "border-white/[0.18] bg-white/[0.07]"
                    : "border-white/6 bg-white/[0.02] hover:border-white/12 hover:bg-white/[0.05]",
                )}
              >
                {activeView === id && (
                  <motion.div
                    layoutId="nav-highlight"
                    className="absolute inset-0 rounded-2xl border border-white/[0.14] bg-gradient-to-r from-white/8 to-transparent"
                  />
                )}
                {jobsActive ? (
                  <motion.div
                    className="absolute inset-0 bg-accent-emerald/5"
                    animate={{ opacity: [0.2, 0.58, 0.2] }}
                    transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
                  />
                ) : null}
                <div className="relative flex items-start gap-3">
                  <div className={cn("relative rounded-2xl bg-white/6 p-2", tone)}>
                    <Icon size={18} />
                    {jobsActive ? (
                      <motion.span
                        className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full bg-accent-emerald"
                        animate={{ scale: [1, 1.55, 1], opacity: [0.8, 0.25, 0.8] }}
                        transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
                      />
                    ) : null}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-3">
                      <div className="font-medium text-white">{label}</div>
                      {jobsActive ? (
                        <div className="text-xs font-semibold text-accent-emerald">{jobsProgress}%</div>
                      ) : null}
                    </div>
                    <div className="mt-1 text-sm text-slate-400">{shownCaption}</div>
                    {jobsActive ? (
                      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/8">
                        <motion.div
                          className="h-full rounded-full bg-accent-emerald"
                          initial={false}
                          animate={{ width: `${Math.max(4, jobsProgress)}%` }}
                          transition={{ duration: 0.35, ease: "easeOut" }}
                        />
                      </div>
                    ) : null}
                  </div>
                </div>
              </button>
              );
            })}
          </nav>
        </aside>

        <main className="min-w-0 rounded-[24px] border border-white/8 bg-gradient-to-b from-graphite-900/82 to-graphite-950/82 p-4 shadow-glow backdrop-blur-xl xl:rounded-[32px] xl:p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
