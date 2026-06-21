import type { PropsWithChildren } from "react";
import { cn } from "../../lib/utils";

export function Badge({
  children,
  className,
}: PropsWithChildren<{ className?: string }>) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium uppercase tracking-[0.18em] text-slate-300",
        className,
      )}
    >
      {children}
    </span>
  );
}

