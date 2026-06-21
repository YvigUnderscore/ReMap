import type { PropsWithChildren } from "react";
import { cn } from "../../lib/utils";

export function Card({
  children,
  className,
}: PropsWithChildren<{ className?: string }>) {
  return (
    <div
      className={cn(
        "min-w-0 rounded-3xl border border-white/8 bg-white/[0.035] p-5 shadow-glow backdrop-blur-xl",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function CardTitle({
  children,
  className,
}: PropsWithChildren<{ className?: string }>) {
  return (
    <h3 className={cn("font-display text-lg font-semibold text-white", className)}>
      {children}
    </h3>
  );
}

export function CardDescription({
  children,
  className,
}: PropsWithChildren<{ className?: string }>) {
  return (
    <p className={cn("mt-1 text-sm text-slate-400", className)}>
      {children}
    </p>
  );
}
