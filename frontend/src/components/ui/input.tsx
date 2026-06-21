import * as React from "react";
import { cn } from "../../lib/utils";

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "h-11 w-full rounded-2xl border border-white/10 bg-graphite-900/80 px-4 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-accent-cyan/60 focus:ring-2 focus:ring-accent-cyan/20",
        props.className,
      )}
    />
  );
}

