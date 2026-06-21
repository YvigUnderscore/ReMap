import * as React from "react";
import { cn } from "../../lib/utils";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger" | "success" | "warning" | "info";
};

export function Button({
  className,
  variant = "primary",
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex min-h-10 items-center justify-center rounded-xl px-4 py-2 text-sm font-medium transition duration-200 hover:-translate-y-0.5 active:translate-y-0 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0",
        variant === "primary" &&
          "bg-accent-cyan/90 text-graphite-950 shadow-glow hover:bg-accent-cyan",
        variant === "secondary" &&
          "border border-white/10 bg-white/5 text-white hover:bg-white/10",
        variant === "ghost" && "text-slate-300 hover:bg-white/5 hover:text-white",
        variant === "danger" &&
          "bg-accent-red/90 text-white hover:bg-accent-red",
        variant === "success" &&
          "bg-accent-emerald/90 text-graphite-950 shadow-glow hover:bg-accent-emerald",
        variant === "warning" &&
          "bg-accent-amber/90 text-graphite-950 shadow-glow hover:bg-accent-amber",
        variant === "info" &&
          "bg-accent-blue/90 text-white shadow-glow hover:bg-accent-blue",
        className,
      )}
      {...props}
    />
  );
}
