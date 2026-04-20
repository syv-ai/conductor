"use client";
import * as React from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { ExecutionEvent } from "@/lib/types";

interface LogItem {
  id: number;
  event: ExecutionEvent;
  time: number;
}

interface Props {
  log: LogItem[];
}

const tone: Record<ExecutionEvent["type"], string> = {
  node_start: "border-l-sky-500 bg-sky-500/5",
  node_complete: "border-l-emerald-500 bg-emerald-500/5",
  node_skipped: "border-l-zinc-500 bg-zinc-500/5",
  node_error: "border-l-rose-500 bg-rose-500/10",
  node_progress: "border-l-amber-500 bg-amber-500/5",
  node_retry: "border-l-amber-500 bg-amber-500/5",
  flow_complete: "border-l-emerald-500 bg-emerald-500/10 font-semibold",
  flow_error: "border-l-rose-500 bg-rose-500/10 font-semibold",
  flow_cancelled: "border-l-zinc-500 bg-zinc-500/10",
  flow_timeout: "border-l-amber-500 bg-amber-500/10",
  flow_paused: "border-l-violet-500 bg-violet-500/10",
};

function summarize(event: ExecutionEvent): { label: string; body?: string } {
  switch (event.type) {
    case "node_start":
      return { label: event.node_id };
    case "node_complete":
      return {
        label: event.node_id,
        body: JSON.stringify(event.result, null, 2),
      };
    case "node_skipped":
      return { label: event.node_id };
    case "node_error":
      return { label: event.node_id, body: event.error };
    case "node_progress":
      return { label: `${event.node_id}: ${event.current}/${event.total}` };
    case "node_retry":
      return {
        label: event.node_id,
        body: `attempt ${event.attempt}/${event.max_retries} — ${event.error}`,
      };
    case "flow_complete":
      return { label: "", body: JSON.stringify(event.results, null, 2) };
    case "flow_error":
      return { label: "", body: event.error };
    case "flow_cancelled":
      return { label: `completed: ${event.completed_nodes.join(", ")}` };
    case "flow_timeout":
      return {
        label: `after ${event.elapsed_seconds.toFixed(1)}s`,
        body: `timeout limit: ${event.timeout_seconds}s`,
      };
    case "flow_paused":
      return { label: event.node_id, body: event.prompt };
  }
}

export function EventsPanel({ log }: Props) {
  const scrollerRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const el = scrollerRef.current?.querySelector("[data-radix-scroll-area-viewport]") as HTMLElement | null;
    if (el) el.scrollTop = el.scrollHeight;
  }, [log.length]);

  if (log.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-center text-xs text-muted-foreground">
        Run a flow to see events.
      </div>
    );
  }

  return (
    <ScrollArea ref={scrollerRef} className="h-full">
      <div className="space-y-1 p-2">
        {log.map(({ id, event }) => {
          const { label, body } = summarize(event);
          return (
            <div
              key={id}
              className={cn("rounded-sm border-l-2 px-2 py-1 font-mono text-[11px]", tone[event.type])}
            >
              <div className="flex items-baseline gap-2">
                <span className="font-semibold">{event.type}</span>
                {label && <span className="text-muted-foreground">{label}</span>}
              </div>
              {body && (
                <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap rounded bg-background/60 p-1 text-[10px]">
                  {body}
                </pre>
              )}
            </div>
          );
        })}
      </div>
    </ScrollArea>
  );
}
