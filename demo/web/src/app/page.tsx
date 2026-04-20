"use client";
import * as React from "react";
import { Loader2, Play, RotateCcw, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Separator } from "@/components/ui/separator";
import { FlowCanvas, type LogItem } from "@/components/flow/FlowCanvas";
import { Palette } from "@/components/flow/Palette";
import { EventsPanel } from "@/components/flow/EventsPanel";
import { fetchNodeTypes } from "@/lib/api";
import type { NodeType } from "@/lib/types";

export default function Page() {
  const [nodeTypes, setNodeTypes] = React.useState<NodeType[] | null>(null);
  const [fetchError, setFetchError] = React.useState<string | null>(null);
  const [log, setLog] = React.useState<LogItem[]>([]);
  const [running, setRunning] = React.useState(false);

  const runSignalRef = React.useRef<() => Promise<void>>(async () => {});
  const clearSignalRef = React.useRef<() => void>(() => {});
  const loadExampleSignalRef = React.useRef<(e: "basic" | "shared-refs" | "for-each") => void>(() => {});
  const runningRef = React.useRef(false);

  React.useEffect(() => {
    let cancelled = false;
    fetchNodeTypes()
      .then((types) => {
        if (!cancelled) setNodeTypes(types);
      })
      .catch((err) => {
        if (!cancelled) setFetchError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleRun = async () => {
    setRunning(true);
    try {
      await runSignalRef.current();
    } finally {
      setRunning(false);
    }
  };

  if (fetchError) {
    return (
      <main className="flex h-screen items-center justify-center p-8">
        <div className="max-w-md rounded-lg border border-destructive/30 bg-destructive/10 p-6 text-center">
          <h1 className="text-lg font-semibold">Can&apos;t reach the backend</h1>
          <p className="mt-2 text-xs text-muted-foreground">{fetchError}</p>
          <p className="mt-4 text-xs">
            Start the FastAPI server from the repo root:
          </p>
          <pre className="mt-2 rounded bg-background/60 p-2 text-left text-[11px]">
            uv run uvicorn demo.app:app --port 8765 --reload
          </pre>
        </div>
      </main>
    );
  }

  if (!nodeTypes) {
    return (
      <main className="flex h-screen items-center justify-center text-muted-foreground">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading registry…
      </main>
    );
  }

  return (
    <main className="grid h-screen" style={{ gridTemplateColumns: "240px 1fr 320px", gridTemplateRows: "48px 1fr" }}>
      {/* Header */}
      <header className="col-span-3 flex items-center justify-between border-b bg-card px-4">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-semibold tracking-tight">Conductor Playground</h1>
          <span className="text-[10px] text-muted-foreground">
            {nodeTypes.length} registered nodes
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <Sparkles className="h-3.5 w-3.5" />
                Examples
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onSelect={() => loadExampleSignalRef.current("basic")}>
                Basic: text → uppercase → template
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => loadExampleSignalRef.current("shared-refs")}>
                Shared references: one source, two consumers
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => loadExampleSignalRef.current("for-each")}>
                For-each: iterate a list
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <Button variant="outline" size="sm" onClick={() => clearSignalRef.current()}>
            <RotateCcw className="h-3.5 w-3.5" />
            Clear
          </Button>
          <Button size="sm" onClick={handleRun} disabled={running}>
            {running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            Run
          </Button>
        </div>
      </header>

      {/* Palette */}
      <aside className="border-r bg-card">
        <Palette types={nodeTypes} />
      </aside>

      {/* Canvas */}
      <section className="relative bg-background">
        <FlowCanvas
          nodeTypes={nodeTypes}
          onEventLog={setLog}
          runSignalRef={runSignalRef}
          clearSignalRef={clearSignalRef}
          loadExampleSignalRef={loadExampleSignalRef}
          runningRef={runningRef}
        />
      </section>

      {/* Events panel */}
      <aside className="flex flex-col border-l bg-card">
        <div className="flex items-center justify-between border-b px-3 py-2">
          <span className="text-xs font-semibold">Events</span>
          <span className="text-[10px] text-muted-foreground">{log.length}</span>
        </div>
        <Separator />
        <div className="flex-1 overflow-hidden">
          <EventsPanel log={log} />
        </div>
      </aside>
    </main>
  );
}
