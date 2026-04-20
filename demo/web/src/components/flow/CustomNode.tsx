"use client";
import * as React from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Info, Link2, Link2Off, Share2, Trash2, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import type { CustomNodeData } from "@/lib/types";
import { WidgetField } from "./WidgetField";
import { NodeInfoDialog } from "./NodeInfoDialog";

const stateBorder: Record<string, string> = {
  idle: "border-border",
  running: "border-amber-500 shadow-[0_0_0_2px_rgba(245,158,11,0.25)]",
  completed: "border-emerald-500",
  error: "border-rose-500 shadow-[0_0_0_2px_rgba(244,63,94,0.25)]",
  skipped: "opacity-50 border-border",
};

export function CustomNode({ id, data }: NodeProps) {
  const d = data as unknown as CustomNodeData;
  const [infoOpen, setInfoOpen] = React.useState(false);
  const [publishLabelFor, setPublishLabelFor] = React.useState<string | null>(null);
  const [pendingLabel, setPendingLabel] = React.useState("");

  const { nodeType, values, produces, consumes, executionState, progress, availableRefs, connectedInputs } = d;
  const isControl = nodeType.category === "control";

  const handlePublishOpen = (outputName: string) => {
    if (produces[outputName]) {
      d.onTogglePublish(id, outputName, null); // already published → unpublish
      return;
    }
    const existingLabels = new Set(availableRefs.map((r) => r.label));
    let candidate = nodeType.outputs.find((o) => o.name === outputName)?.label ?? outputName;
    let n = 1;
    while (existingLabels.has(candidate)) {
      n += 1;
      candidate = `${candidate} ${n}`;
    }
    setPendingLabel(candidate);
    setPublishLabelFor(outputName);
  };

  const commitPublish = () => {
    if (publishLabelFor) {
      d.onTogglePublish(id, publishLabelFor, pendingLabel.trim() || publishLabelFor);
    }
    setPublishLabelFor(null);
  };

  return (
    <>
      <div
        className={cn(
          "w-[260px] rounded-lg border bg-card text-card-foreground shadow-sm transition-colors",
          stateBorder[executionState] ?? stateBorder.idle,
        )}
      >
        {/* Header */}
        <div
          className={cn(
            "flex items-start justify-between gap-2 rounded-t-lg border-b px-3 py-2 drag-handle",
            isControl && "bg-purple-500/10",
          )}
        >
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5">
              <div className="truncate text-xs font-semibold">{nodeType.name}</div>
              {progress && (
                <Badge variant="warning" className="px-1.5 py-0 text-[10px]">
                  {progress.current}/{progress.total}
                </Badge>
              )}
              {executionState === "completed" && !progress && (
                <Badge variant="success" className="px-1.5 py-0 text-[10px]">
                  done
                </Badge>
              )}
              {executionState === "error" && (
                <Badge variant="destructive" className="px-1.5 py-0 text-[10px]">
                  error
                </Badge>
              )}
            </div>
            <div className="mt-0.5 truncate text-[10px] text-muted-foreground">
              {id} · {nodeType.id}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-0.5">
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => setInfoOpen(true)}
              aria-label="Node info"
            >
              <Info className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-muted-foreground hover:text-destructive"
              onClick={() => d.onDelete(id)}
              aria-label="Delete node"
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        {/* Inputs */}
        {nodeType.inputs.length > 0 && (
          <div className="space-y-2 px-3 pt-2">
            {nodeType.inputs.map((inp) => {
              const consumed = consumes[inp.name];
              const producerLabel = consumed
                ? availableRefs.find(
                    (r) => r.nodeId === consumed[0] && r.handle === consumed[1],
                  )?.label ?? `${consumed[0]}.${consumed[1]}`
                : null;
              const connected = connectedInputs.has(inp.name);
              const filled = connected || !!consumed;

              return (
                <div key={inp.name} className="relative">
                  {!inp.disable_handle && (
                    <Handle
                      id={inp.name}
                      type="target"
                      position={Position.Left}
                      className="!-left-[5px]"
                      style={{ top: 14 }}
                    />
                  )}

                  <div className="flex items-center justify-between gap-1">
                    <label className="text-[11px] font-medium text-muted-foreground">
                      {inp.label}
                      {connected && (
                        <Badge variant="outline" className="ml-1 px-1 py-0 text-[9px]">
                          wired
                        </Badge>
                      )}
                      {consumed && (
                        <Badge variant="secondary" className="ml-1 px-1 py-0 text-[9px]">
                          ← {producerLabel}
                        </Badge>
                      )}
                    </label>

                    {!inp.disable_handle && (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            className={cn("h-5 w-5", consumed && "text-primary")}
                            aria-label="Consume shared reference"
                          >
                            {consumed ? <Link2Off className="h-3 w-3" /> : <Link2 className="h-3 w-3" />}
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuLabel>Consume shared reference</DropdownMenuLabel>
                          {consumed && (
                            <>
                              <DropdownMenuItem onSelect={() => d.onSetConsume(id, inp.name, null)}>
                                Disconnect
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                            </>
                          )}
                          {availableRefs.length === 0 ? (
                            <DropdownMenuItem disabled>No published refs yet</DropdownMenuItem>
                          ) : (
                            availableRefs
                              .filter((r) => r.nodeId !== id) // don't consume from self
                              .map((r) => (
                                <DropdownMenuItem
                                  key={`${r.nodeId}:${r.handle}`}
                                  onSelect={() =>
                                    d.onSetConsume(id, inp.name, [r.nodeId, r.handle])
                                  }
                                >
                                  <span className="truncate">{r.label}</span>
                                  <span className="ml-auto text-muted-foreground">
                                    {r.nodeId}
                                  </span>
                                </DropdownMenuItem>
                              ))
                          )}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )}
                  </div>

                  <div className="mt-1">
                    <WidgetField
                      param={inp}
                      value={values[inp.name]}
                      disabled={filled}
                      onChange={(v) => d.onValueChange(id, inp.name, v)}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Outputs */}
        {nodeType.outputs.length > 0 && (
          <div className="mt-2 space-y-1 border-t px-3 py-2">
            {nodeType.outputs.map((out) => {
              const published = produces[out.name];
              return (
                <div key={out.name} className="relative flex items-center justify-end gap-1.5">
                  <Button
                    variant={published ? "default" : "ghost"}
                    size="icon"
                    className="h-5 w-5"
                    onClick={() => handlePublishOpen(out.name)}
                    aria-label={published ? "Unpublish shared ref" : "Publish as shared ref"}
                  >
                    <Share2 className="h-3 w-3" />
                  </Button>
                  <div className="text-[11px] text-muted-foreground">
                    {out.label}
                    {published && (
                      <Badge variant="default" className="ml-1 px-1 py-0 text-[9px]">
                        {published}
                      </Badge>
                    )}
                  </div>
                  <Handle
                    id={out.name}
                    type="source"
                    position={Position.Right}
                    className="!-right-[5px]"
                  />
                </div>
              );
            })}
          </div>
        )}

        {/* Publish-label prompt */}
        {publishLabelFor && (
          <div className="border-t px-3 py-2">
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Publish label
            </div>
            <div className="flex items-center gap-1">
              <Input
                value={pendingLabel}
                onChange={(e) => setPendingLabel(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitPublish();
                  if (e.key === "Escape") setPublishLabelFor(null);
                }}
                autoFocus
                className="h-7 flex-1"
              />
              <Button size="xs" onClick={commitPublish}>
                Publish
              </Button>
              <Button
                size="icon"
                variant="ghost"
                className="h-6 w-6"
                onClick={() => setPublishLabelFor(null)}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          </div>
        )}
      </div>

      <NodeInfoDialog nodeType={nodeType} open={infoOpen} onOpenChange={setInfoOpen} />
    </>
  );
}
