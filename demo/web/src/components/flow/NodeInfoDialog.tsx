"use client";
import * as React from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import type { NodeType } from "@/lib/types";

interface Props {
  nodeType: NodeType;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function NodeInfoDialog({ nodeType, open, onOpenChange }: Props) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <div className="flex items-center gap-2">
            <DialogTitle>{nodeType.name}</DialogTitle>
            <Badge variant="outline">{nodeType.id}</Badge>
            {nodeType.category === "control" && <Badge variant="secondary">control</Badge>}
          </div>
          <DialogDescription>{nodeType.description}</DialogDescription>
        </DialogHeader>

        <div className="space-y-3 text-xs">
          <section>
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Inputs
            </div>
            {nodeType.inputs.length === 0 ? (
              <div className="text-muted-foreground">None</div>
            ) : (
              <ul className="space-y-1">
                {nodeType.inputs.map((inp) => (
                  <li key={inp.name} className="flex items-start gap-2">
                    <code className="rounded bg-muted px-1 py-0.5">{inp.name}</code>
                    <span className="text-muted-foreground">{inp.type}</span>
                    <span>{inp.label}</span>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <Separator />

          <section>
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Outputs
            </div>
            {nodeType.outputs.length === 0 ? (
              <div className="text-muted-foreground">None</div>
            ) : (
              <ul className="space-y-1">
                {nodeType.outputs.map((out) => (
                  <li key={out.name} className="flex items-start gap-2">
                    <code className="rounded bg-muted px-1 py-0.5">{out.name}</code>
                    <span className="text-muted-foreground">{out.type}</span>
                    <span>{out.label}</span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}
