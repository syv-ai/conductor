"use client";
import * as React from "react";
import { GripVertical } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import type { NodeType } from "@/lib/types";

interface Props {
  types: NodeType[];
}

export function Palette({ types }: Props) {
  const io = types.filter((t) => t.category !== "control");
  const control = types.filter((t) => t.category === "control");

  const renderItem = (t: NodeType) => (
    <div
      key={t.id}
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData("application/conductor-node-type", t.id);
        e.dataTransfer.effectAllowed = "move";
      }}
      className="group mb-1.5 cursor-grab rounded-md border bg-card px-2.5 py-2 text-xs transition-colors hover:border-primary/60 active:cursor-grabbing"
    >
      <div className="flex items-center gap-1.5">
        <GripVertical className="h-3 w-3 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
        <span className="truncate font-medium">{t.name}</span>
      </div>
      <div className="mt-0.5 line-clamp-2 pl-4 text-[10px] text-muted-foreground">
        {t.description}
      </div>
    </div>
  );

  return (
    <ScrollArea className="h-full">
      <div className="p-3">
        <div className="mb-1.5 flex items-center justify-between">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            I/O
          </div>
          <Badge variant="outline" className="text-[9px]">
            {io.length}
          </Badge>
        </div>
        {io.map(renderItem)}

        <div className="mb-1.5 mt-4 flex items-center justify-between">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            Control
          </div>
          <Badge variant="outline" className="text-[9px]">
            {control.length}
          </Badge>
        </div>
        {control.map(renderItem)}
      </div>
    </ScrollArea>
  );
}
