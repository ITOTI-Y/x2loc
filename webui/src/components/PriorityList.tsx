import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core"
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable"
import { CSS } from "@dnd-kit/utilities"
import { DotsSixVerticalIcon, LockSimpleIcon } from "@phosphor-icons/react"
import { useTranslation } from "react-i18next"

import type { SourceKind } from "@/lib/api"
import { cn } from "@/lib/utils"

function Row({ kind, rank }: { kind: SourceKind; rank: number }) {
  const { t } = useTranslation()
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: kind,
  })
  return (
    <li
      ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition }}
      className={cn(
        "flex items-center gap-3 border-b bg-background px-3 py-2.5 last:border-b-0",
        isDragging && "relative z-10 shadow-md ring-1 ring-ring",
      )}
    >
      <button
        type="button"
        className="cursor-grab touch-none rounded p-0.5 text-muted-foreground hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring active:cursor-grabbing"
        aria-label={t("settings.priority.drag", { name: t(`sources.${kind}`) })}
        {...attributes}
        {...listeners}
      >
        <DotsSixVerticalIcon className="size-4" weight="bold" />
      </button>
      <span className="w-5 font-mono text-xs text-muted-foreground tabular-nums">{rank}</span>
      <div className="min-w-0">
        <p className="text-sm font-medium">{t(`sources.${kind}`)}</p>
        <p className="text-xs text-muted-foreground">{t(`sources.${kind}_hint`)}</p>
      </div>
    </li>
  )
}

export function PriorityList({
  value,
  onChange,
}: {
  value: SourceKind[]
  onChange: (next: SourceKind[]) => void
}) {
  const { t } = useTranslation()
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  const onDragEnd = ({ active, over }: DragEndEvent) => {
    if (over === null || active.id === over.id) return
    onChange(
      arrayMove(value, value.indexOf(active.id as SourceKind), value.indexOf(over.id as SourceKind)),
    )
  }

  return (
    <ol className="overflow-hidden rounded-lg border">
      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
        <SortableContext items={value} strategy={verticalListSortingStrategy}>
          {value.map((kind, index) => (
            <Row key={kind} kind={kind} rank={index + 1} />
          ))}
        </SortableContext>
      </DndContext>
      <li className="flex items-center gap-3 bg-muted/50 px-3 py-2.5">
        <LockSimpleIcon className="ml-0.5 size-4 text-muted-foreground" />
        <span className="w-5 font-mono text-xs text-muted-foreground tabular-nums">
          {value.length + 1}
        </span>
        <div>
          <p className="text-sm font-medium">{t("sources.source")}</p>
          <p className="text-xs text-muted-foreground">{t("sources.source_hint")}</p>
        </div>
      </li>
    </ol>
  )
}
