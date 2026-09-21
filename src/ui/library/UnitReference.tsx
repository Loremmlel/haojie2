import { createContext, useContext, type ReactNode } from 'react';
import { CATALOG, type Kind } from '../../engine';

export const UnitReferenceContext = createContext<(kind: Kind) => void>(() => {});

/** 名称只取图鉴；长名称优先，避免把“超级跑得快”拆成“跑得快”。 */
const names = new Map(CATALOG.map((d) => [d.name, d.id]));
const pattern = new RegExp(
  `(${[...names.keys()]
    .sort((a, b) => b.length - a.length)
    .map((name) => name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('|')})`,
  'g',
);

/** 打开只读图鉴，不修改局面、地址或宿主导航，也不触发外层的选牌操作。 */
export function UnitLink({ kind, children }: { kind: Kind; children: ReactNode }) {
  const open = useContext(UnitReferenceContext);
  return (
    <a
      className="unit-link"
      href={`#unit-${kind}`}
      aria-haspopup="dialog"
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        open(kind);
      }}
    >
      {children}
    </a>
  );
}

/** 将说明、状态和旧战报中的正式名称转成图鉴链接；不解析 HTML 或执行文案。 */
export function UnitText({ children }: { children: string }) {
  return (
    <>
      {children.split(pattern).map((part, index) => {
        const kind = names.get(part);
        return kind === undefined ? (
          part
        ) : (
          <UnitLink key={index} kind={kind}>
            {part}
          </UnitLink>
        );
      })}
    </>
  );
}
