import { useState, type ReactNode } from 'react';
import { definition, type Kind } from '../../engine';
import { Modal } from '../shared/Modal';
import { DefinitionCard } from './DefinitionCard';
import { UnitReferenceContext } from './UnitReference';

/** 每个游戏实例独立持有介绍弹窗；在规则/图鉴上方打开，关闭后回到原入口。 */
export function UnitReferenceProvider({
  children,
  onOpenChange,
}: {
  children: ReactNode;
  onOpenChange?: (open: boolean) => void;
}) {
  const [history, setHistory] = useState<Kind[]>([]);
  const kind = history.at(-1);
  return (
    <UnitReferenceContext.Provider
      value={(next) => {
        onOpenChange?.(true);
        setHistory((old) => (old.at(-1) === next ? old : [...old, next]));
      }}
    >
      {children}
      {kind !== undefined && (
        <Modal
          title={definition(kind).name}
          onClose={() => {
            setHistory([]);
            onOpenChange?.(false);
          }}
        >
          {history.length > 1 && (
            <button className="text-button" onClick={() => setHistory((old) => old.slice(0, -1))}>
              返回上一介绍
            </button>
          )}
          <DefinitionCard d={definition(kind)} />
        </Modal>
      )}
    </UnitReferenceContext.Provider>
  );
}
