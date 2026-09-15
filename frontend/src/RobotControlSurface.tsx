import { createContext, useContext, useLayoutEffect, useState, type Dispatch, type SetStateAction } from 'react';

export const RobotControlSurface = createContext<Dispatch<SetStateAction<HTMLElement | null>> | null>(null);

export function RobotControlSlot({ active }: { active: boolean }) {
  const setSurface = useContext(RobotControlSurface);
  const [host, setHost] = useState<HTMLDivElement | null>(null);
  useLayoutEffect(() => {
    if (!active || !host || !setSurface) return;
    setSurface(host);
    return () => setSurface(current => current === host ? null : current);
  }, [active, host, setSurface]);
  return <div ref={setHost} className="dialog-controls" />;
}