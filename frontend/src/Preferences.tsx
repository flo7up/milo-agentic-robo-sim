import { createContext, useContext, useEffect, useRef, useState, type ReactNode, type SetStateAction } from 'react';
import { RotateCcw } from 'lucide-react';
import type { ChallengeEnvironment, ChallengeId, Reasoning } from './types';

export type SceneSelection = {challenge_id: ChallengeId; environment: ChallengeEnvironment; reuse_saved_map: boolean;
  orbit_target?: 'table' | 'sofa' | 'chair' | 'floor lamp'; orbit_direction?: 'clockwise' | 'counterclockwise'};
type Preferences = {
  interval: number; turns: number; reasoning: Reasoning; luna_endpoint: string; luna_deployment: string;
  mission_controller: 'luna' | 'qwen' | 'hybrid' | 'policy'; local_model_endpoint: string; local_model_tag: string;
  control_mode: 'task' | 'exploration'; exploration_budget: number; handoff: boolean; skill_composer: boolean;
  navigation_backend: 'builtin' | 'nav2'; ai_routes: boolean; adaptive: boolean;
  navigation_mode: 'luna_continuous' | 'luna_navigation'; inspector: 'conversation' | 'trace' | 'settings';
  connection_open: boolean; run_settings_open: boolean; challenge_details_open: boolean;
  challenge_selection: SceneSelection; goals: Record<string, string>; compact_arms: boolean;
  graphics: 'standard' | 'enhanced'; axes: boolean; manual_open: boolean; manual_tab: 'drive' | 'head' | 'arms';
  manual_arm: 'left' | 'right'; duration: number; head_yaw: number; head_pitch: number; opening: number; force: number;
  position: number[]; joints: number[];
  mission_map_context: boolean;
  max_model_requests: number; max_model_tokens: number;
  kitchen_turns: number; kitchen_max_model_requests: number; kitchen_max_model_tokens: number;
  maze_turns: number; maze_max_model_requests: number; maze_max_model_tokens: number; maze_mission_budget: number;
};
type Values = Partial<Preferences>;
function merge(previous: Values, patch: Values): Values {
  return {...previous, ...patch, ...(patch.goals ? {goals: {...previous.goals, ...patch.goals}} : {})};
}
const cacheKey = 'milo-workspace-preferences-v1';
const pendingKey = 'milo-workspace-preferences-pending-v1';
const choices: Record<string, readonly string[]> = {
  reasoning: ['none', 'low', 'medium', 'high'], control_mode: ['task', 'exploration'],
  mission_controller: ['luna', 'qwen', 'hybrid', 'policy'],
  navigation_backend: ['builtin', 'nav2'], navigation_mode: ['luna_continuous', 'luna_navigation'],
  inspector: ['conversation', 'trace', 'settings'], graphics: ['standard', 'enhanced'],
  manual_tab: ['drive', 'head', 'arms'], manual_arm: ['left', 'right'],
};
const ranges: Record<string, [number, number]> = {interval: [.25, 30], turns: [1, 80], exploration_budget: [1, 300],
  max_model_requests: [1, 200], max_model_tokens: [1, 2000000],
  kitchen_turns: [1, 80], kitchen_max_model_requests: [1, 200], kitchen_max_model_tokens: [1, 2000000],
  maze_turns: [1, 200], maze_max_model_requests: [1, 200], maze_max_model_tokens: [1, 2000000], maze_mission_budget: [5, 600],
  duration: [.1, 2], head_yaw: [-1.5, 1.5], head_pitch: [-.7, 1.15], opening: [0, .11], force: [1, 35]};
const booleans = new Set(['handoff', 'skill_composer', 'ai_routes', 'adaptive', 'connection_open', 'run_settings_open',
  'challenge_details_open', 'compact_arms', 'axes', 'manual_open', 'mission_map_context']);
const scenarios = new Set(['bench', 'park', 'park_left', 'park_right', 'park_far', 'tidy', 'sort', 'recharge', 'apartment', 'kitchen_bathroom', 'clinic_delivery',
  'warehouse', 'inspection', 'workshop', 'local_park', 'pedestrian_crossing', 'flat_kitchen', 'furniture_circuit', 'chair_circuit_far', 'movement_practice', 'maze', 'maze_complex']);

function valid(key: string, value: unknown): boolean {
  if (key === 'position' || key === 'joints') {
    const limits = key === 'position' ? [1, 1, 1] : [1.8, 2.5, 2.7, 3, 2.5, 3];
    return Array.isArray(value) && value.length === limits.length
      && value.every((number, index) => typeof number === 'number' && Number.isFinite(number) && Math.abs(number) <= limits[index]);
  }
  if (choices[key]) return typeof value === 'string' && choices[key].includes(value);
  if (ranges[key]) return typeof value === 'number' && Number.isFinite(value) && value >= ranges[key][0] && value <= ranges[key][1]
    && (!['turns', 'exploration_budget', 'max_model_requests', 'max_model_tokens',
      'kitchen_turns', 'kitchen_max_model_requests', 'kitchen_max_model_tokens',
      'maze_turns', 'maze_max_model_requests', 'maze_max_model_tokens', 'maze_mission_budget'].includes(key) || Number.isInteger(value));
  if (booleans.has(key)) return typeof value === 'boolean';
  if (key === 'luna_deployment') return typeof value === 'string' && value.length <= 128 && /^[\w.-]*$/.test(value);
  if (key === 'local_model_tag') return typeof value === 'string' && value.length <= 120;
  if (key === 'local_model_endpoint') {
    if (value === '') return true;
    if (typeof value !== 'string' || value.length > 2048) return false;
    try { const parsed = new URL(value); return parsed.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(parsed.hostname)
      && !parsed.username && !parsed.password && !parsed.search && !parsed.hash && (parsed.pathname === '/' || parsed.pathname === '') && parsed.port !== '0'; }
    catch { return false; }
  }
  if (key === 'luna_endpoint') {
    if (value === '') return true;
    if (typeof value !== 'string' || value.length > 2048) return false;
    try { const parsed = new URL(value); return ['http:', 'https:'].includes(parsed.protocol) && !!parsed.hostname
      && !parsed.username && !parsed.password && !parsed.search && !parsed.hash; } catch { return false; }
  }
  if (key === 'goals') return !!value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length <= 80
    && Object.entries(value).every(([name, goal]) => name.length > 0 && name.length <= 160 && typeof goal === 'string' && goal.length <= 2000);
  if (key === 'challenge_selection' && value && typeof value === 'object') {
    const scene = value as SceneSelection;
    return scenarios.has(scene.challenge_id) && ['standalone', 'shared_apartment_v1'].includes(scene.environment)
      && typeof scene.reuse_saved_map === 'boolean'
      && (!scene.orbit_target || ['table', 'sofa', 'chair', 'floor lamp'].includes(scene.orbit_target))
      && (!scene.orbit_direction || ['clockwise', 'counterclockwise'].includes(scene.orbit_direction))
      && (scene.challenge_id === 'furniture_circuit' || (!scene.orbit_target && !scene.orbit_direction))
      && (scene.environment !== 'shared_apartment_v1' || (['furniture_circuit', 'apartment', 'flat_kitchen', 'recharge'].includes(scene.challenge_id)
        && (!scene.orbit_target || scene.orbit_target === 'table')));
  }
  return false;
}

function validated(value: unknown): Values {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  return Object.fromEntries(Object.entries(value).filter(([key, entry]) => valid(key, entry)));
}

function cached(): Values {
  try {
    const values = validated(JSON.parse(localStorage.getItem(cacheKey) ?? '{}'));
    for (const [key, legacy] of Object.entries({navigation_backend: 'milo-navigation-backend', ai_routes: 'milo-ai-generated-routes',
      adaptive: 'milo-adaptive-navigation', navigation_mode: 'milo-navigation-mode', compact_arms: 'milo-compact-arms', graphics: 'milo-spectator-graphics'})) {
      const stored = localStorage.getItem(legacy);
      const value = booleans.has(key) ? stored === 'true' ? true : stored === 'false' ? false : null : stored;
      if (!(key in values) && valid(key, value)) Object.assign(values, {[key]: value});
    }
    return values;
  } catch { return {}; }
}

const Context = createContext<{values: Values; update: (patch: Values) => void; save: () => Promise<void>}>(
  {values: {}, update: () => {}, save: async () => {}});

export function PreferencesProvider({children}: {children: ReactNode}) {
  const [values, setValues] = useState<Values>({});
  const current = useRef<Values>({});
  const pending = useRef<Values>({});
  const inFlight = useRef<Values>({});
  const writing = useRef<Promise<void> | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');
  function saveJournal() {
    try { localStorage.setItem(pendingKey, JSON.stringify(merge(inFlight.current, pending.current))); } catch {}
  }
  function update(patch: Values) {
    const safe = validated(patch);
    current.current = merge(current.current, patch);
    setValues(current.current);
    if (!Object.keys(safe).length) return;
    pending.current = merge(pending.current, safe);
    try { localStorage.setItem(cacheKey, JSON.stringify(validated(current.current))); } catch {}
    saveJournal();
    void flush().catch(() => {});
  }
  async function flush(): Promise<void> {
    if (writing.current) {
      await writing.current;
      return flush();
    }
    if (!Object.keys(pending.current).length) return;
    writing.current = (async () => {
      const patch = pending.current;
      inFlight.current = patch;
      pending.current = {};
      try {
        const response = await fetch('/api/preferences', {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(patch), keepalive: true, signal: AbortSignal.timeout(8000)});
        if (!response.ok) throw new Error('Preferences could not be saved.');
        setError('');
      } catch {
        pending.current = merge(patch, pending.current);
        setError('Preferences are not saved to this workspace.');
        throw new Error('Configuration could not be saved. Please retry.');
      } finally { writing.current = null; inFlight.current = {}; saveJournal(); }
    })();
    await writing.current;
    return flush();
  }
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 5000);
    void (async () => {
      const local = cached();
      let unsaved: Values = {};
      try { unsaved = validated(JSON.parse(localStorage.getItem(pendingKey) ?? '{}')); } catch {}
      let initial = merge(local, unsaved);
      pending.current = unsaved;
      try {
        const response = await fetch('/api/preferences', {signal: controller.signal});
        if (!response.ok) throw new Error();
        const document = await response.json();
        if (document.version !== 1) throw new Error();
        const saved = validated(document.preferences);
        initial = merge({...local, ...saved}, unsaved);
        pending.current = merge(Object.fromEntries(Object.entries(local).filter(([key]) => !(key in saved))), unsaved);
      } catch { if (active) setError('Workspace preferences are unavailable. Browser preferences remain available.'); }
      finally {
        window.clearTimeout(timeout);
        if (active) { current.current = initial; setValues(initial); setReady(true); }
      }
    })();
    return () => { active = false; controller.abort(); window.clearTimeout(timeout); };
  }, []);
  useEffect(() => {
    if (!ready) return;
    void flush().catch(() => {});
    const savePending = () => {
      if (!Object.keys(pending.current).length) return;
      void fetch('/api/preferences', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(pending.current), keepalive: true}).catch(() => {});
    };
    window.addEventListener('pagehide', savePending);
    return () => window.removeEventListener('pagehide', savePending);
  }, [ready]);
  if (!ready) return <div className="loading" role="status">Loading saved settings...</div>;
  return <Context.Provider value={{values, update, save: flush}}>
    {error && <div className="error" role="alert">{error} <button type="button" title="Retry saving preferences" aria-label="Retry saving preferences" onClick={() => void flush().catch(() => {})}><RotateCcw size={16}/></button></div>}
    {children}
  </Context.Provider>;
}

export function useSavePreferences() {
  return useContext(Context).save;
}

export function usePreference<Key extends keyof Preferences>(key: Key, fallback: Preferences[Key]) {
  const {values, update} = useContext(Context);
  const value = values[key] ?? fallback;
  function setValue(next: SetStateAction<Preferences[Key]>) {
    const resolved = typeof next === 'function' ? (next as (previous: Preferences[Key]) => Preferences[Key])(value) : next;
    if (JSON.stringify(value) === JSON.stringify(resolved)) return;
    if (key === 'goals') {
      const previous = value as Preferences['goals'];
      update({goals: Object.fromEntries(Object.entries(resolved as Preferences['goals']).filter(([name, goal]) => previous[name] !== goal))});
    } else update({[key]: resolved});
  }
  return [value, setValue] as const;
}
