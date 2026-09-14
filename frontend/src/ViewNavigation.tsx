import { Archive, Gauge } from 'lucide-react';

export type TestView = 'cockpit' | 'archive';

export function ViewNavigation({ current, onNavigate }: { current: TestView; onNavigate?: (view: TestView) => void }) {
  return <nav className="view-navigation" aria-label="Test views">
    <a href="/" aria-current={current === 'cockpit' ? 'page' : undefined}
      onClick={event => {
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey && (current === 'cockpit' || onNavigate)) {
          event.preventDefault(); onNavigate?.('cockpit');
        }
      }}>
      <Gauge size={18} aria-hidden="true" />Test cockpit
    </a>
    <a href="/?view=test-results" aria-current={current === 'archive' ? 'page' : undefined}
      onClick={event => {
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey && (current === 'archive' || onNavigate)) {
          event.preventDefault(); onNavigate?.('archive');
        }
      }}>
      <Archive size={18} aria-hidden="true" />Test archive
    </a>
  </nav>;
}