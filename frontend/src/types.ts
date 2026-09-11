export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];
export type Pose = { key: string; position: Vec3; quaternion: Quat };
export type Geometry = Pose & { type: number; dimensions: number[]; color: number[]; texture: string | null };
export type ProximitySensors = { simulated_time_s: number; max_range_m: number;
  distances: { direction: string; bearing_rad: number; distance_m: number | null; status: 'hit' | 'clear' | 'occluded' }[];
  collisions: { direction: string; force_n: number }[] };
export type CameraFrame = { seq: number; frame_ref: string; simulated_time_s: number; url: string };
export type ExecutionMode = 'single_step' | 'navigation_plan' | 'supervised_policy';
export type SkillState = {
  revision: number; motion_revision: number; status: 'idle' | 'running' | 'awaiting_policy' | 'completed' | 'cancelled' | 'failed';
  skill: string | null; instruction: string; reason: string; remaining_s: number; checkpoint: string;
  policy_requests: number; rejected_chunks: number; policy_latency_s: number | null; completion_source: 'supervisor' | null;
};
export type NavigationState = {
  revision: number; status: 'idle' | 'running' | 'awaiting_feedback' | 'completed' | 'failed' | 'cancelled'; reason: string;
  active_step: number | null; remaining_s: number; expires_in_s: number; velocity_mps_radps: number[];
  travel_m: number; scan_span_rad: number;
  steps: { skill: 'inspect_room' | 'locate_doorway' | 'approach' | 'cross'; goal: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'; evidence: string }[];
};
export type ManualPlacement = { run_id: string; episode_epoch: number; observation_seq: number; xy_m: [number, number] };
export type ChallengeId = 'bench' | 'park' | 'tidy' | 'sort' | 'recharge' | 'apartment' | 'kitchen_bathroom' | 'clinic_delivery' | 'warehouse' | 'inspection' | 'workshop';
export type ChallengePreset = { id: ChallengeId; title: string; skill: string; goal: string; objectives: string[]; suggested_turn_limit: number;
  category?: 'Navigation' | 'Perception' | 'Manipulation'; difficulty?: 'Foundation' | 'Advanced' };
export type ChallengeState = ChallengePreset & {
  status: 'in_progress' | 'completed' | 'failed'; completed_objectives: number;
  progress: { label: string; complete: boolean; detail: string }[];
};
export type Observation = {
  run_id: string; episode_epoch: number; seq: number; frame_ref: string;
  simulated_time_s: number; wall_timestamp: number; head_rad: number[]; odometry_m_rad: number[];
  joints: { name: string; position: number; velocity: number }[];
  grippers: Record<string, { aperture_m: number; contact: boolean[]; load_n: number }>;
  bumpers: string[];
  battery: { charge_pct: number; low: boolean; charging: boolean } | null;
  proximity: ProximitySensors | null;
  navigation?: NavigationState | null;
  skill?: SkillState | null;
};
export type Result = { action_id: string; status: string; error: string | null; message: string; actual_duration_s: number; observation: Observation };
export type Reasoning = 'none' | 'low' | 'medium' | 'high';
export type ModelProfile = { id: string; label: string; deployment: string; provider: 'foundry' | 'ollama'; reasoning_efforts: Reasoning[]; configured: boolean };
export type AgentAction = { turn: number; tool: string; arguments: unknown; status: string; error: string | null; message: string; actual_duration_s: number; observation_seq: number; odometry_delta_m_rad?: number[] | null };
export type CollisionFeedback = { source: 'current' | 'during_action'; contacts: { direction: string; force_n?: number }[]; guidance: string };
export type ExchangeEntry = {
  id: number; title: string; timestamp: number; turn: number; image_url: string | null; image_urls?: string[];
} & (
  { kind: 'session'; payload: { model?: string; deployment?: string; reasoning?: string; goal?: string; instructions?: string; tools?: unknown[]; feedback_interval_s?: number; max_turns?: number; status?: string; message?: string; reason?: string } } |
  { kind: 'feedback'; payload: { observation: Observation; image_detail: string; history_turns: number[]; input_items: number; images_in_request: number; tool_result_call_ids: string[]; context_mode?: string; memory_frame_seq?: number | null; recent_actions?: Partial<AgentAction>[]; collision_feedback?: CollisionFeedback | null; images_per_request?: number; context_tokens?: number; retained_context_tokens_estimate?: number; context_estimator?: string; camera_frames?: { seq: number; frame_ref: string; simulated_time_s: number; wall_timestamp: number; current: boolean }[] } } |
  { kind: 'response'; payload: { text: string; text_truncated: boolean; status: string; latency_s: number; input_tokens: number | null; output_tokens: number | null; calls: { call_id: string; name: string; arguments: string }[]; calls_truncated: boolean; refusals: string[] } } |
  { kind: 'tool'; payload: { tool: string; arguments: unknown; call_id: string } } |
  { kind: 'policy'; payload: Record<string, unknown> } |
  { kind: 'result'; payload: { tool: string; call_id: string; result: Partial<Result> & { status: string; sensor_deltas?: unknown }; model_result?: Record<string, unknown> & { collision_feedback?: CollisionFeedback } } }
);
export type ExchangeFeed = { session_id: string | null; revision: number; first_id: number; capacity: number; events: ExchangeEntry[] };
export type AgentState = {
  execution_mode: ExecutionMode;
  context_usage?: {
    turn: number; observation_seq: number; retained_tokens_estimate: number; retained_budget: number;
    retained_turns: number; images_sent: number; image_limit: number; input_tokens: number | null;
    response_received: boolean; active_command: string; instructions_repeated: boolean;
  } | null;
  policy?: { endpoint: string };
  images_per_request: number; context_tokens: number;
  outcome: { kind: string; message: string; source: 'physics' | 'agent' | 'controller'; timestamp: number } | null;
  auto_wake: boolean; idle_reason: string | null; idle_since: number | null; camera_unchanged_s: number; wake_reason: string | null;
  mode: 'llm' | 'chat' | 'voice';
  chat_messages: { id: string; role: 'user' | 'assistant'; text: string }[];
  active: boolean; phase: string; session_id: string | null; model_id: string; reasoning: Reasoning;
  goal: string; feedback_interval_s: number; turns: number; max_turns: number;
  last_feedback_at: number | null; next_feedback_at: number | null;
  observed_interval_s: number | null; inference_latency_s: number | null;
  input_tokens: number; output_tokens: number; message: string; error: string | null;
  trace_revision: number;
  events: AgentAction[];
  configuration: { provider: string; endpoint: string; ollama_endpoint: string; default_model_id: string; models: ModelProfile[] };
};
export type LiveState = {
  navigation?: NavigationState | null;
  skill?: SkillState | null;
  proximity: ProximitySensors;
  interaction_mode: 'chat' | 'voice';
  camera: CameraFrame;
  robot_body_id: number; manual_placements: number;
  challenge: ChallengeState | null;
  run_id: string; episode_epoch: number; busy: boolean; stopped: boolean; assisted: boolean;
  observation: Observation; geometry: Geometry[]; snapshot: { simulated_time_s: number; poses: Pose[] };
  result: Result | null;
  agent: AgentState;
  realtime: { endpoint: string; deployment: string; voice: string; configured: boolean; target_model: 'gpt-realtime-2'; reasoning_effort: 'low' };
};