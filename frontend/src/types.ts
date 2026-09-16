export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];
export type Pose = { key: string; position: Vec3; quaternion: Quat };
export type Geometry = Pose & { type: number; dimensions: number[]; color: number[]; texture: string | null; name?: string };
export type ProximitySensors = { simulated_time_s: number; max_range_m: number;
  distances: { direction: string; bearing_rad: number; origin_base_m?:number[]; distance_m: number | null; status: 'hit' | 'clear' | 'occluded' }[];
  collisions: { direction: string; force_n: number }[] };
export type CameraFrame = { seq: number; frame_ref: string; simulated_time_s: number; url: string };
export type MotionZones = {run_id:string;episode_epoch:number;frame:'robot_base';clock:'monotonic';captured_at_s:number;
  display_pose?:{frame:'world';position_m:number[];yaw_rad:number};
  sensor_age_s:number | null;maximum_sensor_age_s:number;valid_for_s:number;odometry_m_rad:number[];stale:boolean;source:string;reason:string;motion_authorized:false;
  footprint:{lower_xy_m:number[];upper_xy_m:number[];radius_m:number};planning_radius_m:number;map_margin_m:number | null;
  sectors:{sector:number;inner_m:number;outer_m:number;start_rad:number;end_rad:number;status:'clear'|'restricted'|'unknown'|'unavailable';reason:string}[];
  preview_speed_mps:number;speed_basis:string;beam_stop_distance_m:number;beam_checked_directions:string[];controller_stop:NavigationDiagnostics['stop']};
export type SpatialTelemetry = {run_id:string;episode_epoch:number;received_at_ms:number | null;enabled:boolean | null;paused:boolean;error:string | null;
  request_duration_ms?:number;
  motion_zones?:MotionZones | null;
  frame:{sequence:number;simulated_time_s:number} | null;
  map:{age_s:number | null;stale:boolean;observed_floor_cells:number;obstacle_cells:number} | null};
export type ExecutionMode = 'single_step' | 'navigation_plan' | 'supervised_policy' | 'local_navigation' | 'luna_navigation' | 'luna_continuous';
export type LocalModelStatus = {
  run_id: string; checkpoint: string; case?: number; challenge_id?: string; challenge_title?: string;
  phase: 'checking' | 'loading' | 'warming' | 'supervising' | 'running' | 'saving' | 'completed' | 'failed' | 'interrupted';
  supervisor_turns?: number; instruction?: string;
  requests_completed: number; request_limit: number; elapsed_s: number; success: boolean | null;
};
export type SkillState = {
  revision: number; motion_revision: number; status: 'idle' | 'running' | 'awaiting_policy' | 'completed' | 'cancelled' | 'failed';
  skill: string | null; instruction: string; reason: string; remaining_s: number; checkpoint: string;
  policy_requests: number; rejected_chunks: number; policy_latency_s: number | null; completion_source: 'supervisor' | null;
};
export type AuthorizationEvent = {event?:string;at_s:number;result?:string;rejection_reason?:string | null;
  expires_at_s?:number | null;initiator?:string;reason?:string;revision?:number};
export type NavigationDiagnostics = {
  clock:'monotonic' | 'simulation' | 'test';authorization_id:string | null;scope:string;issuer:string;renewal_owner:string;
  task_id:string | null;objective_id:string | null;issued_at_s:number | null;renewed_at_s:number | null;expires_at_s:number | null;
  last_renewal:AuthorizationEvent | null;last_controller_tick_at_s:number | null;maximum_recent_tick_gap_s:number;tick_window_s:number;
  sensor_at_last_command:{kind:string;clock:string;captured_at_s?:number | null;age_s?:number | null} | null;
  stop:{at_s:number;initiator:string;reason:string} | null;recent_events:AuthorizationEvent[];
};
export type ObjectiveAuthorization = {id:string;scope:string;issuer:string;renewal_owner:string;validator:string;clock:string;
  issued_at_s:number;renewed_at_s:number | null;expires_at_s:number;last_renewal:AuthorizationEvent | null;
  stop:NavigationDiagnostics['stop'];recent_events:AuthorizationEvent[]};
export type RouteFailure = {phase:string;reason:string;segment:number;frontier_id?:string | null;elapsed_s:number;clock?:string;
  motion_diagnostics?:NavigationDiagnostics | null};
export type TaskDiagnostics = {task_id?:string;status?:string;reason?:string;frontier_id?:string | null;segments?:number;retries?:number;route_failures?:RouteFailure[]};
export type ObservedSensorContext = {localization?:{status:string;age_s:number;age_basis?:string;sample_clock?:string;tracking_method?:string;continuous_pose_correction?:boolean};
  task?:TaskDiagnostics | null};
export type NavigationState = {
  revision: number; status: 'idle' | 'running' | 'awaiting_feedback' | 'completed' | 'failed' | 'cancelled'; reason: string;
  active_step: number | null; remaining_s: number; expires_in_s: number; velocity_mps_radps: number[];
  travel_m: number; scan_span_rad: number;
  diagnostics?: NavigationDiagnostics | null;
  steps: { skill: 'inspect_room' | 'locate_doorway' | 'approach' | 'cross'; goal: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'; evidence: string }[];
};
export type ManualPlacement = { run_id: string; episode_epoch: number; observation_seq: number; xy_m: [number, number] };
export type ChallengeId = 'bench' | 'park' | 'tidy' | 'sort' | 'recharge' | 'apartment' | 'kitchen_bathroom' | 'clinic_delivery' | 'warehouse' | 'inspection' | 'workshop' | 'local_park' | 'pedestrian_crossing' | 'flat_kitchen' | 'furniture_circuit';
export type ChallengeEnvironment = 'standalone' | 'shared_apartment_v1';
export type ChallengePreset = { id: ChallengeId; environment?: ChallengeEnvironment; title: string; skill: string; goal: string; objectives: string[]; suggested_turn_limit: number;
  category?: 'Navigation' | 'Perception' | 'Manipulation'; difficulty?: 'Foundation' | 'Advanced';
  orbit?: {target: 'table' | 'sofa' | 'chair' | 'floor lamp'; direction: 'clockwise' | 'counterclockwise'} };
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
  spatial?:ObservedSensorContext | null;
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
  { kind: 'feedback'; payload: { observation: Observation; observed_map_snapshot?:{age_s:number;capture_clock?:string;age_basis?:string;geometry_age_s?:number | null;geometry_age_basis?:string;frame?:string;revision?:number | string}; image_roles?: string[]; image_detail: string; history_turns: number[]; input_items: number; images_in_request: number; tool_result_call_ids: string[]; context_mode?: string; memory_frame_seq?: number | null; recent_actions?: Partial<AgentAction>[]; collision_feedback?: CollisionFeedback | null; images_per_request?: number; context_tokens?: number; retained_context_tokens_estimate?: number; context_estimator?: string; camera_frames?: { seq: number; frame_ref: string; simulated_time_s: number; wall_timestamp: number; current: boolean }[]; camera_history?: { frames: { frame_id: string; simulated_time_s: number }[] }; historical_original?: { frame_id: string; simulated_time_s: number } | null } } |
  { kind: 'response'; payload: { text: string; text_truncated: boolean; status: string; latency_s: number; input_tokens: number | null; output_tokens: number | null; calls: { call_id: string; name: string; arguments: string }[]; calls_truncated: boolean; refusals: string[] } } |
  { kind: 'tool'; payload: { tool: string; arguments: unknown; call_id: string } } |
  { kind: 'policy'; payload: Record<string, unknown> & {task?:TaskDiagnostics | null} } |
  { kind: 'result'; payload: { tool: string; call_id: string; result: Partial<Result> & { status: string; sensor_deltas?: unknown }; model_result?: Record<string, unknown> & { collision_feedback?: CollisionFeedback } } }
);
export type ExchangeFeed = { session_id: string | null; revision: number; first_id: number; capacity: number; events: ExchangeEntry[] };
export type AgentState = {
  inference_budget?: {requests: number; tokens: number; max_requests: number; max_tokens: number; usage_unknown: boolean};
  unified_mission?: boolean;
  mission?: {mission_id: string; phase: string; remaining_s: number; reason: string;
    objective?: {action:string;status:string;remaining_s:number;remaining_travel_m:number;reason?:string;authorization?:ObjectiveAuthorization} | null;
    plan: {kind: string; target: string; return_home: boolean} | null; receipts: Record<string, unknown>} | null;
  execution_mode: ExecutionMode;
  navigation_backend?: 'builtin' | 'nav2';
  run_messages?: {id:string;role:'user'|'assistant';text:string;status:string;source?:string}[];
  run_memory?: {revision:number;visited_positions_m:number[][];inspected_heading_sectors_here:number[];
    rotation_without_translation_rad:number;recent_actions:{action:string;status:string;distance_m:number;turn_rad:number;reason:string}[];
    progress?:{stagnant_actions:number;recovery_needed:boolean;recovery_attempts:number;recovery_limit:number}};
  local_model?: LocalModelStatus | null;
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
  recording?: {enabled:boolean;directory:string;active:boolean;status:'off'|'ready'|'recording'|'finalizing';
    run_directory:string | null;samples:number | null;error:string | null};
  power?: {on: boolean; mode: 'off' | 'idle' | 'working'; revision: number; idle_sensor_interval_s: number};
  preference_error?: string | null;
  map_setup?: { reuse_saved_map: boolean; map_id: string | null; name: string | null; revision: number | null; localization: string };
  rendering?: 'tiny' | 'enhanced';
  continuous_navigation?: {status: string; reason: string; remaining_m: number; updates: number; buffer_stops: number} | null;
  local_navigation_model?: {
    phase: 'unloaded' | 'loading' | 'ready' | 'inferencing' | 'error';
    checkpoint: string; load_count: number; process_id: number | null;
  };
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