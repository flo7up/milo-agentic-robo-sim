import type { LiveState } from './types';

type Sample = {point:[number,number];time:number;received:number;placements:number};

export class TravelledPath {
  static readonly maximumSegments = 4096;
  readonly positions:number[] = [];
  revision = 0;
  private episode = '';
  private previous?:Sample;
  private anchor?:[number,number];

  sample(state:LiveState, connected:boolean, received=performance.now()) {
    const episode = `${state.run_id}:${state.episode_epoch}`;
    if (this.episode !== episode) {
      this.episode = episode;
      this.positions.length = 0;
      this.previous = undefined;
      this.anchor = undefined;
      this.revision++;
    }
    const base = state.snapshot.poses.find(pose=>pose.key===`${state.robot_body_id}:-1`);
    const time = state.snapshot.simulated_time_s;
    if (!connected || state.power?.on === false || !base || ![...base.position,time,received].every(Number.isFinite)) {
      this.previous = undefined;
      this.anchor = undefined;
      return;
    }
    const point:[number,number] = [base.position[0],base.position[1]];
    const previous = this.previous;
    this.previous = {point,time,received,placements:state.manual_placements};
    const elapsed = previous ? time-previous.time : 0;
    const distance = previous ? Math.hypot(point[0]-previous.point[0],point[1]-previous.point[1]) : 0;
    if (!previous || previous.placements !== state.manual_placements || elapsed < 0 || elapsed > .5
        || received-previous.received > 1500 || distance > .8*elapsed+.03 || (elapsed === 0 && distance > 1e-6)) {
      this.anchor = point;
      return;
    }
    if (elapsed === 0 || !this.anchor || Math.hypot(point[0]-this.anchor[0],point[1]-this.anchor[1]) < .02) return;
    this.positions.push(...this.anchor,.035,...point,.035);
    this.anchor = point;
    if (this.positions.length > TravelledPath.maximumSegments*6) this.positions.splice(0,6);
    this.revision++;
  }
}