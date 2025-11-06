import { RedisClient } from '../../../packages/shared/src/cache/redis';

export interface StateSnapshot {
  key: string;
  value: any;
  version: number;
  timestamp: number;
  checksum: string;
}

export class StateManager {
  private redisClient: RedisClient;
  private localState: Map<string, any>;
  private stateVersion: Map<string, number>;
  private checkpointInterval: number = 10000;

  constructor() {
    this.redisClient = new RedisClient();
    this.localState = new Map();
    this.stateVersion = new Map();
    this.initializeCheckpointing();
  }

  async get(key: string): Promise<any> {
    if (this.localState.has(key)) {
      return this.localState.get(key);
    }

    const value = await this.redisClient.get(`state:${key}`);
    if (value) {
      const parsed = JSON.parse(value);
      this.localState.set(key, parsed);
      return parsed;
    }

    return null;
  }

  async set(key: string, value: any): Promise<void> {
    this.localState.set(key, value);
    const version = (this.stateVersion.get(key) || 0) + 1;
    this.stateVersion.set(key, version);

    await this.redisClient.set(
      `state:${key}`,
      JSON.stringify(value),
      3600
    );
  }

  async delete(key: string): Promise<void> {
    this.localState.delete(key);
    this.stateVersion.delete(key);
    await this.redisClient.delete(`state:${key}`);
  }

  async createSnapshot(keys: string[]): Promise<StateSnapshot[]> {
    const snapshots: StateSnapshot[] = [];

    for (const key of keys) {
      const value = await this.get(key);
      if (value !== null) {
        snapshots.push({
          key,
          value,
          version: this.stateVersion.get(key) || 0,
          timestamp: Date.now(),
          checksum: this.calculateChecksum(value)
        });
      }
    }

    return snapshots;
  }

  async restoreSnapshot(snapshots: StateSnapshot[]): Promise<void> {
    for (const snapshot of snapshots) {
      const checksum = this.calculateChecksum(snapshot.value);
      if (checksum === snapshot.checksum) {
        await this.set(snapshot.key, snapshot.value);
        this.stateVersion.set(snapshot.key, snapshot.version);
      } else {
        throw new Error(`Checksum mismatch for key: ${snapshot.key}`);
      }
    }
  }

  private calculateChecksum(value: any): string {
    const str = JSON.stringify(value);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(16);
  }

  private initializeCheckpointing(): void {
    setInterval(async () => {
      const keys = Array.from(this.localState.keys());
      if (keys.length > 0) {
        await this.createSnapshot(keys);
      }
    }, this.checkpointInterval);
  }

  clear(): void {
    this.localState.clear();
    this.stateVersion.clear();
  }
}
