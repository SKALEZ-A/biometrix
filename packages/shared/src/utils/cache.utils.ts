import { Redis } from 'ioredis';

export interface CacheOptions {
  ttl?: number;
  prefix?: string;
}

export class CacheManager {
  private redis: Redis;
  private defaultTTL: number;
  private prefix: string;

  constructor(redis: Redis, options: CacheOptions = {}) {
    this.redis = redis;
    this.defaultTTL = options.ttl || 3600;
    this.prefix = options.prefix || 'cache:';
  }

  async get<T>(key: string): Promise<T | null> {
    const fullKey = this.getFullKey(key);
    const value = await this.redis.get(fullKey);

    if (!value) return null;

    try {
      return JSON.parse(value) as T;
    } catch {
      return value as T;
    }
  }

  async set<T>(key: string, value: T, ttl?: number): Promise<void> {
    const fullKey = this.getFullKey(key);
    const serialized = typeof value === 'string' ? value : JSON.stringify(value);
    const expiry = ttl || this.defaultTTL;

    await this.redis.setex(fullKey, expiry, serialized);
  }

  async delete(key: string): Promise<void> {
    const fullKey = this.getFullKey(key);
    await this.redis.del(fullKey);
  }

  async deletePattern(pattern: string): Promise<number> {
    const fullPattern = this.getFullKey(pattern);
    const keys = await this.redis.keys(fullPattern);

    if (keys.length === 0) return 0;

    return await this.redis.del(...keys);
  }

  async exists(key: string): Promise<boolean> {
    const fullKey = this.getFullKey(key);
    const result = await this.redis.exists(fullKey);
    return result === 1;
  }

  async getOrSet<T>(
    key: string,
    factory: () => Promise<T>,
    ttl?: number
  ): Promise<T> {
    const cached = await this.get<T>(key);

    if (cached !== null) {
      return cached;
    }

    const value = await factory();
    await this.set(key, value, ttl);
    return value;
  }

  async mget<T>(keys: string[]): Promise<(T | null)[]> {
    const fullKeys = keys.map(k => this.getFullKey(k));
    const values = await this.redis.mget(...fullKeys);

    return values.map(value => {
      if (!value) return null;
      try {
        return JSON.parse(value) as T;
      } catch {
        return value as T;
      }
    });
  }

  async mset<T>(entries: Array<{ key: string; value: T; ttl?: number }>): Promise<void> {
    const pipeline = this.redis.pipeline();

    for (const entry of entries) {
      const fullKey = this.getFullKey(entry.key);
      const serialized = typeof entry.value === 'string' ? entry.value : JSON.stringify(entry.value);
      const expiry = entry.ttl || this.defaultTTL;

      pipeline.setex(fullKey, expiry, serialized);
    }

    await pipeline.exec();
  }

  async increment(key: string, amount: number = 1): Promise<number> {
    const fullKey = this.getFullKey(key);
    return await this.redis.incrby(fullKey, amount);
  }

  async decrement(key: string, amount: number = 1): Promise<number> {
    const fullKey = this.getFullKey(key);
    return await this.redis.decrby(fullKey, amount);
  }

  async getTTL(key: string): Promise<number> {
    const fullKey = this.getFullKey(key);
    return await this.redis.ttl(fullKey);
  }

  async expire(key: string, ttl: number): Promise<boolean> {
    const fullKey = this.getFullKey(key);
    const result = await this.redis.expire(fullKey, ttl);
    return result === 1;
  }

  private getFullKey(key: string): string {
    return `${this.prefix}${key}`;
  }
}
