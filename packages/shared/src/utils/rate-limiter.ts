import { logger } from './logger';

interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
}

interface RateLimitEntry {
  count: number;
  resetTime: number;
}

export class RateLimiter {
  private requests: Map<string, RateLimitEntry> = new Map();
  private config: RateLimitConfig;
  private cleanupInterval: NodeJS.Timeout;

  constructor(config: RateLimitConfig) {
    this.config = config;
    this.cleanupInterval = setInterval(() => this.cleanup(), 60000);
  }

  isAllowed(identifier: string): boolean {
    const now = Date.now();
    const entry = this.requests.get(identifier);

    if (!entry || now > entry.resetTime) {
      this.requests.set(identifier, {
        count: 1,
        resetTime: now + this.config.windowMs
      });
      return true;
    }

    if (entry.count < this.config.maxRequests) {
      entry.count++;
      return true;
    }

    logger.warn('Rate limit exceeded', { identifier });
    return false;
  }

  getRemainingRequests(identifier: string): number {
    const entry = this.requests.get(identifier);
    if (!entry || Date.now() > entry.resetTime) {
      return this.config.maxRequests;
    }
    return Math.max(0, this.config.maxRequests - entry.count);
  }

  getResetTime(identifier: string): number {
    const entry = this.requests.get(identifier);
    return entry ? entry.resetTime : Date.now();
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.requests.entries()) {
      if (now > entry.resetTime) {
        this.requests.delete(key);
      }
    }
  }

  destroy(): void {
    clearInterval(this.cleanupInterval);
  }
}
