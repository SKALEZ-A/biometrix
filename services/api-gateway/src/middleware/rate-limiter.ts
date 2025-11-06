import { Request, Response, NextFunction } from 'express';
import { RedisClient } from '../../../packages/shared/src/cache/redis';

interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  keyGenerator?: (req: Request) => string;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
}

interface RateLimitInfo {
  limit: number;
  current: number;
  remaining: number;
  resetTime: number;
}

export class RateLimiter {
  private redis: RedisClient;
  private config: RateLimitConfig;

  constructor(config: RateLimitConfig) {
    this.config = {
      skipSuccessfulRequests: false,
      skipFailedRequests: false,
      ...config
    };
    this.redis = new RedisClient();
  }

  private generateKey(req: Request): string {
    if (this.config.keyGenerator) {
      return this.config.keyGenerator(req);
    }
    const ip = req.ip || req.connection.remoteAddress || 'unknown';
    const userId = (req as any).user?.id || 'anonymous';
    return `rate_limit:${userId}:${ip}`;
  }

  async middleware(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const key = this.generateKey(req);
      const info = await this.checkLimit(key);

      res.setHeader('X-RateLimit-Limit', info.limit.toString());
      res.setHeader('X-RateLimit-Remaining', info.remaining.toString());
      res.setHeader('X-RateLimit-Reset', new Date(info.resetTime).toISOString());

      if (info.remaining < 0) {
        res.status(429).json({
          error: 'Too Many Requests',
          message: 'Rate limit exceeded',
          retryAfter: Math.ceil((info.resetTime - Date.now()) / 1000)
        });
        return;
      }

      const originalSend = res.send;
      res.send = function(data: any): Response {
        const statusCode = res.statusCode;
        const shouldSkip = 
          (statusCode < 400 && this.config.skipSuccessfulRequests) ||
          (statusCode >= 400 && this.config.skipFailedRequests);

        if (!shouldSkip) {
          this.incrementCounter(key).catch(console.error);
        }

        return originalSend.call(this, data);
      }.bind(this);

      next();
    } catch (error) {
      console.error('Rate limiter error:', error);
      next();
    }
  }

  private async checkLimit(key: string): Promise<RateLimitInfo> {
    const now = Date.now();
    const windowStart = now - this.config.windowMs;

    const multi = this.redis.multi();
    multi.zremrangebyscore(key, 0, windowStart);
    multi.zcard(key);
    multi.zadd(key, now, `${now}-${Math.random()}`);
    multi.expire(key, Math.ceil(this.config.windowMs / 1000));

    const results = await multi.exec();
    const current = results[1][1] as number;

    return {
      limit: this.config.maxRequests,
      current,
      remaining: Math.max(0, this.config.maxRequests - current),
      resetTime: now + this.config.windowMs
    };
  }

  private async incrementCounter(key: string): Promise<void> {
    const now = Date.now();
    await this.redis.zadd(key, now, `${now}-${Math.random()}`);
  }

  async resetLimit(identifier: string): Promise<void> {
    const key = `rate_limit:${identifier}`;
    await this.redis.del(key);
  }

  async getLimitInfo(identifier: string): Promise<RateLimitInfo | null> {
    const key = `rate_limit:${identifier}`;
    const now = Date.now();
    const windowStart = now - this.config.windowMs;

    await this.redis.zremrangebyscore(key, 0, windowStart);
    const current = await this.redis.zcard(key);

    if (current === 0) return null;

    return {
      limit: this.config.maxRequests,
      current,
      remaining: Math.max(0, this.config.maxRequests - current),
      resetTime: now + this.config.windowMs
    };
  }
}

export class AdaptiveRateLimiter extends RateLimiter {
  private suspiciousThreshold: number = 0.8;
  private trustScoreCache: Map<string, number> = new Map();

  async middleware(req: Request, res: Response, next: NextFunction): Promise<void> {
    const trustScore = await this.calculateTrustScore(req);
    const adjustedConfig = this.adjustLimitsBasedOnTrust(trustScore);
    
    const tempLimiter = new RateLimiter(adjustedConfig);
    return tempLimiter.middleware(req, res, next);
  }

  private async calculateTrustScore(req: Request): Promise<number> {
    const userId = (req as any).user?.id;
    if (!userId) return 0.5;

    if (this.trustScoreCache.has(userId)) {
      return this.trustScoreCache.get(userId)!;
    }

    let score = 0.5;
    const userAge = await this.getUserAccountAge(userId);
    const violationHistory = await this.getViolationHistory(userId);
    const successRate = await this.getSuccessRate(userId);

    if (userAge > 365) score += 0.2;
    else if (userAge > 90) score += 0.1;

    if (violationHistory === 0) score += 0.2;
    else if (violationHistory < 3) score += 0.1;
    else score -= 0.3;

    if (successRate > 0.95) score += 0.1;
    else if (successRate < 0.7) score -= 0.2;

    score = Math.max(0, Math.min(1, score));
    this.trustScoreCache.set(userId, score);

    setTimeout(() => this.trustScoreCache.delete(userId), 300000);

    return score;
  }

  private adjustLimitsBasedOnTrust(trustScore: number): RateLimitConfig {
    const baseLimit = this.config.maxRequests;
    let adjustedLimit = baseLimit;

    if (trustScore > 0.8) {
      adjustedLimit = Math.floor(baseLimit * 1.5);
    } else if (trustScore < 0.3) {
      adjustedLimit = Math.floor(baseLimit * 0.5);
    }

    return {
      ...this.config,
      maxRequests: adjustedLimit
    };
  }

  private async getUserAccountAge(userId: string): Promise<number> {
    return 180;
  }

  private async getViolationHistory(userId: string): Promise<number> {
    return 0;
  }

  private async getSuccessRate(userId: string): Promise<number> {
    return 0.95;
  }
}

export const createRateLimiter = (config: RateLimitConfig): RateLimiter => {
  return new RateLimiter(config);
};

export const createAdaptiveRateLimiter = (config: RateLimitConfig): AdaptiveRateLimiter => {
  return new AdaptiveRateLimiter(config);
};
