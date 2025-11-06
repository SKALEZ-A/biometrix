import { Request, Response, NextFunction } from 'express';
import { dbManager } from '../config/database.config';

interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  keyGenerator?: (req: Request) => string;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
}

export class RateLimitMiddleware {
  private config: RateLimitConfig;

  constructor(config: RateLimitConfig) {
    this.config = {
      windowMs: config.windowMs || 60000,
      maxRequests: config.maxRequests || 100,
      keyGenerator: config.keyGenerator || this.defaultKeyGenerator,
      skipSuccessfulRequests: config.skipSuccessfulRequests || false,
      skipFailedRequests: config.skipFailedRequests || false,
    };
  }

  private defaultKeyGenerator(req: Request): string {
    return req.ip || req.socket.remoteAddress || 'unknown';
  }

  middleware() {
    return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
      try {
        const redis = dbManager.getRedis();
        const key = `rate_limit:${this.config.keyGenerator!(req)}`;
        const now = Date.now();
        const windowStart = now - this.config.windowMs;

        // Remove old entries
        await redis.zRemRangeByScore(key, 0, windowStart);

        // Count requests in current window
        const requestCount = await redis.zCard(key);

        if (requestCount >= this.config.maxRequests) {
          const oldestRequest = await redis.zRange(key, 0, 0, { REV: false });
          const resetTime = oldestRequest.length > 0
            ? parseInt(oldestRequest[0]) + this.config.windowMs
            : now + this.config.windowMs;

          res.status(429).json({
            success: false,
            error: 'Too many requests',
            code: 'RATE_LIMIT_EXCEEDED',
            retryAfter: Math.ceil((resetTime - now) / 1000),
            limit: this.config.maxRequests,
            windowMs: this.config.windowMs,
          });
          return;
        }

        // Add current request
        await redis.zAdd(key, { score: now, value: `${now}:${Math.random()}` });
        await redis.expire(key, Math.ceil(this.config.windowMs / 1000));

        // Set rate limit headers
        res.setHeader('X-RateLimit-Limit', this.config.maxRequests.toString());
        res.setHeader('X-RateLimit-Remaining', (this.config.maxRequests - requestCount - 1).toString());
        res.setHeader('X-RateLimit-Reset', new Date(now + this.config.windowMs).toISOString());

        next();
      } catch (error) {
        console.error('Rate limit middleware error:', error);
        // Fail open - allow request if rate limiting fails
        next();
      }
    };
  }

  static createLimiter(config: RateLimitConfig) {
    const limiter = new RateLimitMiddleware(config);
    return limiter.middleware();
  }
}

// Predefined rate limiters
export const standardRateLimit = RateLimitMiddleware.createLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  maxRequests: 100,
});

export const strictRateLimit = RateLimitMiddleware.createLimiter({
  windowMs: 15 * 60 * 1000,
  maxRequests: 20,
});

export const apiKeyRateLimit = RateLimitMiddleware.createLimiter({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 1000,
  keyGenerator: (req: Request) => req.headers['x-api-key'] as string || 'unknown',
});
