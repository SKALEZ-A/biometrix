import { Request, Response, NextFunction } from 'express';
import { RedisClient } from '@shared/cache/redis';
import { logger } from '@shared/utils/logger';

interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  keyPrefix: string;
}

const defaultConfig: RateLimitConfig = {
  windowMs: 60000, // 1 minute
  maxRequests: 100,
  keyPrefix: 'rate_limit:fraud_detection',
};

export class RateLimiter {
  private redis: RedisClient;
  private config: RateLimitConfig;

  constructor(config: Partial<RateLimitConfig> = {}) {
    this.redis = new RedisClient();
    this.config = { ...defaultConfig, ...config };
  }

  public middleware = async (
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> => {
    try {
      const identifier = this.getIdentifier(req);
      const key = `${this.config.keyPrefix}:${identifier}`;

      const current = await this.redis.incr(key);

      if (current === 1) {
        await this.redis.expire(key, Math.ceil(this.config.windowMs / 1000));
      }

      const remaining = Math.max(0, this.config.maxRequests - current);

      res.setHeader('X-RateLimit-Limit', this.config.maxRequests.toString());
      res.setHeader('X-RateLimit-Remaining', remaining.toString());
      res.setHeader('X-RateLimit-Reset', new Date(Date.now() + this.config.windowMs).toISOString());

      if (current > this.config.maxRequests) {
        logger.warn(`Rate limit exceeded for ${identifier}`);
        res.status(429).json({
          error: 'Too Many Requests',
          message: 'Rate limit exceeded. Please try again later.',
          retryAfter: Math.ceil(this.config.windowMs / 1000),
        });
        return;
      }

      next();
    } catch (error) {
      logger.error('Rate limiting error', error);
      next();
    }
  };

  private getIdentifier(req: Request): string {
    const user = (req as any).user;
    if (user && user.id) {
      return `user:${user.id}`;
    }
    return `ip:${req.ip}`;
  }
}

export const rateLimitMiddleware = new RateLimiter().middleware;
