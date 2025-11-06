import { Request, Response, NextFunction } from 'express';
import { RedisClient } from '../../../packages/shared/src/cache/redis';

interface ThrottleConfig {
  windowMs: number;
  maxRequests: number;
  keyGenerator?: (req: Request) => string;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
}

export class RequestThrottler {
  private redis: RedisClient;
  private config: ThrottleConfig;

  constructor(config: ThrottleConfig) {
    this.config = {
      windowMs: 60000,
      maxRequests: 100,
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
    return `throttle:${ip}:${userId}:${req.path}`;
  }

  public middleware() {
    return async (req: Request, res: Response, next: NextFunction) => {
      try {
        const key = this.generateKey(req);
        const current = await this.redis.get(key);
        const count = current ? parseInt(current, 10) : 0;

        if (count >= this.config.maxRequests) {
          res.status(429).json({
            error: 'Too Many Requests',
            message: 'Request throttle limit exceeded',
            retryAfter: Math.ceil(this.config.windowMs / 1000)
          });
          return;
        }

        await this.redis.incr(key);
        if (count === 0) {
          await this.redis.expire(key, Math.ceil(this.config.windowMs / 1000));
        }

        res.setHeader('X-RateLimit-Limit', this.config.maxRequests.toString());
        res.setHeader('X-RateLimit-Remaining', (this.config.maxRequests - count - 1).toString());

        next();
      } catch (error) {
        console.error('Throttle middleware error:', error);
        next();
      }
    };
  }

  public async resetKey(key: string): Promise<void> {
    await this.redis.del(key);
  }

  public async getUsage(key: string): Promise<number> {
    const current = await this.redis.get(key);
    return current ? parseInt(current, 10) : 0;
  }
}

export const createThrottler = (config: ThrottleConfig) => {
  return new RequestThrottler(config);
};
