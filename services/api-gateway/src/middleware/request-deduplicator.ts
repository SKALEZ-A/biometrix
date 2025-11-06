import { Request, Response, NextFunction } from 'express';
import { createHash } from 'crypto';
import { RedisClient } from '../../../packages/shared/src/cache/redis';

interface DeduplicationConfig {
  windowMs: number;
  keyGenerator?: (req: Request) => string;
  excludePaths?: string[];
  excludeMethods?: string[];
}

export class RequestDeduplicator {
  private redis: RedisClient;
  private config: DeduplicationConfig;

  constructor(config: DeduplicationConfig) {
    this.config = {
      windowMs: 5000,
      excludePaths: [],
      excludeMethods: ['GET', 'HEAD', 'OPTIONS'],
      ...config
    };
    this.redis = new RedisClient();
  }

  private generateRequestHash(req: Request): string {
    if (this.config.keyGenerator) {
      return this.config.keyGenerator(req);
    }

    const data = {
      method: req.method,
      path: req.path,
      query: req.query,
      body: req.body,
      userId: (req as any).user?.id || 'anonymous'
    };

    return createHash('sha256')
      .update(JSON.stringify(data))
      .digest('hex');
  }

  private shouldSkip(req: Request): boolean {
    if (this.config.excludeMethods?.includes(req.method)) {
      return true;
    }

    if (this.config.excludePaths?.some(path => req.path.startsWith(path))) {
      return true;
    }

    return false;
  }

  public middleware() {
    return async (req: Request, res: Response, next: NextFunction) => {
      if (this.shouldSkip(req)) {
        next();
        return;
      }

      try {
        const requestHash = this.generateRequestHash(req);
        const key = `dedup:${requestHash}`;
        
        const existing = await this.redis.get(key);

        if (existing) {
          res.status(409).json({
            error: 'Duplicate Request',
            message: 'This request is already being processed',
            requestId: requestHash
          });
          return;
        }

        await this.redis.set(
          key,
          JSON.stringify({ timestamp: Date.now() }),
          Math.ceil(this.config.windowMs / 1000)
        );

        res.on('finish', async () => {
          await this.redis.del(key);
        });

        next();
      } catch (error) {
        console.error('Deduplication middleware error:', error);
        next();
      }
    };
  }

  public async clearRequest(requestHash: string): Promise<void> {
    await this.redis.del(`dedup:${requestHash}`);
  }
}

export const createDeduplicator = (config: DeduplicationConfig) => {
  return new RequestDeduplicator(config);
};
