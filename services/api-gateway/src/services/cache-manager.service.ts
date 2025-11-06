import { Request, Response } from 'express';
import { createHash } from 'crypto';
import { RedisClient } from '../../../packages/shared/src/cache/redis';

interface CacheOptions {
  ttl: number;
  keyPrefix?: string;
  varyBy?: string[];
  excludeQuery?: string[];
  cacheControl?: string;
}

interface CachedResponse {
  statusCode: number;
  headers: Record<string, string>;
  body: any;
  timestamp: number;
}

export class CacheManagerService {
  private redis: RedisClient;
  private defaultTTL = 300;

  constructor() {
    this.redis = new RedisClient();
  }

  private generateCacheKey(req: Request, options: CacheOptions): string {
    const parts = [
      options.keyPrefix || 'cache',
      req.method,
      req.path
    ];

    if (options.varyBy) {
      options.varyBy.forEach(header => {
        const value = req.headers[header.toLowerCase()];
        if (value) {
          parts.push(`${header}:${value}`);
        }
      });
    }

    const query = { ...req.query };
    if (options.excludeQuery) {
      options.excludeQuery.forEach(key => delete query[key]);
    }

    if (Object.keys(query).length > 0) {
      parts.push(JSON.stringify(query));
    }

    return createHash('sha256')
      .update(parts.join(':'))
      .digest('hex');
  }

  public async get(req: Request, options: CacheOptions): Promise<CachedResponse | null> {
    const key = this.generateCacheKey(req, options);
    const cached = await this.redis.get(key);

    if (!cached) {
      return null;
    }

    try {
      const response: CachedResponse = JSON.parse(cached);
      const age = Math.floor((Date.now() - response.timestamp) / 1000);
      
      if (age > options.ttl) {
        await this.redis.del(key);
        return null;
      }

      return response;
    } catch (error) {
      console.error('Cache parse error:', error);
      return null;
    }
  }

  public async set(
    req: Request,
    res: Response,
    body: any,
    options: CacheOptions
  ): Promise<void> {
    const key = this.generateCacheKey(req, options);
    
    const cachedResponse: CachedResponse = {
      statusCode: res.statusCode,
      headers: this.extractHeaders(res),
      body,
      timestamp: Date.now()
    };

    await this.redis.set(
      key,
      JSON.stringify(cachedResponse),
      options.ttl || this.defaultTTL
    );
  }

  private extractHeaders(res: Response): Record<string, string> {
    const headers: Record<string, string> = {};
    const headerNames = ['content-type', 'etag', 'last-modified'];
    
    headerNames.forEach(name => {
      const value = res.getHeader(name);
      if (value) {
        headers[name] = value.toString();
      }
    });

    return headers;
  }

  public async invalidate(pattern: string): Promise<number> {
    const keys = await this.redis.keys(`cache:${pattern}*`);
    if (keys.length === 0) {
      return 0;
    }

    await Promise.all(keys.map(key => this.redis.del(key)));
    return keys.length;
  }

  public async invalidateByTag(tag: string): Promise<number> {
    const tagKey = `cache:tag:${tag}`;
    const keys = await this.redis.smembers(tagKey);
    
    if (keys.length === 0) {
      return 0;
    }

    await Promise.all([
      ...keys.map(key => this.redis.del(key)),
      this.redis.del(tagKey)
    ]);

    return keys.length;
  }

  public async tagCache(cacheKey: string, tags: string[]): Promise<void> {
    await Promise.all(
      tags.map(tag => this.redis.sadd(`cache:tag:${tag}`, cacheKey))
    );
  }

  public middleware(options: CacheOptions) {
    return async (req: Request, res: Response, next: Function) => {
      if (req.method !== 'GET') {
        next();
        return;
      }

      const cached = await this.get(req, options);

      if (cached) {
        Object.entries(cached.headers).forEach(([key, value]) => {
          res.setHeader(key, value);
        });

        const age = Math.floor((Date.now() - cached.timestamp) / 1000);
        res.setHeader('X-Cache', 'HIT');
        res.setHeader('Age', age.toString());
        
        if (options.cacheControl) {
          res.setHeader('Cache-Control', options.cacheControl);
        }

        res.status(cached.statusCode).json(cached.body);
        return;
      }

      res.setHeader('X-Cache', 'MISS');

      const originalJson = res.json.bind(res);
      res.json = (body: any) => {
        this.set(req, res, body, options).catch(console.error);
        return originalJson(body);
      };

      next();
    };
  }
}

export const cacheManager = new CacheManagerService();
