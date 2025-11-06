import { Request, Response, NextFunction } from 'express';
import { RateLimiterMemory } from 'rate-limiter-flexible';

interface RateLimitOptions {
  maxRequests: number;
  windowMs: number;
}

const limiters = new Map<string, RateLimiterMemory>();

export const rateLimitMiddleware = (options: RateLimitOptions) => {
  const key = `${options.maxRequests}-${options.windowMs}`;
  
  if (!limiters.has(key)) {
    limiters.set(key, new RateLimiterMemory({
      points: options.maxRequests,
      duration: options.windowMs / 1000
    }));
  }
  
  const limiter = limiters.get(key)!;
  
  return async (req: Request, res: Response, next: NextFunction) => {
    const identifier = req.ip || req.socket.remoteAddress || 'unknown';
    
    try {
      await limiter.consume(identifier);
      next();
    } catch (error) {
      res.status(429).json({
        success: false,
        message: 'Too many requests, please try again later'
      });
    }
  };
};
