import { Request, Response, NextFunction } from 'express';
import { createHmac, timingSafeEqual } from 'crypto';
import { logger } from '../utils/logger';
import { JwtPayload, UserRole } from '../types/auth';

export interface AuthRequest extends Request {
  user?: JwtPayload;
  apiKey?: string;
}

// Verify JWT token
function verifyJwt(token: string, secret: string): JwtPayload | null {
  try {
    const [header, payload, signature] = token.split('.');
    if (!header || !payload || !signature) return null;

    const decodedPayload = JSON.parse(Buffer.from(payload, 'base64').toString());
    const expectedSignature = createHmac('sha256', secret)
      .update(`${header}.${payload}`)
      .digest('base64url');

    if (timingSafeEqual(Buffer.from(signature), Buffer.from(expectedSignature))) {
      return decodedPayload as JwtPayload;
    }
    return null;
  } catch {
    return null;
  }
}

// Verify API key
function verifyApiKey(apiKey: string, validKeys: string[]): boolean {
  return validKeys.some(key => timingSafeEqual(Buffer.from(apiKey), Buffer.from(key)));
}

export const authMiddleware = (req: AuthRequest, res: Response, next: NextFunction) => {
  const authHeader = req.headers.authorization;
  const apiKeyHeader = req.headers['x-api-key'];
  const secret = process.env.JWT_SECRET || 'fallback-secret-change-in-prod';
  const validApiKeys = (process.env.VALID_API_KEYS || '').split(',').filter(Boolean);

  if (!authHeader && !apiKeyHeader) {
    return res.status(401).json({ error: 'Unauthorized: No credentials provided' });
  }

  let user: JwtPayload | undefined;

  if (authHeader && authHeader.startsWith('Bearer ')) {
    const token = authHeader.slice(7);
    user = verifyJwt(token, secret);
    if (!user) {
      return res.status(401).json({ error: 'Unauthorized: Invalid JWT token' });
    }
    req.user = user;
  } else if (apiKeyHeader) {
    if (!verifyApiKey(apiKeyHeader as string, validApiKeys)) {
      return res.status(401).json({ error: 'Unauthorized: Invalid API key' });
    }
    req.apiKey = apiKeyHeader as string;
  }

  // Role-based access check
  if (req.user && req.user.role !== 'admin' && req.path.includes('/admin')) {
    return res.status(403).json({ error: 'Forbidden: Admin access required' });
  }

  logger.info(`Authenticated request from ${req.user?.sub || 'API key'} to ${req.path}`);
  next();
};

// Optional: Rate limiting middleware (simple in-memory)
const requestCounts: { [key: string]: number } = {};
const RATE_LIMIT = 100; // requests per minute
export const rateLimit = (req: AuthRequest, res: Response, next: NextFunction) => {
  const ip = req.ip || req.connection.remoteAddress || 'unknown';
  const now = Date.now();
  const minuteAgo = now - 60000;

  // Clean old entries
  Object.keys(requestCounts).forEach(key => {
    if (parseInt(key.split(':')[1]) < minuteAgo) delete requestCounts[key];
  });

  const key = `${ip}:${now}`;
  requestCounts[key] = (requestCounts[key] || 0) + 1;

  const total = Object.values(requestCounts).filter((_, k) => k.startsWith(ip + ':')).length;
  if (total > RATE_LIMIT) {
    return res.status(429).json({ error: 'Rate limit exceeded' });
  }

  next();
};
