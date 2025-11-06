import { Request, Response, NextFunction } from 'express';
import { verifyToken } from '@shared/crypto/jwt';
import { logger } from '@shared/utils/logger';

export interface AuthRequest extends Request {
  user?: {
    userId: string;
    email: string;
    roles: string[];
    permissions: string[];
  };
}

export const authMiddleware = async (req: AuthRequest, res: Response, next: NextFunction): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({ success: false, error: 'No token provided' });
      return;
    }

    const token = authHeader.substring(7);
    const decoded = await verifyToken(token);
    
    req.user = {
      userId: decoded.userId,
      email: decoded.email,
      roles: decoded.roles || [],
      permissions: decoded.permissions || [],
    };

    next();
  } catch (error) {
    logger.error('Auth middleware error:', error);
    res.status(401).json({ success: false, error: 'Invalid token' });
  }
};

export const requirePermission = (permission: string) => {
  return (req: AuthRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({ success: false, error: 'Unauthorized' });
      return;
    }

    if (!req.user.permissions.includes(permission)) {
      res.status(403).json({ success: false, error: 'Insufficient permissions' });
      return;
    }

    next();
  };
};

export const requireRole = (role: string) => {
  return (req: AuthRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({ success: false, error: 'Unauthorized' });
      return;
    }

    if (!req.user.roles.includes(role)) {
      res.status(403).json({ success: false, error: 'Insufficient role' });
      return;
    }

    next();
  };
};
