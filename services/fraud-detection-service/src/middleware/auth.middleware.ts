import { Request, Response, NextFunction } from 'express';
import { verifyJWT } from '@shared/crypto/jwt';
import { logger } from '@shared/utils/logger';

export interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
    role: string;
    permissions: string[];
  };
}

export const authMiddleware = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({
        error: 'Unauthorized',
        message: 'Missing or invalid authorization header',
      });
      return;
    }

    const token = authHeader.substring(7);
    const decoded = await verifyJWT(token);

    if (!decoded) {
      res.status(401).json({
        error: 'Unauthorized',
        message: 'Invalid or expired token',
      });
      return;
    }

    req.user = {
      id: decoded.userId,
      email: decoded.email,
      role: decoded.role,
      permissions: decoded.permissions || [],
    };

    next();
  } catch (error) {
    logger.error('Authentication error', error);
    res.status(401).json({
      error: 'Unauthorized',
      message: 'Authentication failed',
    });
  }
};

export const requirePermission = (permission: string) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({
        error: 'Unauthorized',
        message: 'User not authenticated',
      });
      return;
    }

    if (!req.user.permissions.includes(permission) && req.user.role !== 'admin') {
      res.status(403).json({
        error: 'Forbidden',
        message: `Missing required permission: ${permission}`,
      });
      return;
    }

    next();
  };
};
