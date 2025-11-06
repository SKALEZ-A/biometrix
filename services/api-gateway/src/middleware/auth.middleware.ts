import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { config } from '../config/gateway.config';
import { logger } from '@shared/utils/logger';

export interface AuthRequest extends Request {
  user?: {
    id: string;
    email: string;
    role: string;
    permissions: string[];
  };
}

export const authMiddleware = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({ error: 'No token provided' });
      return;
    }

    const token = authHeader.substring(7);

    try {
      const decoded = jwt.verify(token, config.jwt.secret) as any;
      
      req.user = {
        id: decoded.id,
        email: decoded.email,
        role: decoded.role,
        permissions: decoded.permissions || [],
      };

      // Add user context to request headers for downstream services
      req.headers['x-user-id'] = decoded.id;
      req.headers['x-user-role'] = decoded.role;

      next();
    } catch (jwtError) {
      logger.warn('Invalid token', { error: jwtError });
      res.status(401).json({ error: 'Invalid or expired token' });
      return;
    }
  } catch (error) {
    logger.error('Authentication error', { error });
    res.status(500).json({ error: 'Authentication failed' });
  }
};

export const requireRole = (...roles: string[]) => {
  return (req: AuthRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }

    if (!roles.includes(req.user.role)) {
      logger.warn('Insufficient permissions', {
        userId: req.user.id,
        requiredRoles: roles,
        userRole: req.user.role,
      });
      res.status(403).json({ error: 'Insufficient permissions' });
      return;
    }

    next();
  };
};

export const requirePermission = (...permissions: string[]) => {
  return (req: AuthRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }

    const hasPermission = permissions.some(permission =>
      req.user!.permissions.includes(permission)
    );

    if (!hasPermission) {
      logger.warn('Missing required permission', {
        userId: req.user.id,
        requiredPermissions: permissions,
        userPermissions: req.user.permissions,
      });
      res.status(403).json({ error: 'Missing required permission' });
      return;
    }

    next();
  };
};
