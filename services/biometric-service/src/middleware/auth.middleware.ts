import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { appConfig } from '../config/app.config';

export interface AuthenticatedRequest extends Request {
  user?: {
    userId: string;
    email: string;
    role: string;
    permissions: string[];
  };
}

export class AuthMiddleware {
  static async verifyToken(
    req: AuthenticatedRequest,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const authHeader = req.headers.authorization;

      if (!authHeader || !authHeader.startsWith('Bearer ')) {
        res.status(401).json({
          success: false,
          error: 'No token provided',
          code: 'AUTH_TOKEN_MISSING',
        });
        return;
      }

      const token = authHeader.substring(7);

      try {
        const decoded = jwt.verify(token, appConfig.security.jwtSecret) as any;
        req.user = {
          userId: decoded.userId,
          email: decoded.email,
          role: decoded.role || 'user',
          permissions: decoded.permissions || [],
        };
        next();
      } catch (jwtError) {
        res.status(401).json({
          success: false,
          error: 'Invalid or expired token',
          code: 'AUTH_TOKEN_INVALID',
        });
        return;
      }
    } catch (error) {
      console.error('Auth middleware error:', error);
      res.status(500).json({
        success: false,
        error: 'Authentication failed',
        code: 'AUTH_ERROR',
      });
    }
  }

  static requireRole(allowedRoles: string[]) {
    return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
      if (!req.user) {
        res.status(401).json({
          success: false,
          error: 'Authentication required',
          code: 'AUTH_REQUIRED',
        });
        return;
      }

      if (!allowedRoles.includes(req.user.role)) {
        res.status(403).json({
          success: false,
          error: 'Insufficient permissions',
          code: 'AUTH_FORBIDDEN',
          requiredRoles: allowedRoles,
          userRole: req.user.role,
        });
        return;
      }

      next();
    };
  }

  static requirePermission(requiredPermissions: string[]) {
    return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
      if (!req.user) {
        res.status(401).json({
          success: false,
          error: 'Authentication required',
          code: 'AUTH_REQUIRED',
        });
        return;
      }

      const hasPermission = requiredPermissions.every((permission) =>
        req.user!.permissions.includes(permission)
      );

      if (!hasPermission) {
        res.status(403).json({
          success: false,
          error: 'Insufficient permissions',
          code: 'AUTH_FORBIDDEN',
          requiredPermissions,
          userPermissions: req.user.permissions,
        });
        return;
      }

      next();
    };
  }

  static async validateApiKey(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const apiKey = req.headers['x-api-key'] as string;

      if (!apiKey) {
        res.status(401).json({
          success: false,
          error: 'API key required',
          code: 'API_KEY_MISSING',
        });
        return;
      }

      // In production, validate against database
      const validApiKeys = process.env.VALID_API_KEYS?.split(',') || [];

      if (!validApiKeys.includes(apiKey)) {
        res.status(401).json({
          success: false,
          error: 'Invalid API key',
          code: 'API_KEY_INVALID',
        });
        return;
      }

      next();
    } catch (error) {
      console.error('API key validation error:', error);
      res.status(500).json({
        success: false,
        error: 'API key validation failed',
        code: 'API_KEY_ERROR',
      });
    }
  }
}
