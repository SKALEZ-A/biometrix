import { Request, Response, NextFunction } from 'express';
import { verifyJWT } from '@shared/crypto/jwt';
import { logger } from '@shared/utils/logger';

export interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
    role: string;
  };
}

export const authMiddleware = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');

    if (!token) {
      res.status(401).json({ error: 'Unauthorized', message: 'No token provided' });
      return;
    }

    const decoded = await verifyJWT(token);

    if (!decoded) {
      res.status(401).json({ error: 'Unauthorized', message: 'Invalid token' });
      return;
    }

    req.user = {
      id: decoded.userId,
      email: decoded.email,
      role: decoded.role,
    };

    next();
  } catch (error) {
    logger.error('Auth middleware error', error);
    res.status(401).json({ error: 'Unauthorized' });
  }
};
