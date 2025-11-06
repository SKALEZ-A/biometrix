import { Request, Response, NextFunction } from 'express';
import { verifyToken } from '@shared/crypto/jwt';
import { logger } from '@shared/utils/logger';

export async function authMiddleware(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');
    
    if (!token) {
      res.status(401).json({ error: 'No token provided' });
      return;
    }

    const decoded = await verifyToken(token);
    (req as any).user = decoded;
    next();
  } catch (error) {
    logger.error('Authentication failed', { error });
    res.status(401).json({ error: 'Invalid token' });
  }
}
