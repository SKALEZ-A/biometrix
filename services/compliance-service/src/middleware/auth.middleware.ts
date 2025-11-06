import { Request, Response, NextFunction } from 'express';
import { verifyJWT } from '@shared/crypto/jwt';

export const authMiddleware = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    const decoded = await verifyJWT(token);
    (req as any).user = decoded;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Authentication failed' });
  }
};
