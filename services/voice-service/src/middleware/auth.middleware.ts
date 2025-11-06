import { Request, Response, NextFunction } from 'express';
import { verify } from 'jsonwebtoken';
import { voiceConfig } from '../config/app.config';

class AuthMiddleware {
  public authenticate(req: Request, res: Response, next: NextFunction): void {
    try {
      const token = req.headers.authorization?.split(' ')[1];

      if (!token) {
        res.status(401).json({
          error: 'Unauthorized',
          message: 'No token provided'
        });
        return;
      }

      const decoded = verify(token, process.env.JWT_SECRET || 'secret');
      (req as any).user = decoded;
      next();
    } catch (error) {
      res.status(401).json({
        error: 'Unauthorized',
        message: 'Invalid token'
      });
    }
  }
}

export const authMiddleware = new AuthMiddleware();
