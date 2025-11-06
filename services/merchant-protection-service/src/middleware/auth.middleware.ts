import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'merchant-protection-secret-key';

export const authenticateMerchant = (req: Request, res: Response, next: NextFunction): void => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({
        success: false,
        error: { message: 'No token provided' }
      });
      return;
    }

    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, JWT_SECRET) as any;

    (req as any).merchant = {
      merchantId: decoded.merchantId,
      role: decoded.role
    };

    next();
  } catch (error) {
    res.status(401).json({
      success: false,
      error: { message: 'Invalid or expired token' }
    });
  }
};
