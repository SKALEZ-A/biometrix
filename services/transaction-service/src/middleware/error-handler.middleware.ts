import { Request, Response, NextFunction } from 'express';
import { logger } from '@shared/utils/logger';

export const errorHandlerMiddleware = (
  error: any,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  logger.error('Transaction service error', {
    error: error.message,
    stack: error.stack,
    path: req.path,
  });

  const statusCode = error.statusCode || 500;
  res.status(statusCode).json({
    error: error.name || 'Internal Server Error',
    message: error.message || 'An unexpected error occurred',
  });
};
