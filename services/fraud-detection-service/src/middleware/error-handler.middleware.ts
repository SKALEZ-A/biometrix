import { Request, Response, NextFunction } from 'express';
import { logger } from '@shared/utils/logger';

export class AppError extends Error {
  public statusCode: number;
  public isOperational: boolean;

  constructor(message: string, statusCode: number = 500, isOperational: boolean = true) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    Error.captureStackTrace(this, this.constructor);
  }
}

export const errorHandlerMiddleware = (
  error: Error | AppError,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  const statusCode = (error as AppError).statusCode || 500;
  const isOperational = (error as AppError).isOperational !== false;

  logger.error('Error occurred', {
    message: error.message,
    stack: error.stack,
    statusCode,
    path: req.path,
    method: req.method,
    isOperational,
  });

  if (!isOperational) {
    logger.error('Critical error - shutting down', error);
    process.exit(1);
  }

  res.status(statusCode).json({
    error: error.name || 'Error',
    message: error.message,
    ...(process.env.NODE_ENV === 'development' && { stack: error.stack }),
  });
};
