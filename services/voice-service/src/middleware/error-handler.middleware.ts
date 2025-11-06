import { Request, Response, NextFunction } from 'express';
import { logger } from '../../../packages/shared/src/utils/logger';

export class ErrorHandlerMiddleware {
  public handle(err: Error, req: Request, res: Response, next: NextFunction): void {
    logger.error('Unhandled error:', err);

    const statusCode = (err as any).statusCode || 500;
    const message = err.message || 'Internal Server Error';

    res.status(statusCode).json({
      error: err.name || 'Error',
      message,
      ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
    });
  }
}

export const errorHandler = new ErrorHandlerMiddleware();
