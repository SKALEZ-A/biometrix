import { Request, Response, NextFunction } from 'express';
import { logger } from '@shared/utils/logger';

export const errorHandlerMiddleware = (error: any, req: Request, res: Response, next: NextFunction): void => {
  logger.error('Alert service error', error);
  res.status(error.statusCode || 500).json({
    error: error.message || 'Internal server error',
  });
};
