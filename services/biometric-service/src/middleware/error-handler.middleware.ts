import { Request, Response, NextFunction } from 'express';

export class AppError extends Error {
  statusCode: number;
  code: string;
  isOperational: boolean;
  details?: any;

  constructor(message: string, statusCode: number, code: string, details?: any) {
    super(message);
    this.statusCode = statusCode;
    this.code = code;
    this.isOperational = true;
    this.details = details;
    Error.captureStackTrace(this, this.constructor);
  }
}

export class ErrorHandler {
  static handle(err: Error | AppError, req: Request, res: Response, next: NextFunction): void {
    if (err instanceof AppError) {
      res.status(err.statusCode).json({
        success: false,
        error: err.message,
        code: err.code,
        details: err.details,
        ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
      });
      return;
    }

    // Handle specific error types
    if (err.name === 'ValidationError') {
      res.status(400).json({
        success: false,
        error: 'Validation failed',
        code: 'VALIDATION_ERROR',
        details: err.message,
      });
      return;
    }

    if (err.name === 'UnauthorizedError') {
      res.status(401).json({
        success: false,
        error: 'Unauthorized',
        code: 'UNAUTHORIZED',
        details: err.message,
      });
      return;
    }

    if (err.name === 'MongoError' || err.name === 'MongoServerError') {
      res.status(500).json({
        success: false,
        error: 'Database error',
        code: 'DATABASE_ERROR',
        ...(process.env.NODE_ENV === 'development' && { details: err.message }),
      });
      return;
    }

    // Log unexpected errors
    console.error('Unexpected error:', err);

    // Generic error response
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      code: 'INTERNAL_ERROR',
      ...(process.env.NODE_ENV === 'development' && {
        details: err.message,
        stack: err.stack,
      }),
    });
  }

  static notFound(req: Request, res: Response): void {
    res.status(404).json({
      success: false,
      error: 'Resource not found',
      code: 'NOT_FOUND',
      path: req.path,
      method: req.method,
    });
  }

  static asyncHandler(fn: Function) {
    return (req: Request, res: Response, next: NextFunction) => {
      Promise.resolve(fn(req, res, next)).catch(next);
    };
  }
}

// Custom error classes
export class NotFoundError extends AppError {
  constructor(resource: string, id?: string) {
    super(
      `${resource}${id ? ` with id ${id}` : ''} not found`,
      404,
      'NOT_FOUND'
    );
  }
}

export class ValidationError extends AppError {
  constructor(message: string, details?: any) {
    super(message, 400, 'VALIDATION_ERROR', details);
  }
}

export class UnauthorizedError extends AppError {
  constructor(message: string = 'Unauthorized access') {
    super(message, 401, 'UNAUTHORIZED');
  }
}

export class ForbiddenError extends AppError {
  constructor(message: string = 'Access forbidden') {
    super(message, 403, 'FORBIDDEN');
  }
}

export class ConflictError extends AppError {
  constructor(message: string, details?: any) {
    super(message, 409, 'CONFLICT', details);
  }
}

export class InternalError extends AppError {
  constructor(message: string = 'Internal server error', details?: any) {
    super(message, 500, 'INTERNAL_ERROR', details);
  }
}
