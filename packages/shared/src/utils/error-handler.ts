export class AppError extends Error {
  constructor(
    public message: string,
    public statusCode: number = 500,
    public code?: string,
    public details?: any
  ) {
    super(message);
    this.name = this.constructor.name;
    Error.captureStackTrace(this, this.constructor);
  }
}

export class ValidationError extends AppError {
  constructor(message: string, details?: any) {
    super(message, 400, 'VALIDATION_ERROR', details);
  }
}

export class AuthenticationError extends AppError {
  constructor(message: string = 'Authentication failed') {
    super(message, 401, 'AUTHENTICATION_ERROR');
  }
}

export class AuthorizationError extends AppError {
  constructor(message: string = 'Insufficient permissions') {
    super(message, 403, 'AUTHORIZATION_ERROR');
  }
}

export class NotFoundError extends AppError {
  constructor(resource: string) {
    super(`${resource} not found`, 404, 'NOT_FOUND');
  }
}

export class ConflictError extends AppError {
  constructor(message: string) {
    super(message, 409, 'CONFLICT');
  }
}

export class RateLimitError extends AppError {
  constructor(message: string = 'Rate limit exceeded') {
    super(message, 429, 'RATE_LIMIT_EXCEEDED');
  }
}

export class ServiceUnavailableError extends AppError {
  constructor(message: string = 'Service temporarily unavailable') {
    super(message, 503, 'SERVICE_UNAVAILABLE');
  }
}

export function handleError(error: Error): {
  message: string;
  statusCode: number;
  code?: string;
  details?: any;
} {
  if (error instanceof AppError) {
    return {
      message: error.message,
      statusCode: error.statusCode,
      code: error.code,
      details: error.details
    };
  }

  // Handle known error types
  if (error.name === 'ValidationError') {
    return {
      message: error.message,
      statusCode: 400,
      code: 'VALIDATION_ERROR'
    };
  }

  if (error.name === 'MongoError' || error.name === 'MongoServerError') {
    return {
      message: 'Database error occurred',
      statusCode: 500,
      code: 'DATABASE_ERROR'
    };
  }

  // Default error
  return {
    message: 'Internal server error',
    statusCode: 500,
    code: 'INTERNAL_ERROR'
  };
}

export function isOperationalError(error: Error): boolean {
  if (error instanceof AppError) {
    return error.statusCode < 500;
  }
  return false;
}
