import { Request, Response, NextFunction } from 'express';
import Joi from 'joi';
import { logger } from '../../../packages/shared/src/utils/logger';

export interface ValidationSchema {
  body?: Joi.Schema;
  query?: Joi.Schema;
  params?: Joi.Schema;
  headers?: Joi.Schema;
}

export class RequestValidator {
  static validate(schema: ValidationSchema) {
    return (req: Request, res: Response, next: NextFunction) => {
      const errors: any[] = [];

      if (schema.body) {
        const { error } = schema.body.validate(req.body);
        if (error) {
          errors.push({ location: 'body', details: error.details });
        }
      }

      if (schema.query) {
        const { error } = schema.query.validate(req.query);
        if (error) {
          errors.push({ location: 'query', details: error.details });
        }
      }

      if (schema.params) {
        const { error } = schema.params.validate(req.params);
        if (error) {
          errors.push({ location: 'params', details: error.details });
        }
      }

      if (schema.headers) {
        const { error } = schema.headers.validate(req.headers);
        if (error) {
          errors.push({ location: 'headers', details: error.details });
        }
      }

      if (errors.length > 0) {
        logger.warn('Request validation failed', { errors, path: req.path });
        return res.status(400).json({
          success: false,
          message: 'Validation error',
          errors
        });
      }

      next();
    };
  }

  static sanitizeInput(req: Request, res: Response, next: NextFunction) {
    if (req.body) {
      req.body = RequestValidator.sanitizeObject(req.body);
    }

    if (req.query) {
      req.query = RequestValidator.sanitizeObject(req.query);
    }

    next();
  }

  private static sanitizeObject(obj: any): any {
    if (typeof obj !== 'object' || obj === null) {
      return obj;
    }

    const sanitized: any = Array.isArray(obj) ? [] : {};

    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        const value = obj[key];

        if (typeof value === 'string') {
          sanitized[key] = RequestValidator.sanitizeString(value);
        } else if (typeof value === 'object') {
          sanitized[key] = RequestValidator.sanitizeObject(value);
        } else {
          sanitized[key] = value;
        }
      }
    }

    return sanitized;
  }

  private static sanitizeString(str: string): string {
    return str
      .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
      .replace(/javascript:/gi, '')
      .replace(/on\w+\s*=/gi, '')
      .trim();
  }
}

export const commonSchemas = {
  pagination: Joi.object({
    page: Joi.number().integer().min(1).default(1),
    limit: Joi.number().integer().min(1).max(100).default(20)
  }),

  dateRange: Joi.object({
    startDate: Joi.date().iso(),
    endDate: Joi.date().iso().min(Joi.ref('startDate'))
  }),

  userId: Joi.string().uuid().required(),

  transactionId: Joi.string().uuid().required()
};
