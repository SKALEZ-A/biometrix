import { Request, Response, NextFunction } from 'express';
import { z, ZodSchema, ZodError } from 'zod';

export class ValidationMiddleware {
  static validate(schema: ZodSchema) {
    return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
      try {
        await schema.parseAsync({
          body: req.body,
          query: req.query,
          params: req.params,
        });
        next();
      } catch (error) {
        if (error instanceof ZodError) {
          const errors = error.errors.map((err) => ({
            field: err.path.join('.'),
            message: err.message,
            code: err.code,
          }));

          res.status(400).json({
            success: false,
            error: 'Validation failed',
            code: 'VALIDATION_ERROR',
            details: errors,
          });
          return;
        }

        res.status(500).json({
          success: false,
          error: 'Validation error',
          code: 'VALIDATION_INTERNAL_ERROR',
        });
      }
    };
  }
}

// Common validation schemas
export const BiometricEventSchema = z.object({
  body: z.object({
    userId: z.string().min(1, 'User ID is required'),
    sessionId: z.string().min(1, 'Session ID is required'),
    events: z.array(
      z.object({
        type: z.enum(['keystroke', 'mouse', 'touch', 'scroll', 'click']),
        timestamp: z.number().positive(),
        features: z.record(z.any()),
      })
    ).min(1, 'At least one event is required'),
    deviceInfo: z.object({
      userAgent: z.string(),
      platform: z.string(),
      screenResolution: z.string().optional(),
      timezone: z.string().optional(),
    }).optional(),
  }),
});

export const ProfileGenerationSchema = z.object({
  body: z.object({
    userId: z.string().min(1, 'User ID is required'),
    events: z.array(z.any()).min(100, 'Minimum 100 events required for profile generation'),
    profileType: z.enum(['keystroke', 'mouse', 'touch', 'combined']).optional(),
  }),
});

export const BiometricMatchSchema = z.object({
  body: z.object({
    userId: z.string().min(1, 'User ID is required'),
    sessionId: z.string().min(1, 'Session ID is required'),
    events: z.array(z.any()).min(10, 'Minimum 10 events required for matching'),
    matchType: z.enum(['keystroke', 'mouse', 'touch', 'combined']).optional(),
  }),
});
