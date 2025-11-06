import { Request, Response, NextFunction } from 'express';
import Joi from 'joi';

const createTransactionSchema = Joi.object({
  userId: Joi.string().required(),
  amount: Joi.number().positive().required(),
  currency: Joi.string().length(3).uppercase().required(),
  merchantId: Joi.string().required(),
  paymentMethod: Joi.string().valid('card', 'bank_transfer', 'crypto', 'wallet').required(),
  deviceFingerprint: Joi.string().required(),
  ipAddress: Joi.string().ip().required(),
  location: Joi.object({
    latitude: Joi.number().min(-90).max(90).required(),
    longitude: Joi.number().min(-180).max(180).required(),
    country: Joi.string().length(2).uppercase().required(),
  }).optional(),
  metadata: Joi.object().optional(),
});

const flagTransactionSchema = Joi.object({
  reason: Joi.string().required(),
  severity: Joi.string().valid('low', 'medium', 'high', 'critical').required(),
  details: Joi.string().optional(),
});

export const validationMiddleware = {
  validateCreateTransaction: (req: Request, res: Response, next: NextFunction): void => {
    const { error } = createTransactionSchema.validate(req.body);
    if (error) {
      res.status(400).json({
        error: 'Validation Error',
        message: error.details[0].message,
      });
      return;
    }
    next();
  },

  validateFlagTransaction: (req: Request, res: Response, next: NextFunction): void => {
    const { error } = flagTransactionSchema.validate(req.body);
    if (error) {
      res.status(400).json({
        error: 'Validation Error',
        message: error.details[0].message,
      });
      return;
    }
    next();
  },
};
