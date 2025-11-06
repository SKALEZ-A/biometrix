import { Request, Response, NextFunction } from 'express';
import Joi from 'joi';

const transactionSchema = Joi.object({
  transactionId: Joi.string().required(),
  amount: Joi.number().positive().required(),
  merchantId: Joi.string().required(),
  userId: Joi.string().required(),
  timestamp: Joi.date().iso().required(),
  location: Joi.object({
    lat: Joi.number().min(-90).max(90).required(),
    lon: Joi.number().min(-180).max(180).required()
  }).optional(),
  deviceFingerprint: Joi.string().optional(),
  ipAddress: Joi.string().ip().optional(),
  paymentMethod: Joi.string().valid('credit_card', 'debit_card', 'bank_transfer', 'crypto', 'wallet').optional()
});

const batchTransactionSchema = Joi.object({
  transactions: Joi.array().items(transactionSchema).min(1).max(100).required()
});

export const validateTransaction = (req: Request, res: Response, next: NextFunction): void => {
  const { error } = transactionSchema.validate(req.body);
  
  if (error) {
    res.status(400).json({
      success: false,
      error: {
        message: 'Validation error',
        details: error.details.map(d => d.message)
      }
    });
    return;
  }
  
  next();
};

export const validateBatchTransactions = (req: Request, res: Response, next: NextFunction): void => {
  const { error } = batchTransactionSchema.validate(req.body);
  
  if (error) {
    res.status(400).json({
      success: false,
      error: {
        message: 'Validation error',
        details: error.details.map(d => d.message)
      }
    });
    return;
  }
  
  next();
};
