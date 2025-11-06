import { Request, Response, NextFunction } from 'express';
import Joi from 'joi';

const merchantRiskScoreSchema = Joi.object({
  merchantId: Joi.string().required(),
  transactionData: Joi.object().required(),
});

const chargebackPreventionSchema = Joi.object({
  transactionId: Joi.string().required(),
  merchantId: Joi.string().required(),
  amount: Joi.number().positive().required(),
});

export const validationMiddleware = {
  validateMerchantRiskScore: (req: Request, res: Response, next: NextFunction): void => {
    const { error } = merchantRiskScoreSchema.validate(req.body);
    if (error) {
      res.status(400).json({ error: 'Validation Error', message: error.details[0].message });
      return;
    }
    next();
  },

  validateChargebackPrevention: (req: Request, res: Response, next: NextFunction): void => {
    const { error } = chargebackPreventionSchema.validate(req.body);
    if (error) {
      res.status(400).json({ error: 'Validation Error', message: error.details[0].message });
      return;
    }
    next();
  },
};
