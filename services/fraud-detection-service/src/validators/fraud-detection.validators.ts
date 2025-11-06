import Joi from 'joi';

export const fraudDetectionValidationSchemas = {
  detectFraud: Joi.object({
    body: Joi.object({
      transactionId: Joi.string().uuid().required(),
      userId: Joi.string().uuid().required(),
      amount: Joi.number().positive().required(),
      currency: Joi.string().length(3).uppercase().required(),
      merchantId: Joi.string().uuid().required(),
      paymentMethod: Joi.string().valid('credit_card', 'debit_card', 'bank_transfer', 'digital_wallet').required(),
      deviceFingerprint: Joi.string().required(),
      ipAddress: Joi.string().ip().required(),
      location: Joi.object({
        country: Joi.string().required(),
        city: Joi.string().required(),
        latitude: Joi.number().min(-90).max(90).required(),
        longitude: Joi.number().min(-180).max(180).required()
      }).required(),
      biometricData: Joi.object({
        type: Joi.string().valid('face', 'fingerprint', 'voice', 'iris').required(),
        data: Joi.string().required(),
        confidence: Joi.number().min(0).max(100).optional()
      }).optional(),
      metadata: Joi.object().optional()
    })
  }),

  batchAnalysis: Joi.object({
    body: Joi.object({
      transactions: Joi.array().items(
        Joi.object({
          transactionId: Joi.string().uuid().required(),
          userId: Joi.string().uuid().required(),
          amount: Joi.number().positive().required(),
          timestamp: Joi.date().iso().required()
        })
      ).min(1).max(1000).required()
    })
  }),

  riskScore: Joi.object({
    body: Joi.object({
      transactionId: Joi.string().uuid().required(),
      features: Joi.object().optional()
    })
  }),

  patternAnalysis: Joi.object({
    body: Joi.object({
      userId: Joi.string().uuid().optional(),
      merchantId: Joi.string().uuid().optional(),
      startDate: Joi.date().iso().required(),
      endDate: Joi.date().iso().min(Joi.ref('startDate')).required(),
      patternType: Joi.string().valid('velocity', 'location', 'amount', 'device').optional()
    })
  })
};
