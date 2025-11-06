import Joi from 'joi';

export const alertValidationSchemas = {
  createAlert: Joi.object({
    body: Joi.object({
      type: Joi.string().valid('fraud', 'security', 'compliance', 'system').required(),
      severity: Joi.string().valid('low', 'medium', 'high', 'critical').required(),
      title: Joi.string().min(5).max(200).required(),
      description: Joi.string().min(10).max(2000).required(),
      source: Joi.string().required(),
      metadata: Joi.object().optional(),
      affectedEntities: Joi.array().items(Joi.string()).optional(),
      recommendedActions: Joi.array().items(Joi.string()).optional()
    })
  }),

  listAlerts: Joi.object({
    query: Joi.object({
      page: Joi.number().integer().min(1).default(1),
      limit: Joi.number().integer().min(1).max(100).default(20),
      status: Joi.string().valid('open', 'acknowledged', 'resolved', 'escalated').optional(),
      severity: Joi.string().valid('low', 'medium', 'high', 'critical').optional(),
      type: Joi.string().valid('fraud', 'security', 'compliance', 'system').optional(),
      startDate: Joi.date().iso().optional(),
      endDate: Joi.date().iso().min(Joi.ref('startDate')).optional()
    })
  }),

  getAlert: Joi.object({
    params: Joi.object({
      alertId: Joi.string().uuid().required()
    })
  }),

  updateAlert: Joi.object({
    params: Joi.object({
      alertId: Joi.string().uuid().required()
    }),
    body: Joi.object({
      status: Joi.string().valid('open', 'acknowledged', 'resolved', 'escalated').optional(),
      assignedTo: Joi.string().uuid().optional(),
      notes: Joi.string().max(1000).optional(),
      tags: Joi.array().items(Joi.string()).optional()
    })
  }),

  acknowledgeAlert: Joi.object({
    params: Joi.object({
      alertId: Joi.string().uuid().required()
    }),
    body: Joi.object({
      acknowledgedBy: Joi.string().uuid().required(),
      notes: Joi.string().max(500).optional()
    })
  }),

  resolveAlert: Joi.object({
    params: Joi.object({
      alertId: Joi.string().uuid().required()
    }),
    body: Joi.object({
      resolvedBy: Joi.string().uuid().required(),
      resolution: Joi.string().min(10).max(1000).required(),
      actionsTaken: Joi.array().items(Joi.string()).required()
    })
  }),

  escalateAlert: Joi.object({
    params: Joi.object({
      alertId: Joi.string().uuid().required()
    }),
    body: Joi.object({
      escalatedBy: Joi.string().uuid().required(),
      escalationReason: Joi.string().min(10).max(500).required(),
      escalatedTo: Joi.string().uuid().required()
    })
  })
};
