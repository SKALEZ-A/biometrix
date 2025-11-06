import Joi from 'joi';

export const biometricValidationSchemas = {
  enrollFingerprint: Joi.object({
    userId: Joi.string().uuid().required(),
    fingerprintData: Joi.object({
      imageData: Joi.string().base64().required(),
      quality: Joi.number().min(0).max(100).required(),
      fingerPosition: Joi.string().valid(
        'RIGHT_THUMB', 'RIGHT_INDEX', 'RIGHT_MIDDLE', 'RIGHT_RING', 'RIGHT_LITTLE',
        'LEFT_THUMB', 'LEFT_INDEX', 'LEFT_MIDDLE', 'LEFT_RING', 'LEFT_LITTLE'
      ).required(),
      captureDevice: Joi.string().required(),
      timestamp: Joi.date().iso().required()
    }).required(),
    metadata: Joi.object({
      deviceId: Joi.string().required(),
      ipAddress: Joi.string().ip().required(),
      userAgent: Joi.string().required()
    }).optional()
  }),

  enrollFacial: Joi.object({
    userId: Joi.string().uuid().required(),
    facialData: Joi.object({
      imageData: Joi.string().base64().required(),
      quality: Joi.number().min(0).max(100).required(),
      faceDetected: Joi.boolean().required(),
      landmarks: Joi.array().items(Joi.object({
        x: Joi.number().required(),
        y: Joi.number().required(),
        type: Joi.string().required()
      })).min(68).required(),
      captureDevice: Joi.string().required(),
      timestamp: Joi.date().iso().required()
    }).required(),
    livenessCheck: Joi.object({
      passed: Joi.boolean().required(),
      confidence: Joi.number().min(0).max(1).required(),
      method: Joi.string().valid('BLINK', 'HEAD_MOVEMENT', 'CHALLENGE_RESPONSE').required()
    }).required(),
    metadata: Joi.object({
      deviceId: Joi.string().required(),
      ipAddress: Joi.string().ip().required(),
      userAgent: Joi.string().required()
    }).optional()
  }),

  enrollIris: Joi.object({
    userId: Joi.string().uuid().required(),
    irisData: Joi.object({
      imageData: Joi.string().base64().required(),
      quality: Joi.number().min(0).max(100).required(),
      eye: Joi.string().valid('LEFT', 'RIGHT', 'BOTH').required(),
      pupilDiameter: Joi.number().positive().required(),
      captureDevice: Joi.string().required(),
      timestamp: Joi.date().iso().required()
    }).required(),
    metadata: Joi.object({
      deviceId: Joi.string().required(),
      ipAddress: Joi.string().ip().required(),
      userAgent: Joi.string().required()
    }).optional()
  }),

  enrollVoice: Joi.object({
    userId: Joi.string().uuid().required(),
    voiceData: Joi.object({
      audioData: Joi.string().base64().required(),
      duration: Joi.number().positive().required(),
      sampleRate: Joi.number().valid(8000, 16000, 44100, 48000).required(),
      format: Joi.string().valid('WAV', 'MP3', 'FLAC').required(),
      phrase: Joi.string().required(),
      captureDevice: Joi.string().required(),
      timestamp: Joi.date().iso().required()
    }).required(),
    metadata: Joi.object({
      deviceId: Joi.string().required(),
      ipAddress: Joi.string().ip().required(),
      userAgent: Joi.string().required()
    }).optional()
  }),

  verifyFingerprint: Joi.object({
    userId: Joi.string().uuid().required(),
    fingerprintData: Joi.object({
      imageData: Joi.string().base64().required(),
      quality: Joi.number().min(0).max(100).required(),
      fingerPosition: Joi.string().valid(
        'RIGHT_THUMB', 'RIGHT_INDEX', 'RIGHT_MIDDLE', 'RIGHT_RING', 'RIGHT_LITTLE',
        'LEFT_THUMB', 'LEFT_INDEX', 'LEFT_MIDDLE', 'LEFT_RING', 'LEFT_LITTLE'
      ).required(),
      captureDevice: Joi.string().required(),
      timestamp: Joi.date().iso().required()
    }).required(),
    threshold: Joi.number().min(0).max(1).optional().default(0.85),
    metadata: Joi.object({
      deviceId: Joi.string().required(),
      ipAddress: Joi.string().ip().required(),
      userAgent: Joi.string().required(),
      sessionId: Joi.string().uuid().optional()
    }).optional()
  }),

  verifyFacial: Joi.object({
    userId: Joi.string().uuid().required(),
    facialData: Joi.object({
      imageData: Joi.string().base64().required(),
      quality: Joi.number().min(0).max(100).required(),
      faceDetected: Joi.boolean().required(),
      landmarks: Joi.array().items(Joi.object({
        x: Joi.number().required(),
        y: Joi.number().required(),
        type: Joi.string().required()
      })).min(68).required(),
      captureDevice: Joi.string().required(),
      timestamp: Joi.date().iso().required()
    }).required(),
    livenessCheck: Joi.object({
      passed: Joi.boolean().required(),
      confidence: Joi.number().min(0).max(1).required(),
      method: Joi.string().valid('BLINK', 'HEAD_MOVEMENT', 'CHALLENGE_RESPONSE').required()
    }).required(),
    threshold: Joi.number().min(0).max(1).optional().default(0.90),
    metadata: Joi.object({
      deviceId: Joi.string().required(),
      ipAddress: Joi.string().ip().required(),
      userAgent: Joi.string().required(),
      sessionId: Joi.string().uuid().optional()
    }).optional()
  }),

  verifyIris: Joi.object({
    userId: Joi.string().uuid().required(),
    irisData: Joi.object({
      imageData: Joi.string().base64().required(),
      quality: Joi.number().min(0).max(100).required(),
      eye: Joi.string().valid('LEFT', 'RIGHT', 'BOTH').required(),
      pupilDiameter: Joi.number().positive().required(),
      captureDevice: Joi.string().required(),
      timestamp: Joi.date().iso().required()
    }).required(),
    threshold: Joi.number().min(0).max(1).optional().default(0.95),
    metadata: Joi.object({
      deviceId: Joi.string().required(),
      ipAddress: Joi.string().ip().required(),
      userAgent: Joi.string().required(),
      sessionId: Joi.string().uuid().optional()
    }).optional()
  }),

  verifyVoice: Joi.object({
    userId: Joi.string().uuid().required(),
    voiceData: Joi.object({
      audioData: Joi.string().base64().required(),
      duration: Joi.number().positive().required(),
      sampleRate: Joi.number().valid(8000, 16000, 44100, 48000).required(),
      format: Joi.string().valid('WAV', 'MP3', 'FLAC').required(),
      phrase: Joi.string().required(),
      captureDevice: Joi.string().required(),
      timestamp: Joi.date().iso().required()
    }).required(),
    threshold: Joi.number().min(0).max(1).optional().default(0.88),
    metadata: Joi.object({
      deviceId: Joi.string().required(),
      ipAddress: Joi.string().ip().required(),
      userAgent: Joi.string().required(),
      sessionId: Joi.string().uuid().optional()
    }).optional()
  }),

  verifyBehavioral: Joi.object({
    userId: Joi.string().uuid().required(),
    behavioralData: Joi.object({
      keystrokeDynamics: Joi.object({
        keyPressTimings: Joi.array().items(Joi.object({
          key: Joi.string().required(),
          pressTime: Joi.number().required(),
          releaseTime: Joi.number().required(),
          duration: Joi.number().required()
        })).min(10).required(),
        typingSpeed: Joi.number().positive().required(),
        errorRate: Joi.number().min(0).max(1).required()
      }).optional(),
      mouseMovements: Joi.object({
        movements: Joi.array().items(Joi.object({
          x: Joi.number().required(),
          y: Joi.number().required(),
          timestamp: Joi.number().required(),
          velocity: Joi.number().required()
        })).min(20).required(),
        clickPatterns: Joi.array().items(Joi.object({
          x: Joi.number().required(),
          y: Joi.number().required(),
          timestamp: Joi.number().required(),
          button: Joi.string().valid('LEFT', 'RIGHT', 'MIDDLE').required()
        })).optional()
      }).optional(),
      touchGestures: Joi.object({
        swipes: Joi.array().items(Joi.object({
          startX: Joi.number().required(),
          startY: Joi.number().required(),
          endX: Joi.number().required(),
          endY: Joi.number().required(),
          duration: Joi.number().required(),
          velocity: Joi.number().required()
        })).optional(),
        taps: Joi.array().items(Joi.object({
          x: Joi.number().required(),
          y: Joi.number().required(),
          pressure: Joi.number().min(0).max(1).required(),
          duration: Joi.number().required()
        })).optional()
      }).optional()
    }).required(),
    threshold: Joi.number().min(0).max(1).optional().default(0.75),
    metadata: Joi.object({
      deviceId: Joi.string().required(),
      ipAddress: Joi.string().ip().required(),
      userAgent: Joi.string().required(),
      sessionId: Joi.string().uuid().optional()
    }).optional()
  }),

  verifyMultiFactor: Joi.object({
    userId: Joi.string().uuid().required(),
    factors: Joi.array().items(Joi.object({
      type: Joi.string().valid('FINGERPRINT', 'FACIAL', 'IRIS', 'VOICE', 'BEHAVIORAL').required(),
      data: Joi.object().required(),
      weight: Joi.number().min(0).max(1).optional().default(1)
    })).min(2).max(5).required(),
    combinationStrategy: Joi.string().valid('AND', 'OR', 'WEIGHTED_AVERAGE', 'THRESHOLD').optional().default('WEIGHTED_AVERAGE'),
    overallThreshold: Joi.number().min(0).max(1).optional().default(0.85),
    metadata: Joi.object({
      deviceId: Joi.string().required(),
      ipAddress: Joi.string().ip().required(),
      userAgent: Joi.string().required(),
      sessionId: Joi.string().uuid().optional()
    }).optional()
  }),

  updateProfile: Joi.object({
    enabledBiometrics: Joi.array().items(
      Joi.string().valid('FINGERPRINT', 'FACIAL', 'IRIS', 'VOICE', 'BEHAVIORAL')
    ).optional(),
    preferences: Joi.object({
      defaultVerificationMethod: Joi.string().valid('FINGERPRINT', 'FACIAL', 'IRIS', 'VOICE', 'BEHAVIORAL').optional(),
      multiFactor: Joi.boolean().optional(),
      livenessDetection: Joi.boolean().optional(),
      adaptiveSecurity: Joi.boolean().optional()
    }).optional(),
    securitySettings: Joi.object({
      maxFailedAttempts: Joi.number().integer().min(1).max(10).optional(),
      lockoutDuration: Joi.number().integer().min(60).max(86400).optional(),
      requireReEnrollment: Joi.boolean().optional()
    }).optional()
  })
};
