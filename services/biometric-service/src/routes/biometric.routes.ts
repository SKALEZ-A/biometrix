import { Router } from 'express';
import { BiometricController } from '../controllers/biometric.controller';
import { authMiddleware } from '../middleware/auth.middleware';
import { rateLimitMiddleware } from '../middleware/rate-limit.middleware';
import { validationMiddleware } from '../middleware/validation.middleware';
import { biometricValidationSchemas } from '../validators/biometric.validators';

const router = Router();
const biometricController = new BiometricController();

// Biometric enrollment routes
router.post(
  '/enroll/fingerprint',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 10, windowMs: 60000 }),
  validationMiddleware(biometricValidationSchemas.enrollFingerprint),
  biometricController.enrollFingerprint.bind(biometricController)
);

router.post(
  '/enroll/facial',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 10, windowMs: 60000 }),
  validationMiddleware(biometricValidationSchemas.enrollFacial),
  biometricController.enrollFacial.bind(biometricController)
);

router.post(
  '/enroll/iris',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 10, windowMs: 60000 }),
  validationMiddleware(biometricValidationSchemas.enrollIris),
  biometricController.enrollIris.bind(biometricController)
);

router.post(
  '/enroll/voice',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 10, windowMs: 60000 }),
  validationMiddleware(biometricValidationSchemas.enrollVoice),
  biometricController.enrollVoice.bind(biometricController)
);

// Biometric verification routes
router.post(
  '/verify/fingerprint',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 50, windowMs: 60000 }),
  validationMiddleware(biometricValidationSchemas.verifyFingerprint),
  biometricController.verifyFingerprint.bind(biometricController)
);

router.post(
  '/verify/facial',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 50, windowMs: 60000 }),
  validationMiddleware(biometricValidationSchemas.verifyFacial),
  biometricController.verifyFacial.bind(biometricController)
);

router.post(
  '/verify/iris',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 50, windowMs: 60000 }),
  validationMiddleware(biometricValidationSchemas.verifyIris),
  biometricController.verifyIris.bind(biometricController)
);

router.post(
  '/verify/voice',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 50, windowMs: 60000 }),
  validationMiddleware(biometricValidationSchemas.verifyVoice),
  biometricController.verifyVoice.bind(biometricController)
);

router.post(
  '/verify/behavioral',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 100, windowMs: 60000 }),
  validationMiddleware(biometricValidationSchemas.verifyBehavioral),
  biometricController.verifyBehavioral.bind(biometricController)
);

// Multi-factor biometric verification
router.post(
  '/verify/multi-factor',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 30, windowMs: 60000 }),
  validationMiddleware(biometricValidationSchemas.verifyMultiFactor),
  biometricController.verifyMultiFactor.bind(biometricController)
);

// Profile management routes
router.get(
  '/profile/:userId',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 100, windowMs: 60000 }),
  biometricController.getBiometricProfile.bind(biometricController)
);

router.put(
  '/profile/:userId',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 20, windowMs: 60000 }),
  validationMiddleware(biometricValidationSchemas.updateProfile),
  biometricController.updateBiometricProfile.bind(biometricController)
);

router.delete(
  '/profile/:userId',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 10, windowMs: 60000 }),
  biometricController.deleteBiometricProfile.bind(biometricController)
);

// Analytics and monitoring routes
router.get(
  '/analytics/verification-stats',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 50, windowMs: 60000 }),
  biometricController.getVerificationStats.bind(biometricController)
);

router.get(
  '/analytics/fraud-attempts',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 50, windowMs: 60000 }),
  biometricController.getFraudAttempts.bind(biometricController)
);

// Health check
router.get('/health', biometricController.healthCheck.bind(biometricController));

export default router;
