import { Router } from 'express';
import { voiceController } from '../controllers/voice.controller';
import { authMiddleware } from '../middleware/auth.middleware';

const router = Router();

router.post(
  '/enroll',
  authMiddleware.authenticate,
  voiceController.enrollVoiceprint
);

router.post(
  '/verify',
  authMiddleware.authenticate,
  voiceController.verifyVoice
);

router.get(
  '/:userId',
  authMiddleware.authenticate,
  voiceController.getVoiceprint
);

router.put(
  '/:userId',
  authMiddleware.authenticate,
  voiceController.updateVoiceprint
);

router.delete(
  '/:userId',
  authMiddleware.authenticate,
  voiceController.deleteVoiceprint
);

export default router;
