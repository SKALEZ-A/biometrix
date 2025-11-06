import { Router } from 'express';
import { webhookController } from '../controllers/webhook.controller';

const router = Router();

router.post('/', webhookController.registerWebhook);
router.get('/', webhookController.getWebhooks);
router.get('/:id', webhookController.getWebhook);
router.put('/:id', webhookController.updateWebhook);
router.delete('/:id', webhookController.deleteWebhook);
router.post('/:id/test', webhookController.testWebhook);

export default router;
