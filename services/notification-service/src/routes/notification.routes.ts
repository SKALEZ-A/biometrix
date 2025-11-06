import { Router } from 'express';
import { NotificationController } from '../controllers/notification.controller';
import { authMiddleware } from '../middleware/auth.middleware';

const router = Router();
const notificationController = new NotificationController();

router.post('/send',
  authMiddleware,
  notificationController.sendNotification.bind(notificationController)
);

router.post('/send-bulk',
  authMiddleware,
  notificationController.sendBulkNotifications.bind(notificationController)
);

router.get('/',
  authMiddleware,
  notificationController.getNotifications.bind(notificationController)
);

router.get('/:id',
  authMiddleware,
  notificationController.getNotificationById.bind(notificationController)
);

router.put('/:id/read',
  authMiddleware,
  notificationController.markAsRead.bind(notificationController)
);

router.put('/read-all',
  authMiddleware,
  notificationController.markAllAsRead.bind(notificationController)
);

router.delete('/:id',
  authMiddleware,
  notificationController.deleteNotification.bind(notificationController)
);

router.get('/preferences/get',
  authMiddleware,
  notificationController.getPreferences.bind(notificationController)
);

router.put('/preferences/update',
  authMiddleware,
  notificationController.updatePreferences.bind(notificationController)
);

export default router;
