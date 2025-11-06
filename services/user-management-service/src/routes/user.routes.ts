import { Router } from 'express';
import { userController } from '../controllers/user.controller';
import { authMiddleware } from '../middleware/auth.middleware';
import { validationMiddleware } from '../middleware/validation.middleware';
import { createUserSchema, updateUserSchema } from '../validators/user.validators';

const router = Router();

router.post(
  '/',
  authMiddleware.authenticate,
  authMiddleware.authorize(['user:create']),
  validationMiddleware(createUserSchema),
  userController.createUser
);

router.get(
  '/',
  authMiddleware.authenticate,
  authMiddleware.authorize(['user:read']),
  userController.getAllUsers
);

router.get(
  '/:id',
  authMiddleware.authenticate,
  authMiddleware.authorize(['user:read']),
  userController.getUserById
);

router.put(
  '/:id',
  authMiddleware.authenticate,
  authMiddleware.authorize(['user:update']),
  validationMiddleware(updateUserSchema),
  userController.updateUser
);

router.delete(
  '/:id',
  authMiddleware.authenticate,
  authMiddleware.authorize(['user:delete']),
  userController.deleteUser
);

router.post(
  '/:id/roles/:roleId',
  authMiddleware.authenticate,
  authMiddleware.authorize(['user:update']),
  userController.assignRole
);

router.delete(
  '/:id/roles/:roleId',
  authMiddleware.authenticate,
  authMiddleware.authorize(['user:update']),
  userController.removeRole
);

export default router;
