import { Router } from 'express';
import { PermissionController } from '../controllers/permission.controller';
import { requireRole, requirePermission } from '../middleware/auth.middleware';

const router = Router();
const permissionController = new PermissionController();

router.post('/', 
  requireRole('admin'),
  requirePermission('permission:create'),
  permissionController.createPermission.bind(permissionController)
);

router.get('/', 
  requirePermission('permission:read'),
  permissionController.listPermissions.bind(permissionController)
);

router.get('/:id', 
  requirePermission('permission:read'),
  permissionController.getPermissionById.bind(permissionController)
);

router.put('/:id', 
  requireRole('admin'),
  requirePermission('permission:update'),
  permissionController.updatePermission.bind(permissionController)
);

router.delete('/:id', 
  requireRole('admin'),
  requirePermission('permission:delete'),
  permissionController.deletePermission.bind(permissionController)
);

export { router as permissionRoutes };
