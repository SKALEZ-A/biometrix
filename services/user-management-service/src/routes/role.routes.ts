import { Router } from 'express';
import { RoleController } from '../controllers/role.controller';
import { requireRole, requirePermission } from '../middleware/auth.middleware';

const router = Router();
const roleController = new RoleController();

router.post('/', 
  requireRole('admin'),
  requirePermission('role:create'),
  roleController.createRole.bind(roleController)
);

router.get('/', 
  requirePermission('role:read'),
  roleController.listRoles.bind(roleController)
);

router.get('/:id', 
  requirePermission('role:read'),
  roleController.getRoleById.bind(roleController)
);

router.put('/:id', 
  requireRole('admin'),
  requirePermission('role:update'),
  roleController.updateRole.bind(roleController)
);

router.delete('/:id', 
  requireRole('admin'),
  requirePermission('role:delete'),
  roleController.deleteRole.bind(roleController)
);

router.post('/:id/permissions', 
  requireRole('admin'),
  roleController.addPermission.bind(roleController)
);

router.delete('/:id/permissions', 
  requireRole('admin'),
  roleController.removePermission.bind(roleController)
);

export { router as roleRoutes };
