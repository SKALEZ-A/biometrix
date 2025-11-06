import { z } from 'zod';

export const createRoleSchema = z.object({
  name: z.string().min(1, 'Role name is required'),
  description: z.string().optional(),
  permissions: z.array(z.string()).optional()
});

export const updateRoleSchema = z.object({
  name: z.string().min(1).optional(),
  description: z.string().optional(),
  permissions: z.array(z.string()).optional()
});

export const assignPermissionSchema = z.object({
  permissionId: z.string().min(1, 'Permission ID is required')
});

export type CreateRoleInput = z.infer<typeof createRoleSchema>;
export type UpdateRoleInput = z.infer<typeof updateRoleSchema>;
export type AssignPermissionInput = z.infer<typeof assignPermissionSchema>;
