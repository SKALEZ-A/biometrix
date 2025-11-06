import { z } from 'zod';

export const createPermissionSchema = z.object({
  name: z.string().min(1, 'Permission name is required'),
  resource: z.string().min(1, 'Resource is required'),
  action: z.enum(['create', 'read', 'update', 'delete', 'execute']),
  description: z.string().optional()
});

export const updatePermissionSchema = z.object({
  name: z.string().min(1).optional(),
  resource: z.string().min(1).optional(),
  action: z.enum(['create', 'read', 'update', 'delete', 'execute']).optional(),
  description: z.string().optional()
});

export type CreatePermissionInput = z.infer<typeof createPermissionSchema>;
export type UpdatePermissionInput = z.infer<typeof updatePermissionSchema>;
