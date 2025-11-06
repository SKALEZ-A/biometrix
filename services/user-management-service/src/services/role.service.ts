import { roleRepository } from '../repositories/role.repository';
import { permissionRepository } from '../repositories/permission.repository';
import { Role } from '../models/role.model';

export class RoleService {
  public async createRole(data: {
    name: string;
    description?: string;
    permissions?: string[];
  }): Promise<Role> {
    const existing = await roleRepository.findByName(data.name);
    if (existing) {
      throw new Error('Role with this name already exists');
    }

    if (data.permissions) {
      for (const permId of data.permissions) {
        const permission = await permissionRepository.findById(permId);
        if (!permission) {
          throw new Error(`Permission ${permId} not found`);
        }
      }
    }

    return await roleRepository.create({
      name: data.name,
      description: data.description,
      permissions: data.permissions || []
    });
  }

  public async getRoleById(id: string): Promise<Role | null> {
    return await roleRepository.findById(id);
  }

  public async getAllRoles(): Promise<Role[]> {
    return await roleRepository.findAll();
  }

  public async updateRole(id: string, updates: Partial<Role>): Promise<Role | null> {
    const role = await roleRepository.findById(id);
    if (!role) {
      throw new Error('Role not found');
    }

    if (updates.name && updates.name !== role.name) {
      const existing = await roleRepository.findByName(updates.name);
      if (existing) {
        throw new Error('Role with this name already exists');
      }
    }

    return await roleRepository.update(id, updates);
  }

  public async deleteRole(id: string): Promise<boolean> {
    return await roleRepository.delete(id);
  }

  public async addPermissionToRole(roleId: string, permissionId: string): Promise<Role | null> {
    const permission = await permissionRepository.findById(permissionId);
    if (!permission) {
      throw new Error('Permission not found');
    }

    return await roleRepository.addPermission(roleId, permissionId);
  }

  public async removePermissionFromRole(roleId: string, permissionId: string): Promise<Role | null> {
    return await roleRepository.removePermission(roleId, permissionId);
  }

  public async getRolePermissions(roleId: string): Promise<string[]> {
    const role = await roleRepository.findById(roleId);
    return role ? role.permissions : [];
  }
}

export const roleService = new RoleService();
