import { Request, Response } from 'express';
import { RoleService } from '../services/role.service';
import { logger } from '@shared/utils/logger';
import { validateRole } from '../validators/role.validators';

export class RoleController {
  private roleService: RoleService;

  constructor() {
    this.roleService = new RoleService();
  }

  async createRole(req: Request, res: Response): Promise<void> {
    try {
      const validation = validateRole(req.body);
      if (!validation.valid) {
        res.status(400).json({ errors: validation.errors });
        return;
      }

      const role = await this.roleService.createRole(req.body);
      logger.info('Role created', { roleId: role.id });
      
      res.status(201).json({
        success: true,
        data: role,
      });
    } catch (error: any) {
      logger.error('Error creating role', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async getRoleById(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const role = await this.roleService.getRoleById(id);

      if (!role) {
        res.status(404).json({ error: 'Role not found' });
        return;
      }

      res.status(200).json({
        success: true,
        data: role,
      });
    } catch (error: any) {
      logger.error('Error fetching role', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async updateRole(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const role = await this.roleService.updateRole(id, req.body);

      if (!role) {
        res.status(404).json({ error: 'Role not found' });
        return;
      }

      logger.info('Role updated', { roleId: id });
      res.status(200).json({
        success: true,
        data: role,
      });
    } catch (error: any) {
      logger.error('Error updating role', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async deleteRole(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      await this.roleService.deleteRole(id);
      
      logger.info('Role deleted', { roleId: id });
      res.status(200).json({
        success: true,
        message: 'Role deleted successfully',
      });
    } catch (error: any) {
      logger.error('Error deleting role', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async listRoles(req: Request, res: Response): Promise<void> {
    try {
      const { page = 1, limit = 20, search } = req.query;
      
      const result = await this.roleService.listRoles({
        page: Number(page),
        limit: Number(limit),
        search: search as string,
      });

      res.status(200).json({
        success: true,
        data: result.roles,
        pagination: {
          page: result.page,
          limit: result.limit,
          total: result.total,
          totalPages: result.totalPages,
        },
      });
    } catch (error: any) {
      logger.error('Error listing roles', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async addPermission(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const { permissionId } = req.body;

      await this.roleService.addPermission(id, permissionId);
      
      logger.info('Permission added to role', { roleId: id, permissionId });
      res.status(200).json({
        success: true,
        message: 'Permission added successfully',
      });
    } catch (error: any) {
      logger.error('Error adding permission', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async removePermission(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const { permissionId } = req.body;

      await this.roleService.removePermission(id, permissionId);
      
      logger.info('Permission removed from role', { roleId: id, permissionId });
      res.status(200).json({
        success: true,
        message: 'Permission removed successfully',
      });
    } catch (error: any) {
      logger.error('Error removing permission', { error });
      res.status(500).json({ error: error.message });
    }
  }
}
