import { Request, Response } from 'express';
import { PermissionService } from '../services/permission.service';
import { logger } from '@shared/utils/logger';
import { validatePermission } from '../validators/permission.validators';

export class PermissionController {
  private permissionService: PermissionService;

  constructor() {
    this.permissionService = new PermissionService();
  }

  async createPermission(req: Request, res: Response): Promise<void> {
    try {
      const validation = validatePermission(req.body);
      if (!validation.valid) {
        res.status(400).json({ errors: validation.errors });
        return;
      }

      const permission = await this.permissionService.createPermission(req.body);
      logger.info('Permission created', { permissionId: permission.id });
      
      res.status(201).json({
        success: true,
        data: permission,
      });
    } catch (error: any) {
      logger.error('Error creating permission', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async getPermissionById(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const permission = await this.permissionService.getPermissionById(id);

      if (!permission) {
        res.status(404).json({ error: 'Permission not found' });
        return;
      }

      res.status(200).json({
        success: true,
        data: permission,
      });
    } catch (error: any) {
      logger.error('Error fetching permission', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async updatePermission(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const permission = await this.permissionService.updatePermission(id, req.body);

      if (!permission) {
        res.status(404).json({ error: 'Permission not found' });
        return;
      }

      logger.info('Permission updated', { permissionId: id });
      res.status(200).json({
        success: true,
        data: permission,
      });
    } catch (error: any) {
      logger.error('Error updating permission', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async deletePermission(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      await this.permissionService.deletePermission(id);
      
      logger.info('Permission deleted', { permissionId: id });
      res.status(200).json({
        success: true,
        message: 'Permission deleted successfully',
      });
    } catch (error: any) {
      logger.error('Error deleting permission', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async listPermissions(req: Request, res: Response): Promise<void> {
    try {
      const { page = 1, limit = 50, search, category } = req.query;
      
      const result = await this.permissionService.listPermissions({
        page: Number(page),
        limit: Number(limit),
        search: search as string,
        category: category as string,
      });

      res.status(200).json({
        success: true,
        data: result.permissions,
        pagination: {
          page: result.page,
          limit: result.limit,
          total: result.total,
          totalPages: result.totalPages,
        },
      });
    } catch (error: any) {
      logger.error('Error listing permissions', { error });
      res.status(500).json({ error: error.message });
    }
  }
}
