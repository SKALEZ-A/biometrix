import { Request, Response } from 'express';
import { UserService } from '../services/user.service';
import { logger } from '@shared/utils/logger';

export class UserController {
  private userService: UserService;

  constructor() {
    this.userService = new UserService();
  }

  async createUser(req: Request, res: Response): Promise<void> {
    try {
      const user = await this.userService.createUser(req.body);
      res.status(201).json({ success: true, data: user.toJSON() });
    } catch (error) {
      logger.error('Error in createUser:', error);
      res.status(500).json({ success: false, error: (error as Error).message });
    }
  }

  async getUser(req: Request, res: Response): Promise<void> {
    try {
      const { userId } = req.params;
      const user = await this.userService.getUserById(userId);
      
      if (!user) {
        res.status(404).json({ success: false, error: 'User not found' });
        return;
      }

      res.status(200).json({ success: true, data: user.toJSON() });
    } catch (error) {
      logger.error('Error in getUser:', error);
      res.status(500).json({ success: false, error: (error as Error).message });
    }
  }

  async updateUser(req: Request, res: Response): Promise<void> {
    try {
      const { userId } = req.params;
      const user = await this.userService.updateUser(userId, req.body);
      
      if (!user) {
        res.status(404).json({ success: false, error: 'User not found' });
        return;
      }

      res.status(200).json({ success: true, data: user.toJSON() });
    } catch (error) {
      logger.error('Error in updateUser:', error);
      res.status(500).json({ success: false, error: (error as Error).message });
    }
  }

  async deleteUser(req: Request, res: Response): Promise<void> {
    try {
      const { userId } = req.params;
      const result = await this.userService.deleteUser(userId);
      
      if (!result) {
        res.status(404).json({ success: false, error: 'User not found' });
        return;
      }

      res.status(200).json({ success: true, message: 'User deleted successfully' });
    } catch (error) {
      logger.error('Error in deleteUser:', error);
      res.status(500).json({ success: false, error: (error as Error).message });
    }
  }

  async listUsers(req: Request, res: Response): Promise<void> {
    try {
      const { page = 1, limit = 20, ...filters } = req.query;
      const result = await this.userService.listUsers(filters, Number(page), Number(limit));
      
      res.status(200).json({
        success: true,
        data: result.users.map(u => u.toJSON()),
        pagination: {
          page: Number(page),
          limit: Number(limit),
          total: result.total,
        },
      });
    } catch (error) {
      logger.error('Error in listUsers:', error);
      res.status(500).json({ success: false, error: (error as Error).message });
    }
  }

  async searchUsers(req: Request, res: Response): Promise<void> {
    try {
      const { q, page = 1, limit = 20 } = req.query;
      const result = await this.userService.searchUsers(String(q), Number(page), Number(limit));
      
      res.status(200).json({
        success: true,
        data: result.users.map(u => u.toJSON()),
        pagination: {
          page: Number(page),
          limit: Number(limit),
          total: result.total,
        },
      });
    } catch (error) {
      logger.error('Error in searchUsers:', error);
      res.status(500).json({ success: false, error: (error as Error).message });
    }
  }
}
