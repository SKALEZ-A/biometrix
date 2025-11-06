import { Request, Response } from 'express';
import { AuthService } from '../services/auth.service';
import { logger } from '@shared/utils/logger';
import { validateLogin, validateRegister } from '../validators/auth.validators';

export class AuthController {
  private authService: AuthService;

  constructor() {
    this.authService = new AuthService();
  }

  async register(req: Request, res: Response): Promise<void> {
    try {
      const validation = validateRegister(req.body);
      if (!validation.valid) {
        res.status(400).json({ errors: validation.errors });
        return;
      }

      const result = await this.authService.register(req.body);
      logger.info('User registered', { email: req.body.email });
      
      res.status(201).json({
        success: true,
        data: result,
      });
    } catch (error: any) {
      logger.error('Registration error', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async login(req: Request, res: Response): Promise<void> {
    try {
      const validation = validateLogin(req.body);
      if (!validation.valid) {
        res.status(400).json({ errors: validation.errors });
        return;
      }

      const result = await this.authService.login(req.body.email, req.body.password);
      
      if (!result) {
        res.status(401).json({ error: 'Invalid credentials' });
        return;
      }

      logger.info('User logged in', { email: req.body.email });
      res.status(200).json({
        success: true,
        data: result,
      });
    } catch (error: any) {
      logger.error('Login error', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async logout(req: Request, res: Response): Promise<void> {
    try {
      const token = req.headers.authorization?.split(' ')[1];
      if (token) {
        await this.authService.logout(token);
      }

      res.status(200).json({
        success: true,
        message: 'Logged out successfully',
      });
    } catch (error: any) {
      logger.error('Logout error', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async refreshToken(req: Request, res: Response): Promise<void> {
    try {
      const { refreshToken } = req.body;
      
      if (!refreshToken) {
        res.status(400).json({ error: 'Refresh token required' });
        return;
      }

      const result = await this.authService.refreshToken(refreshToken);
      
      if (!result) {
        res.status(401).json({ error: 'Invalid refresh token' });
        return;
      }

      res.status(200).json({
        success: true,
        data: result,
      });
    } catch (error: any) {
      logger.error('Token refresh error', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async verifyEmail(req: Request, res: Response): Promise<void> {
    try {
      const { token } = req.body;
      
      if (!token) {
        res.status(400).json({ error: 'Verification token required' });
        return;
      }

      await this.authService.verifyEmail(token);
      
      res.status(200).json({
        success: true,
        message: 'Email verified successfully',
      });
    } catch (error: any) {
      logger.error('Email verification error', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async forgotPassword(req: Request, res: Response): Promise<void> {
    try {
      const { email } = req.body;
      
      if (!email) {
        res.status(400).json({ error: 'Email required' });
        return;
      }

      await this.authService.forgotPassword(email);
      
      res.status(200).json({
        success: true,
        message: 'Password reset email sent',
      });
    } catch (error: any) {
      logger.error('Forgot password error', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async resetPassword(req: Request, res: Response): Promise<void> {
    try {
      const { token, newPassword } = req.body;
      
      if (!token || !newPassword) {
        res.status(400).json({ error: 'Token and new password required' });
        return;
      }

      await this.authService.resetPassword(token, newPassword);
      
      res.status(200).json({
        success: true,
        message: 'Password reset successfully',
      });
    } catch (error: any) {
      logger.error('Password reset error', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async verify2FA(req: Request, res: Response): Promise<void> {
    try {
      const { userId, code } = req.body;
      
      if (!userId || !code) {
        res.status(400).json({ error: 'User ID and code required' });
        return;
      }

      const result = await this.authService.verify2FA(userId, code);
      
      if (!result) {
        res.status(401).json({ error: 'Invalid 2FA code' });
        return;
      }

      res.status(200).json({
        success: true,
        data: result,
      });
    } catch (error: any) {
      logger.error('2FA verification error', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async enable2FA(req: Request, res: Response): Promise<void> {
    try {
      const { userId } = req.body;
      
      if (!userId) {
        res.status(400).json({ error: 'User ID required' });
        return;
      }

      const result = await this.authService.enable2FA(userId);
      
      res.status(200).json({
        success: true,
        data: result,
      });
    } catch (error: any) {
      logger.error('Enable 2FA error', { error });
      res.status(500).json({ error: error.message });
    }
  }

  async disable2FA(req: Request, res: Response): Promise<void> {
    try {
      const { userId, code } = req.body;
      
      if (!userId || !code) {
        res.status(400).json({ error: 'User ID and code required' });
        return;
      }

      await this.authService.disable2FA(userId, code);
      
      res.status(200).json({
        success: true,
        message: '2FA disabled successfully',
      });
    } catch (error: any) {
      logger.error('Disable 2FA error', { error });
      res.status(500).json({ error: error.message });
    }
  }
}
