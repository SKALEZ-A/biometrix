import { User, IUser } from '../models/user.model';
import { UserRepository } from '../repositories/user.repository';
import { logger } from '@shared/utils/logger';
import * as bcrypt from 'bcrypt';
import * as crypto from 'crypto';

export class UserService {
  private userRepository: UserRepository;

  constructor() {
    this.userRepository = new UserRepository();
  }

  async createUser(userData: Partial<IUser>): Promise<User> {
    try {
      // Validate email uniqueness
      const existingUser = await this.userRepository.findByEmail(userData.email!);
      if (existingUser) {
        throw new Error('Email already exists');
      }

      // Hash password
      if (userData.passwordHash) {
        userData.passwordHash = await bcrypt.hash(userData.passwordHash, 12);
      }

      // Generate verification token
      userData.emailVerificationToken = crypto.randomBytes(32).toString('hex');

      const user = new User(userData);
      const createdUser = await this.userRepository.create(user);

      logger.info(`User created: ${createdUser.userId}`);
      return createdUser;
    } catch (error) {
      logger.error('Error creating user:', error);
      throw error;
    }
  }

  async getUserById(userId: string): Promise<User | null> {
    try {
      return await this.userRepository.findById(userId);
    } catch (error) {
      logger.error('Error getting user:', error);
      throw error;
    }
  }

  async getUserByEmail(email: string): Promise<User | null> {
    try {
      return await this.userRepository.findByEmail(email);
    } catch (error) {
      logger.error('Error getting user by email:', error);
      throw error;
    }
  }

  async updateUser(userId: string, updates: Partial<IUser>): Promise<User | null> {
    try {
      // Don't allow direct password updates through this method
      if (updates.passwordHash) {
        delete updates.passwordHash;
      }

      updates.updatedAt = new Date();
      const updatedUser = await this.userRepository.update(userId, updates);

      if (updatedUser) {
        logger.info(`User updated: ${userId}`);
      }

      return updatedUser;
    } catch (error) {
      logger.error('Error updating user:', error);
      throw error;
    }
  }

  async deleteUser(userId: string): Promise<boolean> {
    try {
      const result = await this.userRepository.delete(userId);
      if (result) {
        logger.info(`User deleted: ${userId}`);
      }
      return result;
    } catch (error) {
      logger.error('Error deleting user:', error);
      throw error;
    }
  }

  async verifyEmail(token: string): Promise<boolean> {
    try {
      const user = await this.userRepository.findByVerificationToken(token);
      if (!user) {
        return false;
      }

      await this.userRepository.update(user.userId, {
        isVerified: true,
        emailVerificationToken: undefined,
        updatedAt: new Date(),
      });

      logger.info(`Email verified for user: ${user.userId}`);
      return true;
    } catch (error) {
      logger.error('Error verifying email:', error);
      throw error;
    }
  }

  async changePassword(userId: string, currentPassword: string, newPassword: string): Promise<boolean> {
    try {
      const user = await this.userRepository.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Verify current password
      const isValid = await bcrypt.compare(currentPassword, user.passwordHash);
      if (!isValid) {
        throw new Error('Invalid current password');
      }

      // Hash new password
      const newPasswordHash = await bcrypt.hash(newPassword, 12);

      await this.userRepository.update(userId, {
        passwordHash: newPasswordHash,
        updatedAt: new Date(),
      });

      logger.info(`Password changed for user: ${userId}`);
      return true;
    } catch (error) {
      logger.error('Error changing password:', error);
      throw error;
    }
  }

  async requestPasswordReset(email: string): Promise<string | null> {
    try {
      const user = await this.userRepository.findByEmail(email);
      if (!user) {
        return null;
      }

      const resetToken = crypto.randomBytes(32).toString('hex');
      const resetExpires = new Date(Date.now() + 3600000); // 1 hour

      await this.userRepository.update(user.userId, {
        passwordResetToken: resetToken,
        passwordResetExpires: resetExpires,
        updatedAt: new Date(),
      });

      logger.info(`Password reset requested for user: ${user.userId}`);
      return resetToken;
    } catch (error) {
      logger.error('Error requesting password reset:', error);
      throw error;
    }
  }

  async resetPassword(token: string, newPassword: string): Promise<boolean> {
    try {
      const user = await this.userRepository.findByResetToken(token);
      if (!user || !user.passwordResetExpires || user.passwordResetExpires < new Date()) {
        return false;
      }

      const newPasswordHash = await bcrypt.hash(newPassword, 12);

      await this.userRepository.update(user.userId, {
        passwordHash: newPasswordHash,
        passwordResetToken: undefined,
        passwordResetExpires: undefined,
        loginAttempts: 0,
        lockUntil: undefined,
        updatedAt: new Date(),
      });

      logger.info(`Password reset for user: ${user.userId}`);
      return true;
    } catch (error) {
      logger.error('Error resetting password:', error);
      throw error;
    }
  }

  async enableTwoFactor(userId: string, secret: string): Promise<boolean> {
    try {
      await this.userRepository.update(userId, {
        twoFactorEnabled: true,
        twoFactorSecret: secret,
        updatedAt: new Date(),
      });

      logger.info(`Two-factor authentication enabled for user: ${userId}`);
      return true;
    } catch (error) {
      logger.error('Error enabling two-factor:', error);
      throw error;
    }
  }

  async disableTwoFactor(userId: string): Promise<boolean> {
    try {
      await this.userRepository.update(userId, {
        twoFactorEnabled: false,
        twoFactorSecret: undefined,
        updatedAt: new Date(),
      });

      logger.info(`Two-factor authentication disabled for user: ${userId}`);
      return true;
    } catch (error) {
      logger.error('Error disabling two-factor:', error);
      throw error;
    }
  }

  async listUsers(filters: any = {}, page: number = 1, limit: number = 20): Promise<{ users: User[]; total: number }> {
    try {
      return await this.userRepository.findAll(filters, page, limit);
    } catch (error) {
      logger.error('Error listing users:', error);
      throw error;
    }
  }

  async searchUsers(query: string, page: number = 1, limit: number = 20): Promise<{ users: User[]; total: number }> {
    try {
      return await this.userRepository.search(query, page, limit);
    } catch (error) {
      logger.error('Error searching users:', error);
      throw error;
    }
  }
}
