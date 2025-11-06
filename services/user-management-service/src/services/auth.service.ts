import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import speakeasy from 'speakeasy';
import QRCode from 'qrcode';
import { v4 as uuidv4 } from 'uuid';
import { UserRepository } from '../repositories/user.repository';
import { logger } from '@shared/utils/logger';
import { sendEmail } from '../utils/email.utils';
import { config } from '../config/app.config';

export interface RegisterDTO {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  phone?: string;
}

export class AuthService {
  private userRepository: UserRepository;

  constructor() {
    this.userRepository = new UserRepository();
  }

  async register(data: RegisterDTO): Promise<any> {
    const existingUser = await this.userRepository.findByEmail(data.email);
    if (existingUser) {
      throw new Error('User already exists');
    }

    const hashedPassword = await bcrypt.hash(data.password, 10);
    const verificationToken = uuidv4();

    const user = {
      id: uuidv4(),
      email: data.email,
      password: hashedPassword,
      firstName: data.firstName,
      lastName: data.lastName,
      phone: data.phone,
      status: 'pending',
      emailVerified: false,
      verificationToken,
      twoFactorEnabled: false,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    await this.userRepository.create(user);

    // Send verification email
    await sendEmail({
      to: user.email,
      subject: 'Verify Your Email',
      template: 'email-verification',
      data: {
        firstName: user.firstName,
        verificationUrl: `${process.env.APP_URL}/verify-email?token=${verificationToken}`,
      },
    });

    const { password, ...userWithoutPassword } = user;
    return userWithoutPassword;
  }

  async login(email: string, password: string): Promise<any> {
    const user = await this.userRepository.findByEmail(email);
    
    if (!user) {
      logger.warn('Login attempt for non-existent user', { email });
      return null;
    }

    if (user.status !== 'active') {
      throw new Error('Account is not active');
    }

    const isValidPassword = await bcrypt.compare(password, user.password);
    if (!isValidPassword) {
      logger.warn('Invalid password attempt', { email });
      return null;
    }

    if (user.twoFactorEnabled) {
      // Return temporary token for 2FA verification
      const tempToken = jwt.sign(
        { userId: user.id, requires2FA: true },
        config.jwt.secret,
        { expiresIn: '5m' }
      );

      return {
        requires2FA: true,
        tempToken,
      };
    }

    const accessToken = this.generateAccessToken(user);
    const refreshToken = this.generateRefreshToken(user);

    // Store refresh token
    await this.userRepository.update(user.id, {
      refreshToken,
      lastLogin: new Date(),
    });

    const { password: _, ...userWithoutPassword } = user;

    return {
      user: userWithoutPassword,
      accessToken,
      refreshToken,
    };
  }

  async logout(token: string): Promise<void> {
    try {
      const decoded: any = jwt.verify(token, config.jwt.secret);
      await this.userRepository.update(decoded.id, {
        refreshToken: null,
      });
    } catch (error) {
      logger.error('Logout error', { error });
    }
  }

  async refreshToken(refreshToken: string): Promise<any> {
    try {
      const decoded: any = jwt.verify(refreshToken, config.jwt.secret);
      const user = await this.userRepository.findById(decoded.id);

      if (!user || user.refreshToken !== refreshToken) {
        return null;
      }

      const newAccessToken = this.generateAccessToken(user);
      const newRefreshToken = this.generateRefreshToken(user);

      await this.userRepository.update(user.id, {
        refreshToken: newRefreshToken,
      });

      return {
        accessToken: newAccessToken,
        refreshToken: newRefreshToken,
      };
    } catch (error) {
      logger.error('Token refresh error', { error });
      return null;
    }
  }

  async verifyEmail(token: string): Promise<void> {
    const user = await this.userRepository.collection.findOne({
      verificationToken: token,
    });

    if (!user) {
      throw new Error('Invalid verification token');
    }

    await this.userRepository.update(user.id, {
      emailVerified: true,
      status: 'active',
      verificationToken: null,
    });

    await sendEmail({
      to: user.email,
      subject: 'Email Verified Successfully',
      template: 'email-verified',
      data: { firstName: user.firstName },
    });
  }

  async forgotPassword(email: string): Promise<void> {
    const user = await this.userRepository.findByEmail(email);
    
    if (!user) {
      // Don't reveal if user exists
      logger.warn('Password reset for non-existent email', { email });
      return;
    }

    const resetToken = uuidv4();
    const resetExpiry = new Date(Date.now() + 3600000); // 1 hour

    await this.userRepository.update(user.id, {
      resetToken,
      resetExpiry,
    });

    await sendEmail({
      to: email,
      subject: 'Password Reset Request',
      template: 'password-reset',
      data: {
        firstName: user.firstName,
        resetUrl: `${process.env.APP_URL}/reset-password?token=${resetToken}`,
      },
    });
  }

  async resetPassword(token: string, newPassword: string): Promise<void> {
    const user = await this.userRepository.collection.findOne({
      resetToken: token,
      resetExpiry: { $gt: new Date() },
    });

    if (!user) {
      throw new Error('Invalid or expired reset token');
    }

    const hashedPassword = await bcrypt.hash(newPassword, 10);

    await this.userRepository.update(user.id, {
      password: hashedPassword,
      resetToken: null,
      resetExpiry: null,
    });

    await sendEmail({
      to: user.email,
      subject: 'Password Reset Successful',
      template: 'password-reset-success',
      data: { firstName: user.firstName },
    });
  }

  async enable2FA(userId: string): Promise<any> {
    const user = await this.userRepository.findById(userId);
    
    if (!user) {
      throw new Error('User not found');
    }

    const secret = speakeasy.generateSecret({
      name: `FraudPrevention (${user.email})`,
      length: 32,
    });

    await this.userRepository.update(userId, {
      twoFactorSecret: secret.base32,
    });

    const qrCodeUrl = await QRCode.toDataURL(secret.otpauth_url!);

    return {
      secret: secret.base32,
      qrCode: qrCodeUrl,
    };
  }

  async verify2FA(userId: string, code: string): Promise<any> {
    const user = await this.userRepository.findById(userId);
    
    if (!user || !user.twoFactorSecret) {
      throw new Error('2FA not enabled');
    }

    const verified = speakeasy.totp.verify({
      secret: user.twoFactorSecret,
      encoding: 'base32',
      token: code,
      window: 2,
    });

    if (!verified) {
      return null;
    }

    // Enable 2FA if this is first verification
    if (!user.twoFactorEnabled) {
      await this.userRepository.update(userId, {
        twoFactorEnabled: true,
      });
    }

    const accessToken = this.generateAccessToken(user);
    const refreshToken = this.generateRefreshToken(user);

    await this.userRepository.update(user.id, {
      refreshToken,
      lastLogin: new Date(),
    });

    const { password, ...userWithoutPassword } = user;

    return {
      user: userWithoutPassword,
      accessToken,
      refreshToken,
    };
  }

  async disable2FA(userId: string, code: string): Promise<void> {
    const user = await this.userRepository.findById(userId);
    
    if (!user || !user.twoFactorEnabled) {
      throw new Error('2FA not enabled');
    }

    const verified = speakeasy.totp.verify({
      secret: user.twoFactorSecret,
      encoding: 'base32',
      token: code,
      window: 2,
    });

    if (!verified) {
      throw new Error('Invalid 2FA code');
    }

    await this.userRepository.update(userId, {
      twoFactorEnabled: false,
      twoFactorSecret: null,
    });
  }

  private generateAccessToken(user: any): string {
    return jwt.sign(
      {
        id: user.id,
        email: user.email,
        role: user.role || 'user',
        permissions: user.permissions || [],
      },
      config.jwt.secret,
      { expiresIn: config.jwt.expiresIn }
    );
  }

  private generateRefreshToken(user: any): string {
    return jwt.sign(
      { id: user.id },
      config.jwt.secret,
      { expiresIn: '7d' }
    );
  }
}
