import { ObjectId } from 'mongodb';

export interface IUser {
  _id?: ObjectId;
  userId: string;
  email: string;
  passwordHash: string;
  firstName: string;
  lastName: string;
  phoneNumber?: string;
  roles: string[];
  permissions: string[];
  isActive: boolean;
  isVerified: boolean;
  emailVerificationToken?: string;
  passwordResetToken?: string;
  passwordResetExpires?: Date;
  lastLogin?: Date;
  loginAttempts: number;
  lockUntil?: Date;
  twoFactorEnabled: boolean;
  twoFactorSecret?: string;
  biometricEnabled: boolean;
  biometricIds: string[];
  preferences: UserPreferences;
  metadata: UserMetadata;
  createdAt: Date;
  updatedAt: Date;
}

export interface UserPreferences {
  language: string;
  timezone: string;
  notifications: {
    email: boolean;
    sms: boolean;
    push: boolean;
  };
  theme: 'light' | 'dark' | 'auto';
}

export interface UserMetadata {
  registrationIp: string;
  lastLoginIp?: string;
  deviceFingerprints: string[];
  securityQuestions?: SecurityQuestion[];
  kycStatus: 'pending' | 'verified' | 'rejected';
  riskScore: number;
}

export interface SecurityQuestion {
  question: string;
  answerHash: string;
}

export class User implements IUser {
  _id?: ObjectId;
  userId: string;
  email: string;
  passwordHash: string;
  firstName: string;
  lastName: string;
  phoneNumber?: string;
  roles: string[];
  permissions: string[];
  isActive: boolean;
  isVerified: boolean;
  emailVerificationToken?: string;
  passwordResetToken?: string;
  passwordResetExpires?: Date;
  lastLogin?: Date;
  loginAttempts: number;
  lockUntil?: Date;
  twoFactorEnabled: boolean;
  twoFactorSecret?: string;
  biometricEnabled: boolean;
  biometricIds: string[];
  preferences: UserPreferences;
  metadata: UserMetadata;
  createdAt: Date;
  updatedAt: Date;

  constructor(data: Partial<IUser>) {
    this._id = data._id;
    this.userId = data.userId || this.generateUserId();
    this.email = data.email || '';
    this.passwordHash = data.passwordHash || '';
    this.firstName = data.firstName || '';
    this.lastName = data.lastName || '';
    this.phoneNumber = data.phoneNumber;
    this.roles = data.roles || ['user'];
    this.permissions = data.permissions || [];
    this.isActive = data.isActive !== undefined ? data.isActive : true;
    this.isVerified = data.isVerified || false;
    this.emailVerificationToken = data.emailVerificationToken;
    this.passwordResetToken = data.passwordResetToken;
    this.passwordResetExpires = data.passwordResetExpires;
    this.lastLogin = data.lastLogin;
    this.loginAttempts = data.loginAttempts || 0;
    this.lockUntil = data.lockUntil;
    this.twoFactorEnabled = data.twoFactorEnabled || false;
    this.twoFactorSecret = data.twoFactorSecret;
    this.biometricEnabled = data.biometricEnabled || false;
    this.biometricIds = data.biometricIds || [];
    this.preferences = data.preferences || this.getDefaultPreferences();
    this.metadata = data.metadata || this.getDefaultMetadata();
    this.createdAt = data.createdAt || new Date();
    this.updatedAt = data.updatedAt || new Date();
  }

  private generateUserId(): string {
    return `USR-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private getDefaultPreferences(): UserPreferences {
    return {
      language: 'en',
      timezone: 'UTC',
      notifications: {
        email: true,
        sms: false,
        push: true,
      },
      theme: 'auto',
    };
  }

  private getDefaultMetadata(): UserMetadata {
    return {
      registrationIp: '',
      deviceFingerprints: [],
      kycStatus: 'pending',
      riskScore: 0,
    };
  }

  isLocked(): boolean {
    return !!(this.lockUntil && this.lockUntil > new Date());
  }

  incrementLoginAttempts(): void {
    this.loginAttempts += 1;
    if (this.loginAttempts >= 5) {
      this.lockUntil = new Date(Date.now() + 30 * 60 * 1000); // 30 minutes
    }
  }

  resetLoginAttempts(): void {
    this.loginAttempts = 0;
    this.lockUntil = undefined;
  }

  getFullName(): string {
    return `${this.firstName} ${this.lastName}`.trim();
  }

  hasRole(role: string): boolean {
    return this.roles.includes(role);
  }

  hasPermission(permission: string): boolean {
    return this.permissions.includes(permission);
  }

  addRole(role: string): void {
    if (!this.roles.includes(role)) {
      this.roles.push(role);
    }
  }

  removeRole(role: string): void {
    this.roles = this.roles.filter(r => r !== role);
  }

  addPermission(permission: string): void {
    if (!this.permissions.includes(permission)) {
      this.permissions.push(permission);
    }
  }

  removePermission(permission: string): void {
    this.permissions = this.permissions.filter(p => p !== permission);
  }

  toJSON(): Partial<IUser> {
    const { passwordHash, twoFactorSecret, emailVerificationToken, passwordResetToken, ...rest } = this;
    return rest;
  }
}
