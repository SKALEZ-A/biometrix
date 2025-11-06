import { Request, Response, NextFunction } from 'express';
import { createHash } from 'crypto';
import { RedisClient } from '../../../packages/shared/src/cache/redis';

interface ApiKeyMetadata {
  id: string;
  name: string;
  permissions: string[];
  rateLimit: number;
  expiresAt?: Date;
  createdAt: Date;
  lastUsedAt?: Date;
}

export class ApiKeyValidator {
  private redis: RedisClient;
  private keyPrefix = 'apikey:';

  constructor() {
    this.redis = new RedisClient();
  }

  private hashApiKey(apiKey: string): string {
    return createHash('sha256').update(apiKey).digest('hex');
  }

  public async validateApiKey(apiKey: string): Promise<ApiKeyMetadata | null> {
    const hashedKey = this.hashApiKey(apiKey);
    const metadataJson = await this.redis.get(`${this.keyPrefix}${hashedKey}`);
    
    if (!metadataJson) {
      return null;
    }

    const metadata: ApiKeyMetadata = JSON.parse(metadataJson);

    if (metadata.expiresAt && new Date(metadata.expiresAt) < new Date()) {
      await this.revokeApiKey(apiKey);
      return null;
    }

    metadata.lastUsedAt = new Date();
    await this.redis.set(
      `${this.keyPrefix}${hashedKey}`,
      JSON.stringify(metadata),
      3600 * 24 * 365
    );

    return metadata;
  }

  public middleware() {
    return async (req: Request, res: Response, next: NextFunction) => {
      const apiKey = req.headers['x-api-key'] as string;

      if (!apiKey) {
        res.status(401).json({
          error: 'Unauthorized',
          message: 'API key is required'
        });
        return;
      }

      const metadata = await this.validateApiKey(apiKey);

      if (!metadata) {
        res.status(401).json({
          error: 'Unauthorized',
          message: 'Invalid or expired API key'
        });
        return;
      }

      (req as any).apiKey = metadata;
      next();
    };
  }

  public async createApiKey(metadata: Omit<ApiKeyMetadata, 'createdAt'>): Promise<string> {
    const apiKey = this.generateApiKey();
    const hashedKey = this.hashApiKey(apiKey);
    
    const fullMetadata: ApiKeyMetadata = {
      ...metadata,
      createdAt: new Date()
    };

    await this.redis.set(
      `${this.keyPrefix}${hashedKey}`,
      JSON.stringify(fullMetadata),
      3600 * 24 * 365
    );

    return apiKey;
  }

  public async revokeApiKey(apiKey: string): Promise<void> {
    const hashedKey = this.hashApiKey(apiKey);
    await this.redis.del(`${this.keyPrefix}${hashedKey}`);
  }

  private generateApiKey(): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let key = 'sk_';
    for (let i = 0; i < 48; i++) {
      key += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return key;
  }

  public async checkPermission(apiKey: string, permission: string): Promise<boolean> {
    const metadata = await this.validateApiKey(apiKey);
    return metadata ? metadata.permissions.includes(permission) : false;
  }
}

export const apiKeyValidator = new ApiKeyValidator();
