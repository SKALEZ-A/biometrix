import crypto from 'crypto';
import { logger } from './logger';

export class EncryptionUtils {
  private static readonly ALGORITHM = 'aes-256-gcm';
  private static readonly KEY_LENGTH = 32;
  private static readonly IV_LENGTH = 16;
  private static readonly AUTH_TAG_LENGTH = 16;
  private static readonly SALT_LENGTH = 64;

  public static generateKey(): string {
    return crypto.randomBytes(this.KEY_LENGTH).toString('hex');
  }

  public static generateSalt(): string {
    return crypto.randomBytes(this.SALT_LENGTH).toString('hex');
  }

  public static deriveKey(password: string, salt: string): Buffer {
    return crypto.pbkdf2Sync(password, salt, 100000, this.KEY_LENGTH, 'sha512');
  }

  public static encrypt(data: string, key: string): {
    encrypted: string;
    iv: string;
    authTag: string;
  } {
    try {
      const iv = crypto.randomBytes(this.IV_LENGTH);
      const keyBuffer = Buffer.from(key, 'hex');
      const cipher = crypto.createCipheriv(this.ALGORITHM, keyBuffer, iv);

      let encrypted = cipher.update(data, 'utf8', 'hex');
      encrypted += cipher.final('hex');

      const authTag = cipher.getAuthTag();

      return {
        encrypted,
        iv: iv.toString('hex'),
        authTag: authTag.toString('hex'),
      };
    } catch (error) {
      logger.error('Encryption error', error);
      throw new Error('Encryption failed');
    }
  }

  public static decrypt(
    encrypted: string,
    key: string,
    iv: string,
    authTag: string
  ): string {
    try {
      const keyBuffer = Buffer.from(key, 'hex');
      const ivBuffer = Buffer.from(iv, 'hex');
      const authTagBuffer = Buffer.from(authTag, 'hex');

      const decipher = crypto.createDecipheriv(this.ALGORITHM, keyBuffer, ivBuffer);
      decipher.setAuthTag(authTagBuffer);

      let decrypted = decipher.update(encrypted, 'hex', 'utf8');
      decrypted += decipher.final('utf8');

      return decrypted;
    } catch (error) {
      logger.error('Decryption error', error);
      throw new Error('Decryption failed');
    }
  }

  public static hash(data: string, algorithm: string = 'sha256'): string {
    return crypto.createHash(algorithm).update(data).digest('hex');
  }

  public static hmac(data: string, key: string, algorithm: string = 'sha256'): string {
    return crypto.createHmac(algorithm, key).update(data).digest('hex');
  }

  public static generateRandomToken(length: number = 32): string {
    return crypto.randomBytes(length).toString('hex');
  }

  public static constantTimeCompare(a: string, b: string): boolean {
    if (a.length !== b.length) return false;
    return crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b));
  }

  public static encryptObject<T>(obj: T, key: string): string {
    const jsonString = JSON.stringify(obj);
    const { encrypted, iv, authTag } = this.encrypt(jsonString, key);
    return JSON.stringify({ encrypted, iv, authTag });
  }

  public static decryptObject<T>(encryptedData: string, key: string): T {
    const { encrypted, iv, authTag } = JSON.parse(encryptedData);
    const decrypted = this.decrypt(encrypted, key, iv, authTag);
    return JSON.parse(decrypted);
  }
}
