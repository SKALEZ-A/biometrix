import * as crypto from 'crypto';

export class EncryptionService {
  private static readonly ALGORITHM = 'aes-256-gcm';
  private static readonly KEY_LENGTH = 32;
  private static readonly IV_LENGTH = 16;
  private static readonly AUTH_TAG_LENGTH = 16;
  private static readonly SALT_LENGTH = 64;
  private static readonly ITERATIONS = 100000;

  static generateKey(): string {
    return crypto.randomBytes(this.KEY_LENGTH).toString('hex');
  }

  static generateSalt(): string {
    return crypto.randomBytes(this.SALT_LENGTH).toString('hex');
  }

  static deriveKey(password: string, salt: string): Buffer {
    return crypto.pbkdf2Sync(
      password,
      Buffer.from(salt, 'hex'),
      this.ITERATIONS,
      this.KEY_LENGTH,
      'sha512'
    );
  }

  static encrypt(data: string, key: string): {
    encrypted: string;
    iv: string;
    authTag: string;
  } {
    const iv = crypto.randomBytes(this.IV_LENGTH);
    const keyBuffer = Buffer.from(key, 'hex');
    
    const cipher = crypto.createCipheriv(this.ALGORITHM, keyBuffer, iv);
    
    let encrypted = cipher.update(data, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const authTag = cipher.getAuthTag();
    
    return {
      encrypted,
      iv: iv.toString('hex'),
      authTag: authTag.toString('hex')
    };
  }

  static decrypt(
    encrypted: string,
    key: string,
    iv: string,
    authTag: string
  ): string {
    const keyBuffer = Buffer.from(key, 'hex');
    const ivBuffer = Buffer.from(iv, 'hex');
    const authTagBuffer = Buffer.from(authTag, 'hex');
    
    const decipher = crypto.createDecipheriv(this.ALGORITHM, keyBuffer, ivBuffer);
    decipher.setAuthTag(authTagBuffer);
    
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  }

  static encryptObject<T>(obj: T, key: string): {
    encrypted: string;
    iv: string;
    authTag: string;
  } {
    const jsonString = JSON.stringify(obj);
    return this.encrypt(jsonString, key);
  }

  static decryptObject<T>(
    encrypted: string,
    key: string,
    iv: string,
    authTag: string
  ): T {
    const decrypted = this.decrypt(encrypted, key, iv, authTag);
    return JSON.parse(decrypted);
  }

  static hash(data: string, algorithm: string = 'sha256'): string {
    return crypto.createHash(algorithm).update(data).digest('hex');
  }

  static hmac(data: string, key: string, algorithm: string = 'sha256'): string {
    return crypto.createHmac(algorithm, key).update(data).digest('hex');
  }

  static generateKeyPair(): {
    publicKey: string;
    privateKey: string;
  } {
    const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', {
      modulusLength: 4096,
      publicKeyEncoding: {
        type: 'spki',
        format: 'pem'
      },
      privateKeyEncoding: {
        type: 'pkcs8',
        format: 'pem'
      }
    });
    
    return { publicKey, privateKey };
  }

  static encryptWithPublicKey(data: string, publicKey: string): string {
    const buffer = Buffer.from(data, 'utf8');
    const encrypted = crypto.publicEncrypt(
      {
        key: publicKey,
        padding: crypto.constants.RSA_PKCS1_OAEP_PADDING,
        oaepHash: 'sha256'
      },
      buffer
    );
    return encrypted.toString('base64');
  }

  static decryptWithPrivateKey(encrypted: string, privateKey: string): string {
    const buffer = Buffer.from(encrypted, 'base64');
    const decrypted = crypto.privateDecrypt(
      {
        key: privateKey,
        padding: crypto.constants.RSA_PKCS1_OAEP_PADDING,
        oaepHash: 'sha256'
      },
      buffer
    );
    return decrypted.toString('utf8');
  }

  static sign(data: string, privateKey: string): string {
    const sign = crypto.createSign('RSA-SHA256');
    sign.update(data);
    sign.end();
    return sign.sign(privateKey, 'base64');
  }

  static verify(data: string, signature: string, publicKey: string): boolean {
    const verify = crypto.createVerify('RSA-SHA256');
    verify.update(data);
    verify.end();
    return verify.verify(publicKey, signature, 'base64');
  }

  static generateNonce(length: number = 32): string {
    return crypto.randomBytes(length).toString('hex');
  }

  static constantTimeCompare(a: string, b: string): boolean {
    if (a.length !== b.length) return false;
    return crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b));
  }
}

export class HomomorphicEncryption {
  static addEncrypted(a: bigint, b: bigint, modulus: bigint): bigint {
    return (a * b) % modulus;
  }

  static multiplyByConstant(encrypted: bigint, constant: bigint, modulus: bigint): bigint {
    return this.modPow(encrypted, constant, modulus);
  }

  private static modPow(base: bigint, exponent: bigint, modulus: bigint): bigint {
    if (modulus === 1n) return 0n;
    let result = 1n;
    base = base % modulus;
    while (exponent > 0n) {
      if (exponent % 2n === 1n) {
        result = (result * base) % modulus;
      }
      exponent = exponent / 2n;
      base = (base * base) % modulus;
    }
    return result;
  }
}

export class ZeroKnowledgeProof {
  static generateProof(secret: string, challenge: string): {
    commitment: string;
    response: string;
  } {
    const secretHash = crypto.createHash('sha256').update(secret).digest('hex');
    const nonce = crypto.randomBytes(32).toString('hex');
    
    const commitment = crypto
      .createHash('sha256')
      .update(secretHash + nonce)
      .digest('hex');
    
    const response = crypto
      .createHash('sha256')
      .update(nonce + challenge)
      .digest('hex');
    
    return { commitment, response };
  }

  static verifyProof(
    commitment: string,
    response: string,
    challenge: string,
    publicValue: string
  ): boolean {
    const reconstructed = crypto
      .createHash('sha256')
      .update(response + challenge)
      .digest('hex');
    
    return reconstructed === commitment;
  }
}
